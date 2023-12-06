import ray

import configuration as cfg
import util
import util.constants
from actors.abstract import Collector, AbstractNode
from evaluation import CommunicationEvaluator
from evaluation.util import get_tensor_size

cpu_res = cfg.threads_available / util.constants.total_nodes


@ray.remote(num_cpus=(cpu_res if cpu_res < 1 else int(cpu_res)))
class AuxiliaryNode(AbstractNode):

    def __init__(self, group_sizes: list, node_id: int) -> None:
        self.comm_evaluator = CommunicationEvaluator()
        AbstractNode.__init__(self, node_id, util.constants.byzantine_setup['auxiliary'][node_id])
        # self.nodes is initialized to None and set via 'set_workers' to resolve circular dependency
        self.aux_collector = Collector(group_sizes, self.comm_evaluator)

    async def commit(self, group_num: int, node_id: int, iteration, commit_values, share_ids, batch=False):
        return await self._generic_support(commit_values, share_ids, group_num, node_id, iteration, None, batch, commit_support=True)

    async def sec_mul_support(self, operator, group_num: int, node_id: int, iteration, shares, share_ids, batch=False, commit=True):
        """
        Collects all additive shares ``e_i``, ``f_i`` from all groups i. ``e_i`` and ``f_i`` are masked shares of some
        tensors `x`, `y` to be multiplied. Consensus confirmation is performed for all ``e_i``, ``f_i`` received by the nodes
        of group i. `e`, `f` are reconstructed from all ``e_i``, ``f_i`` and `g = e * f` is calculated.

        Parameters
        ----------
        group_num:
            Group number of the node calling this method.
        e_i:
            Masked share of some multiplicand (tensor) `x` or a list of the like if ``batch==True``.
        f_i:
            Masked share of some multiplicand (tensor) `y` or a list of the like if ``batch==True``.
        batch:
            Toggle between single (False) and batched (True) support. If ``batch==True``, ``e_i`` and ``f_i`` have to be
            equal-sized lists and the result will be list of pair-wise support results of the elements of ``zip(e_i, f_i)``.
        byzantine: bool
            Whether the given shares are / contain byzantine values.

        Returns
        -------
        e, f, g:
            ``(e, f, g)`` where ``e`` is the masked tensor of multiplicand `x`, ``f`` is the masked tensor of multiplicand `y`
            and `g = e * f` or a list of the like if ``batch==True``.
        """

        if not node_id == 0:
            if batch:
                msg_size = len(shares) * len(shares[0]) * (get_tensor_size(shares[0][0][0]) + get_tensor_size(shares[0][0][1]))
            else:
                msg_size = len(shares) * (get_tensor_size(shares[0][0]) + get_tensor_size(shares[0][1]))

            self.comm_evaluator.msgs += 1
            self.comm_evaluator.msg_size += msg_size

        """
        Input format (e_i, f_i):
        - len(e_i) = len(f_i) = batch_size
        - len(e_i[*]) = len(f_i[*]) = len(share_ids)
        
        (No outer dimension for non-batched inputs)
        
        Expected input format @Collector.save_shares (single 'share' variable):
        - len(share) = len(share_ids)
        - len(share[*]) = batch_size
        """

        return await self._generic_support(shares, share_ids, group_num, node_id, iteration, self._reconstruct_mul(operator, batch), batch, commit)

    async def sec_comp_support(self, group_num: int, node_id: int, iteration, t_alpha_i, share_ids, batch=False, commit=True):
        """
        Collects all additive shares ``t_alpha_i`` from all groups i. ``t_alpha_i`` are masked shares of the difference
        `x-y` of some tensors `x` and `y` which are to be compared. Consensus confirmation is performed for all ``t_alpha_i``
        received by the nodes of group i. `t_alpha` is reconstructed from all `t_alpha_i`` which contains the information of
        `sign(x-y)`.

        Parameters
        ----------
        group_num:
            Group number of the node calling this method.
        t_alpha_i:
            Masked share of the difference `x-y` of some tensors `x` and `y` or a list of the like if ``batch==True``.
        batch:
            Toggle between single (False) and batched (True) support. If ``batch==True``, ``t_alpha_i`` has to be
            a list and the result will be list of support results of the elements of ``t_alpha_i``.
        byzantine: bool
            Whether the given share(s) are / contain byzantine values.

        Returns
        -------
        t_alpha:
            The masked tensor `t_alpha = t*(x-y)` of `x-y`, where `sign(t_alpha) = sign(x-y)`, or a list of the like if ``batch==True``.
        """

        if node_id == 0:
            if batch:
                msg_size = len(t_alpha_i) * len(t_alpha_i[0]) * get_tensor_size(t_alpha_i[0][0])
            else:
                msg_size = len(t_alpha_i) * get_tensor_size(t_alpha_i[0])

            self.comm_evaluator.msg_size += msg_size
            self.comm_evaluator.msgs += 1

        return await self._generic_support(t_alpha_i, share_ids, group_num, node_id, iteration, self._reconstruct_cmp(batch), batch, commit)

    def get_comm_cost(self):
        return self.comm_evaluator.msgs, self.comm_evaluator.msg_size

    async def _generic_support(self, shares, share_ids, group_num, node_id, iteration, reconstruction_fct, batch, commit=True, commit_support=False):
        """
        Generic support function to collect secret shares, reconstruct secrets and queue the reconstructed result.

        This function will block if not all shares of all groups were received to avoid callbacks.

        Parameters
        ----------
        share:
            An additive secret share of group ``group_num``. Shares of all groups will be saved in the ``sec_shares``
            dictionary inherited by :class:`actors.abstract.Collector`. If all shares of a group were collected, a consensus
            confirmation is performed for this group, whose result is placed in the ``sec_shares_cons`` dictionary.
            If the shares of `all` groups were collected (and consensus-confirmed), the secret is reconstructed via
            a callable reconstruction function (see below) and placed in the result queue ``sec_result``.
        group_num:
            The group number of the node calling a support function.
        reconstruction_fct:
            A callable function with no parameters. Reconstructs the collected secret from the consensus-confirmed shares
            stored in ``sec_shares_cons``.
        byzantine: bool
            Whether the given share(s) are / contain byzantine values.

        Returns
        -------
        result:
            The reconstructed secret.
        """

        # reset helper variables or wait until the previous result was received by all other nodes
        await self.aux_collector.reset_or_wait(iteration)

        if commit_support:
            await self.aux_collector.save_commits(shares, group_num, node_id)
        else:
            await self.aux_collector.save_shares(shares, group_num, node_id, share_ids, batch, commit)

        # check if all shares of all groups were received ('ready' condition)
        ready = await self.aux_collector.check_ready()

        # reconstruct and save results if ready
        if ready:
            res = reconstruction_fct() if not commit_support else None

            await self.aux_collector.sec_result.put(res)

        result = await self.aux_collector.await_result()

        await self.aux_collector.optional_reset(commit_support)

        return result

    def _reconstruct_mul(self, operator, batch):

        def _reconstruct_mul():
            """
            Reconstruct e, f, g=e*f from consensus-confirmed additive shares.
            """
            e, f = self.aux_collector.generic_double_reconstruct()
            return e, f, operator(e, f)

        def _reconstruct_mul_batch():
            """
            Reconstruct all e, f, g=e*f from consensus-confirmed additive shares.
            """
            e, f = self.aux_collector.generic_double_reconstruct_batch()
            g = [operator(e_elem, f_elem) for e_elem, f_elem in zip(e, f)]
            return e, f, g

        return _reconstruct_mul_batch if batch else _reconstruct_mul

    def _reconstruct_cmp(self, batch):

        def _reconstruct_cmp():
            """
            Reconstruct t*\alpha from consensus-confirmed additive shares.
            """
            return self.aux_collector.generic_single_reconstruct()

        def _reconstruct_cmp_batch():
            """
            Reconstruct t*\alpha from consensus-confirmed additive shares.
            """
            return self.aux_collector.generic_single_reconstruct_batch()

        return _reconstruct_cmp_batch if batch else _reconstruct_cmp
