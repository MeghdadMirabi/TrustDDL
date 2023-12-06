import ray

import configuration as cfg
import util
import util.constants
from actors.abstract import Collector
from evaluation import CommunicationEvaluator
from evaluation.util import get_tensor_size
from fixedpoint.fp_util import to_float
from neural_network import DistributedNeuralNetwork
from neural_network.distributed.layers import Convolution, FullyConnected
from secret_sharing.additive_secret_sharing import create_shares

cpu_res = cfg.threads_available / util.constants.total_nodes


@ray.remote(num_cpus=(cpu_res if cpu_res < 1 else int(cpu_res)))
class ModelOwner:

    def __init__(self, num_groups: int, exp, empty_like, abs, random_fct):
        self.comm_evaluator = CommunicationEvaluator()
        self.mo_collector = Collector(util.constants.group_sizes, self.comm_evaluator)
        self.num_groups = num_groups

        self._exp = exp
        self._empty_like = empty_like
        self._abs = abs
        self._random_fct = random_fct
        self.auxiliary_tensors = {}
        self.auxiliary_cmp_num = None

    def create_models(self, nn: DistributedNeuralNetwork, num_groups: int):
        """
        Function to create a list of :class:`neural_network.DistributedNeuralNetwork`, one for each group of
        :class:`actors.WorkerNode`.

        Duplicates the layers of the given :class:`neural_network.DistributedNeuralNetwork`. The weights of each layer
        are split into additive secret shares and used in the :class:`neural_network.DistributedNeuralNetwork` created for
        each group.

        Parameters
        ----------
        nn:
            A :class:`neural_network.DistributedNeuralNetwork` with initialized layers.
        num_groups:
            The number of groups used, i.e. the number of duplicates and secret shares to create.

        Returns
        -------
        models: list of :class:`neural_network.DistributedNeuralNetwork`
            A list of :class:`neural_network.DistributedNeuralNetwork` containing one network for each group (i.e. a total
            of ``num_groups`` elements). Each created network uses additive secret shares in the trainable layers
            (:class:`neural_network.distributed.layers.FullyConnected`, :class:`neural_network.distributed.layers.Convolution`),
            created from the parameters used in the passed network ``nn``.
        """

        params = []

        for layer in nn.layers:
            if type(layer) == Convolution:
                F_shares = create_shares(layer.F, num_groups)
                b_shares = create_shares(layer.b, num_groups)
                params.append((F_shares, b_shares))
            elif type(layer) == FullyConnected:
                W_shares = create_shares(layer.W, num_groups)
                b_shares = create_shares(layer.b, num_groups)
                params.append((W_shares, b_shares))
            else:
                params.append(None)

        models = []
        for replica in range(util.constants.replication_factor):
            replica_models = []
            for group_number in range(num_groups):
                new_nn = DistributedNeuralNetwork(nn.auxiliary_nodes, group_number, replica, nn.ran_group,
                                                  byzantine=util.constants.byzantine_setup['worker'][group_number][replica],
                                                  lr=nn.lr)
                share_ids = util.constants.share_placement[replica][group_number]
                for i, layer in enumerate(nn.layers):
                    clone = layer.clone(group_number, replica, new_nn.byzantine)
                    if type(layer) == Convolution:
                        F_shares, b_shares = params[i]
                        clone.F = [F_shares[share_id][group_number] for share_id in share_ids]
                        clone.b = [b_shares[share_id][group_number] for share_id in share_ids]
                    elif type(layer) == FullyConnected:
                        W_shares, b_shares = params[i]
                        clone.W = [W_shares[share_id][group_number] for share_id in share_ids]
                        clone.b = [b_shares[share_id][group_number] for share_id in share_ids]
                    new_nn.layers.append(clone)
                replica_models.append(new_nn)
            models.append(replica_models)

        return models

    async def commit(self, group_num: int, node_id, iteration, commit_values, share_ids, batch=False):
        return await self._generic_support(commit_values, share_ids, group_num, node_id, iteration, batch, commit_support=True)

    async def support_softmax(self, group_num: int, node_id, iteration, x_i, share_ids, batch=False, commit=True):
        """
        Collects the shares ``x_i``, masked shares of the output of some layer, of all groups i. Consensus confirmation
        is performed for each group once every share has been received. The masked output is reconstructed from the
        consensus-confirmed shares and sent to the model owner.

        The model owner will unmask the output, calculate the softmax activation and send shares of this result back to
        the mediator nodes. Each mediator node will respond to the worker node with the share belonging to group ``group_num``.

        Parameters
        ----------
        group_num:
            Group number of the node calling this method.
        x_i:
            Masked share of the output of some layer or a list of the like if ``batch==True``.
        batch:
            Toggle between single (False) and batched (True) support. If ``batch==True``, ``x_i`` has to be
            a list and the result will be list of support results of the elements of ``x_i``.

        Returns
        -------
        result:
            A share of the softmax activation, belonging to group ``group_num``, of the reconstructed layer output.
        """

        if not group_num == 0:
            if batch:
                msg_size = len(x_i) * len(x_i[0]) * get_tensor_size(x_i[0][0])
            else:
                msg_size = len(x_i) * get_tensor_size(x_i[0])

            self.comm_evaluator.msgs += 1
            self.comm_evaluator.msg_size += msg_size

        return await self._generic_support(x_i, share_ids, group_num, node_id, iteration, batch, commit)

    def get_cmp_auxiliary_share(self, group_num: int, share_ids):
        """
        Function to provide auxiliary shares for a secure comparison (additive secret shares of a positive number).

        Parameters
        ----------
        group_num:
            Group number of the node calling this method.

        Returns
        -------
        share_random_pos:
            An additive secret share, belonging to group ``group_num``, of a random positive number.
        """

        if self.auxiliary_cmp_num is None:
            rand_pos_num = self._abs(self._random_fct(1))
            shares = create_shares(rand_pos_num, self.num_groups)
            self.auxiliary_cmp_num = shares

        response_shares = [self.auxiliary_cmp_num[share_id][group_num] for share_id in share_ids]

        return response_shares

    def get_mul_auxiliary_shares(self, operator, group_num: int, x_dim, y_dim, share_ids):
        if (x_dim, y_dim) not in self.auxiliary_tensors.keys():
            shares = self._create_auxiliary_shares(operator, x_dim, y_dim)
            self.auxiliary_tensors[(x_dim, y_dim)] = shares

        response_shares = []
        for share_id in share_ids:
            a_shares, b_shares, c_shares = self.auxiliary_tensors[(x_dim, y_dim)][share_id]
            response_shares.append((a_shares[group_num], b_shares[group_num], c_shares[group_num]))

        return response_shares

    def _create_auxiliary_shares(self, operator, X_dim, Y_dim):
        """
        Creates auxiliary secret shares (of a beaver triple) to perform multiplication / matrix multiplication.
        """

        A = self._random_fct(X_dim)
        B = self._random_fct(Y_dim)
        C = operator(A, B, fpr=False)

        A_shares = create_shares(A, self.num_groups)
        B_shares = create_shares(B, self.num_groups)
        C_shares = create_shares(C, self.num_groups)

        return [t for t in zip(A_shares, B_shares, C_shares)]

    async def _generic_support(self, shares, share_ids, group_num: int, node_id, iteration, batch=False, commit=True, commit_support=False):

        await self.mo_collector.reset_or_wait(iteration)

        if commit_support:
            await self.mo_collector.save_commits(shares, group_num, node_id)
        else:
            await self.mo_collector.save_shares(shares, group_num, node_id, share_ids, batch, commit)

        ready = await self.mo_collector.check_ready()

        if ready:
            if not commit_support:
                res = await self._reconstruct_softmax(batch)
            else:
                res = None

            await self.mo_collector.sec_result.put(res)

        results = await self.mo_collector.await_result()

        if commit_support:
            await self.mo_collector.optional_reset(commit_support)
            return results

        share_ids = util.constants.share_placement[node_id][group_num]
        if batch:
            worker_result_shares = [[_result[share_id][group_num] for share_id in share_ids] for _result in results]
        else:
            worker_result_shares = [results[share_id][group_num] for share_id in share_ids]

        await self.mo_collector.optional_reset(commit)

        return worker_result_shares

    async def _reconstruct_softmax(self, batch):
        if batch:
            output_neurons = self.mo_collector.generic_single_reconstruct_batch()
            result = []
            for _out in output_neurons:
                _out = to_float(_out)
                s = self._exp(_out).reshape(-1).sum(0)
                softmax = self._exp(_out) / s
                result.append(create_shares(softmax, self.num_groups))
        else:
            output_neurons = self.mo_collector.generic_single_reconstruct()
            output_neurons = to_float(output_neurons)
            s = self._exp(output_neurons).reshape(-1).sum(0)
            softmax = self._exp(output_neurons) / s
            result = create_shares(softmax, self.num_groups)

        return result

    def get_comm_cost(self):
        return self.comm_evaluator.msgs, self.comm_evaluator.msg_size
