from asyncio import Lock
from hashlib import sha256

import ray

import util
from evaluation import CommunicationEvaluator
from evaluation.util import get_tensor_size
from fixedpoint.fp_util import mul, matmul
from secret_sharing.additive_secret_sharing import random_share


class SecComp:
    """
    A class used to provide secure computations (multiplication, matrix multiplication, comparison) over additive secret shares.
    """

    _auxiliary_nodes: []
    _group_num: int

    def __init__(self, auxiliary_nodes, model_owner, group_num, node_id, worker_node, random_fct) -> None:
        """
        Parameters
        ----------
        auxiliary_nodes:
            A list of :class:`actors.AuxiliaryNode` ray stubs.
        model_owner:
            The :class:`actors.ModelOwner` ray stub.
        group_num:
            The group number to use for the secure computations.
        worker_node:
            The worker node using this object.
        """

        self.online_comm_evaluator = CommunicationEvaluator()

        self._auxiliary_nodes = auxiliary_nodes
        self._model_owner = model_owner
        self._group_num = group_num
        self.worker_node = worker_node
        self.node_id = node_id

        self._random_fct = random_fct

        # auxiliary matrices for secure (matrix) multiplication (for different tensor sizes)
        self.a_i = {}
        self.b_i = {}
        self.c_i = {}

        # auxiliary positive number for secure comparison
        self.t_i = None

        self.iteration_lock = Lock()

    async def sec_mul(self, x_i, y_i, share_ids, batch=False, commit=util.constants.commit_default):
        """
        Method to perform a secure multiplication with the shares ``x_i`` and ``y_i`` of some tensors `x` and `y`.

        This method uses the support function `actors.AuxiliaryNode.sec_mul_support` of all :class:`actors.AuxiliaryNode` stubs via RPC to obtain a share ``xy_i`` of `x*y`.

        Parameters
        ----------
        x_i:
            An additive share (tensor) of some other tensor `x` or a list of the like if ``batch==True``.
        y_i:
            An additive share (tensor) of some other tensor `y` or a list of the like if ``batch==True``.
        batch: bool
            Toggle between single (False) and batched (True) secure multiplications. If ``batch==True``, ``x_i`` and ``y_i`` have to be
            equal-sized lists and the result will be list of pair-wise secure multiplications of the elements of ``zip(x_i, y_i)``.
        byzantine: bool
            Whether the given shares are / contain byzantine values.

        Returns
        -------
        xy_i:
            An additive share (tensor) of `x*y` or a list of the like if ``batch==True``.
        """

        return await self._sec_mul(mul, x_i, y_i, share_ids, batch, commit)

    async def sec_mat_mul(self, X_i, Y_i, share_ids, batch=False, commit=util.constants.commit_default):
        """
        Function to perform a secure matrix multiplication with the shares ``X_i`` and ``Y_i`` of some tensors `X` and `Y`.

        This function uses the support function `actors.AuxiliaryNode.sec_mat_mul_support` of all :class:`actors.AuxiliaryNode` stubs via RPC to obtain a share ``XY_i`` of `X@Y`.

        Parameters
        ----------
        X_i:
            An additive share (tensor) of some other tensor `X` or a list of the like if ``batch==True``.
        Y_i:
            An additive share (tensor) of some other tensor `Y` or a list of the like if ``batch==True``.
        batch: bool
            Toggle between single (False) and batched (True) secure matrix multiplications. If ``batch==True``, ``X_i`` and ``Y_i`` have to be
            equal-sized lists and the result will be list of pair-wise secure matrix multiplications of the elements of ``zip(X_i, Y_i)``.
        byzantine: bool
            Whether the given shares are / contain byzantine values.

        Returns
        -------
        XY_i:
            An additive share (tensor) of `X@Y` or a list of the like if ``batch==True``.
        """

        return await self._sec_mul(matmul, X_i, Y_i, share_ids, batch, commit)

    async def _sec_mul(self, operator, x_i, y_i, share_ids, batch, commit=util.constants.commit_default):
        """
        Share format (x_i, y_i):
        - len(x_i) = len(y_i) = batch_size
        - len(x_i[*]) = len(y_i[*]) = len(share_ids)

        (No outer dimension for non-batched inputs)
        """

        if operator not in self.a_i.keys():
            self.a_i[operator] = {}
            self.b_i[operator] = {}
            self.c_i[operator] = {}

        if batch:
            shapes = {(x_i[j][0].shape, y_i[j][0].shape) for j in range(len(x_i))}

            for x_i_shape, y_i_shape in shapes:
                if (x_i_shape, y_i_shape) not in self.a_i[operator].keys():
                    await self.add_auxiliary_shares(operator, share_ids, x_i_shape, y_i_shape)

            # same share format as for x_i, y_i; inner-most element-type: tuples instead of tensors
            shares = []

            for _x_i, _y_i in zip(x_i, y_i):  # iterate over batch
                _x_i_shape, _y_i_shape = _x_i[0].shape, _y_i[0].shape

                a_i = self.a_i[operator][(_x_i_shape, _y_i_shape)]
                b_i = self.b_i[operator][(_x_i_shape, _y_i_shape)]
                c_i = self.c_i[operator][(_x_i_shape, _y_i_shape)]

                # add lists over assigned shares for each batch-element
                if self.worker_node.byzantine:
                    shares.append([(random_share(__x_i.shape, self._random_fct), random_share(__y_i.shape, self._random_fct)) for __x_i, __y_i in zip(_x_i, _y_i)])
                else:
                    shares.append([(__x_i - _a_i, __y_i - _b_i) for _a_i, _b_i, __x_i, __y_i in zip(a_i, b_i, _x_i, _y_i)])

        else:
            x_i_shape = x_i[0].shape
            y_i_shape = y_i[0].shape

            if (x_i_shape, y_i_shape) not in self.a_i[operator].keys():
                await self.add_auxiliary_shares(operator, share_ids, x_i_shape, y_i_shape)

            a_i = self.a_i[operator][(x_i_shape, y_i_shape)]
            b_i = self.b_i[operator][(x_i_shape, y_i_shape)]
            c_i = self.c_i[operator][(x_i_shape, y_i_shape)]

            if self.worker_node.byzantine:
                shares = [(random_share(_x_i.shape, self._random_fct), random_share(_y_i.shape, self._random_fct)) for _x_i, _y_i in zip(x_i, y_i)]
            else:
                shares = [(_x_i - _a_i, _y_i - _b_i) for _a_i, _b_i, _x_i, _y_i in zip(a_i, b_i, x_i, y_i)]

        async with self.iteration_lock:

            if commit:
                commit_value = sha256(str(shares).encode('utf-8')).hexdigest()

                ray.get([a.commit.remote(self._group_num,
                                         self.node_id,
                                         self.worker_node.iteration_auxiliaries,
                                         commit_value,
                                         share_ids,
                                         batch)
                         for a in self._auxiliary_nodes])

                self.worker_node.increase_auxiliary_iteration()

            results = ray.get([a.sec_mul_support.remote(operator,
                                                        self._group_num,
                                                        self.node_id,
                                                        self.worker_node.iteration_auxiliaries,
                                                        shares,
                                                        share_ids,
                                                        batch,
                                                        commit)
                               for a in self._auxiliary_nodes])

            self.worker_node.increase_auxiliary_iteration()

        e, f, g = results[0]

        if batch:
            xy_i = []
            for _e, _f, _g in zip(e, f, g):
                a_i = self.a_i[operator][(_e.shape, _f.shape)]
                b_i = self.b_i[operator][(_e.shape, _f.shape)]
                c_i = self.c_i[operator][(_e.shape, _f.shape)]

                _xy_i = [_c_i + operator(_e, _b_i) + operator(_a_i, _f) + (_g if self._group_num == 0 else 0) for
                         _a_i, _b_i, _c_i in zip(a_i, b_i, c_i)]

                xy_i.append(_xy_i)
        else:
            xy_i = [_c_i + operator(e, _b_i) + operator(_a_i, f) + (g if self._group_num == 0 else 0) for
                    _a_i, _b_i, _c_i in zip(a_i, b_i, c_i)]

        if batch:
            e_size = get_tensor_size(e[0]) * len(e)
            f_size = get_tensor_size(e[0]) * len(f)
        else:
            e_size = get_tensor_size(e)
            f_size = get_tensor_size(f)

        self.online_comm_evaluator.msgs += len(results) - 1
        self.online_comm_evaluator.msg_size += (len(results) - 1) * (e_size + f_size)

        return xy_i

    async def add_auxiliary_shares(self, operator, share_ids, x_i_shape, y_i_shape):
        auxiliary_shares = ray.get(self._model_owner.get_mul_auxiliary_shares.remote(operator, self._group_num, x_i_shape, y_i_shape, share_ids))
        self.a_i[operator][(x_i_shape, y_i_shape)] = []
        self.b_i[operator][(x_i_shape, y_i_shape)] = []
        self.c_i[operator][(x_i_shape, y_i_shape)] = []
        for _a_i, _b_i, _c_i in auxiliary_shares:
            self.a_i[operator][(x_i_shape, y_i_shape)].append(_a_i)
            self.b_i[operator][(x_i_shape, y_i_shape)].append(_b_i)
            self.c_i[operator][(x_i_shape, y_i_shape)].append(_c_i)

    async def sec_cmp(self, x_i, y_i, share_ids, batch=False, commit=util.constants.commit_default):
        """
        Function to perform a secure element-wise comparison with the shares ``x_i`` and ``y_i`` of some tensors `x` and `y`.

        This function uses the support function `actors.AuxiliaryNode.sec_comp_support` of all :class:`actors.AuxiliaryNode` stubs via RPC to obtain `t * alpha = t * (x - y)`.

        Parameters
        ----------
        x_i:
            An additive share (tensor) of some other tensor `x` or a list of the like if ``batch==True``.
        y_i:
            An additive share (tensor) of some other tensor `y` or a list of the like if ``batch==True``.
        batch: bool
            Toggle between single (False) and batched (True) secure comparisons. If ``batch==True``, ``x_i`` and ``y_i`` have to be
            equal-sized lists and the result will be list of pair-wise secure comparisons of the elements of ``zip(x_i, y_i)``.

        Returns
        -------
        t_alpha:
            An tensor `t * alpha = t * (x - y)` where `sign(t*alpha) = sign(x-y)` or a list of the like if ``batch==True``.
        """

        if self.t_i is None:
            self.t_i = ray.get(self._model_owner.get_cmp_auxiliary_share.remote(self._group_num, share_ids))

        if batch:
            alpha_i = [[__x_i - __y_i for __x_i, __y_i in zip(_x_i, _y_i)] for _x_i, _y_i in zip(x_i, y_i)]
            t_i = [self.t_i for _ in range(len(alpha_i))]
        else:
            alpha_i = [_x_i - _y_i for _x_i, _y_i in zip(x_i, y_i)]
            t_i = self.t_i

        t_alpha_i = await self.sec_mul(t_i, alpha_i, share_ids, batch, commit)

        if self.worker_node.byzantine:
            if batch:
                t_alpha_i = [[random_share(__t_alpha_i.shape, self._random_fct) for __t_alpha_i in _t_alpha_i] for _t_alpha_i in t_alpha_i]
            else:
                t_alpha_i = [random_share(_t_alpha_i.shape, self._random_fct) for _t_alpha_i in t_alpha_i]

        async with self.iteration_lock:

            if commit:
                commit_value = sha256(str(t_alpha_i).encode('utf-8')).hexdigest()

                ray.get([a.commit.remote(self._group_num,
                                         self.node_id,
                                         self.worker_node.iteration_auxiliaries,
                                         commit_value,
                                         share_ids,
                                         batch)
                         for a in self._auxiliary_nodes])

                self.worker_node.increase_auxiliary_iteration()

            results = ray.get([a.sec_comp_support.remote(self._group_num,
                                                         self.node_id,
                                                         self.worker_node.iteration_auxiliaries,
                                                         t_alpha_i,
                                                         share_ids,
                                                         batch,
                                                         commit)
                               for a in self._auxiliary_nodes])

            self.worker_node.increase_auxiliary_iteration()

        t_alpha = results[0]

        if batch:
            t_alpha_size = get_tensor_size(t_alpha[0]) * len(t_alpha)
        else:
            t_alpha_size = get_tensor_size(t_alpha)

        self.online_comm_evaluator.msgs += len(results) - 1
        self.online_comm_evaluator.msg_size += (len(results) - 1) * t_alpha_size

        return t_alpha

    async def sec_softmax(self, x, share_ids, batch=False, commit=util.constants.commit_default):

        async with self.iteration_lock:

            commit_value = sha256(str(x).encode('utf-8')).hexdigest()

            ray.get(self._model_owner.commit.remote(self._group_num,
                                                    self.node_id,
                                                    self.worker_node.iteration_mo,
                                                    commit_value,
                                                    share_ids,
                                                    batch)
                    )

            result = ray.get(self._model_owner.support_softmax.remote(self._group_num,
                                                                      self.node_id,
                                                                      self.worker_node.iteration_mo,
                                                                      x,
                                                                      share_ids,
                                                                      batch,
                                                                      commit)
                             )

        self.worker_node.increase_mo_iteration()

        softmax = result

        if batch:
            softmax_size = get_tensor_size(softmax[0][0]) * len(softmax[0]) * len(softmax)
        else:
            softmax_size = get_tensor_size(softmax[0])

        self.online_comm_evaluator.msgs += len(softmax) - 1
        self.online_comm_evaluator.msg_size += len(softmax) * softmax_size

        return softmax
