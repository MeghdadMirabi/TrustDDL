from fixedpoint.fp_util import mul, to_config_representation, div
from neural_network.distributed.layers import Layer


class FullyConnected(Layer):
    """
    A fully connected layer of a neural network.
    """

    def __init__(self, batch_size, lr, W, b, calc_partials, empty_like, diag, zeros_like, group_num, node_id, byzantine) -> None:
        """
        Parameters
        ----------
        batch_size: int
            The size of the batches that will be processed.
        lr: float
            The learning rate of the layer.
        W:
            The weight tensor of the layer. Must be a 2D tensor.
        b:
            The bias tensor of the layer. Must be a 1D tensor.
        empty_like: callable
            A function that returns an empty tensor with the same shape and data type as the input tensor.
        diag: callable
            A function that returns a 2D tensor with the input ndarray on the diagonal.
        zeros_like: callable
            A function that returns a tensor of zeros with the same shape and data type as the input tensor.
        group_num: int
            The group number of the node using this network (used during secure computations).
        """

        super().__init__(batch_size, lr, byzantine, group_num, node_id)
        self.W = W  # weight (share) matrix
        self.b = b  # bias (share) matrix
        self._gradient = None
        self._empty_like = empty_like
        self._zeros_like = zeros_like
        self._diag = diag
        self._input = None
        self.calc_partials = calc_partials

    async def forward(self, x, batch):
        """
        Compute the forward pass of the fully connected layer.

        Parameters
        ----------
        x:
            A share of the input data for the forward pass. Might be a tensor or a list of tensors (batched input). The input
            tensor(s) can have any dimensions.

        Returns
        -------
        output:
            A share of the output of the forward pass. If ``x`` is a list of tensors (batched input),
            ``output`` is a list of tensors with the same length as ``x``.
        """

        if batch:
            x = [[__x.reshape(-1) for __x in _x] for _x in x]
            self._input = x
            W = [self.W for _ in range(len(x))]  # TODO: no replication
        else:
            x = [_x.reshape(-1) for _x in x]
            self._input = x
            W = self.W

        result = await self.sec_comp.sec_mat_mul(W, x, self.replicas_to_maintain, batch)

        if batch:
            for i in range(len(result)):
                result[i] += [_res + _b for _res, _b in zip(result[i], self.b)]
        else:
            result = [_res + _b for _res, _b in zip(result, self.b)]

        return result

    async def backward(self, partials_prev, batch):
        """
        Compute the backward pass of the fully connected layer.

        Parameters
        ----------
        partials_prev:
            A share of the partial derivatives of the loss with respect to the output of the layer. Might be a tensor or a list of
            tensors (batched backward pass). The given tensor(s) can have any dimensions.


        Returns
        -------
        new_partials:
            Shares of the partial derivatives of the loss with respect to the input of the layer. If ``partials_prev``
            is a list of tensors, ``new_partials`` is a list of tensors with the same length as ``partials_prev``.
        """

        if batch:
            partials_prev = [[__partials_prev.reshape(-1, 1) for __partials_prev in _partials_prev] for _partials_prev in partials_prev]
        else:
            partials_prev = [_partials_prev.reshape(-1, 1) for _partials_prev in partials_prev]

        W_update = await self.sec_comp.sec_mul(self._input, partials_prev, self.replicas_to_maintain, batch)

        if batch:
            W_update = [sum([_W_update[i] for _W_update in W_update]) for i in range(len(self.replicas_to_maintain))]
            b_update = [sum([_partials_prev[i].reshape(-1) for _partials_prev in partials_prev]) for i in range(len(self.replicas_to_maintain))]
        else:
            b_update = [_partials_prev.reshape(-1) for _partials_prev in partials_prev]

        if self.calc_partials:
            if batch:
                new_partials = await self.sec_comp.sec_mat_mul([[_W.T for _W in self.W] for _ in range(len(partials_prev))], partials_prev, self.replicas_to_maintain, batch)
            else:
                new_partials = await self.sec_comp.sec_mat_mul([_W.T for _W in self.W], partials_prev, self.replicas_to_maintain, batch)
        else:
            new_partials = None

        if batch:
            W_update = [div(mul(to_config_representation(self._lr, scalar=True, value_type='float'), _W_update), to_config_representation(len(partials_prev), scalar=True, value_type='int')) for _W_update in W_update]
            b_update = [div(mul(to_config_representation(self._lr, scalar=True, value_type='float'), _b_update), to_config_representation(len(partials_prev), scalar=True, value_type='int')) for _b_update in b_update]
        else:
            W_update = [mul(to_config_representation(self._lr, scalar=True, value_type='float'), _W_update) for _W_update in W_update]
            b_update = [mul(to_config_representation(self._lr, scalar=True, value_type='float'), _b_update) for _b_update in b_update]

        self.W = [_W - _W_update for _W, _W_update in zip(self.W, W_update)]
        self.b = [_b - _b_update for _b, _b_update in zip(self.b, b_update)]

        return new_partials

    def clone(self, group_num, node_id, byzantine):
        """
        Create a clone of this layer for the given group number ``group_num``.

        Parameters
        ----------
        group_num: int
            The group number to create the copy for.

        Returns
        -------
        clone:
            A clone of this layer for the given group number ``group_num``. The returned layer has the same
            attributes (batch size, learning rate, ...) as this layer, except for the group number. Secret-shared
            parameters (weights, biases) are not copied as they will be overwritten by other secret shares.
        """

        return FullyConnected(self._batch_size, self._lr, None, None, self.calc_partials, self._empty_like, self._diag, self._zeros_like, group_num, node_id, byzantine)
