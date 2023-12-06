from neural_network.distributed.layers import Layer


class ExactReLU(Layer):
    """
    A layer that applies the exact Rectified Linear Unit (ReLU) activation function.
    """

    _group_num: int

    def __init__(self, batch_size, lr, clone, zeros_like, group_num, node_id, byzantine) -> None:
        """
        Parameters
        ----------
        batch_size: int
            The number of samples in a batch.
        lr: float
            The learning rate to be used for training the layer.
        clone: callable
            A function that clones the input array to prevent modifying the original data.
        zeros_like: callable
            A function that returns a tensor of zeros with the same shape and data type as the input tensor.
        group_num: int
            The group number of the node using this network (used during secure computations).
        """

        super().__init__(batch_size, lr, byzantine, group_num, node_id)
        self._clone = clone
        self._zeros_like = zeros_like
        self._input = None
        self._group_num = group_num
        self._input_signs = None

    async def forward(self, x, batch):
        """
        Computes the forward pass of the ExactReLU layer.

        Parameters
        ----------
        x:
            A share of the input tensor to be processed by the layer. If a list is provided, it is treated as a batch.

        Returns
        -------
        output:
            A share of the output data after applying the ExactReLU activation function to the input data.
            If ``x`` is a list of tensors (batched input), ``output`` is a list of tensors with the same length as ``x``.
        """

        if batch:
            x = [[__x.reshape(-1) for __x in _x] for _x in x]
            self._input = [[self._clone(__x) for __x in _x] for _x in x]
            y = [[self._zeros_like(__x) for __x in _x] for _x in x]
        else:
            x = [_x.reshape(-1) for _x in x]
            self._input = [self._clone(_x) for _x in x]
            y = [self._zeros_like(_x) for _x in x]

        self._input_signs = await self.sec_comp.sec_cmp(x, y, self.replicas_to_maintain, batch)

        if batch:
            for _x, _input_signs in zip(x, self._input_signs):
                for i in range(len(_x)):
                    _x[i][_input_signs < 0] = 0
        else:
            for i in range(len(x)):
                x[i][self._input_signs < 0] = 0

        return x

    async def backward(self, partials_prev, batch):
        """
        Computes the backward pass of the ExactReLU layer.

        Parameters
        ----------
        partials_prev:
            A share of the partial derivatives of the loss with respect to the output of this layer in the previous iteration
            of backpropagation. If a list is provided, it is treated as a batch.

        Returns
        -------
        new_partials:
            Shares of the partial derivatives of the loss with respect to the input data, computed by backpropagating the
            partial derivatives of the loss with respect to the output of this layer.
        """

        if batch:
            for _in_signs, _partials_prev in zip(self._input_signs, partials_prev):
                for i, __partials_prev in enumerate(_partials_prev):
                    _partials_prev[i] = __partials_prev.reshape(-1)
                    _partials_prev[i][_in_signs < 0] = 0

        else:
            for i, _partials_prev in enumerate(partials_prev):
                partials_prev[i] = _partials_prev.reshape(-1)
                partials_prev[i][self._input_signs < 0] = 0

        return partials_prev

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
            attributes (batch size, learning rate, ...) as this layer, except for the group number.
        """

        return ExactReLU(self._batch_size, self._lr, self._clone, self._zeros_like, group_num, node_id, byzantine)
