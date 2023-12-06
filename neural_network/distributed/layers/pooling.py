from fixedpoint.fp_util import div, to_config_representation
from neural_network.distributed.layers import Layer


class AvgPooling(Layer):
    """
    A layer that applies average pooling to the input.
    """

    def __init__(self, batch_size, lr, k, in_dim, output_dim, empty, empty_like, group_num, node_id, byzantine) -> None:
        """
        Parameters
        ----------
        batch_size: int
            The number of samples in a batch.
        lr: float
            The learning rate for the layer.
        k:
            The size of the square region to apply the average. Also used as the stride when sliding over the input.
        in_dim:
            The dimensions of the input.
        output_dim:
            The dimensions of the output.
        empty: callable
            A function that returns an empty tensor.
        empty_like: callable
            A function that returns an empty tensor like the input.
        """

        super().__init__(batch_size, lr, byzantine, group_num, node_id)
        self.in_dim = in_dim
        self._output = None
        self.output_dim = output_dim
        self.k = k
        self._empty = empty
        self._empty_like = empty_like
        self._input = None

    async def forward(self, X, batch):
        """
        Applies the average pooling operation to the input ``X`` using the configured kernel size ``k`` and stride ``k``.
        Returns the result of the convolution plus the bias.

        Parameters
        ----------
        X:
            A share of the 3D input tensor over which to apply the pooling operation. The input will not be padded.

        Returns
        -------
        conv_result:
            A share of the result of the pooling operation as a 3D tensor obtained by calculation the average of every 2D :math:`k \times k`
            region of the input data.
        """

        if batch:
            X = [[__X.reshape(self.in_dim) for __X in _X] for _X in X]
            self._output = [[self._empty(self.output_dim) for _ in _X] for _X in X]
        else:
            X = [_X.reshape(self.in_dim) for _X in X]
            self._output = [self._empty(self.output_dim) for _ in X]

        self._input = X

        for m in range(int(self.in_dim[1] / self.k)):
            for n in range(int(self.in_dim[2] / self.k)):
                if batch:
                    for i, _X in enumerate(X):
                        for j, __X in enumerate(_X):
                            s = __X[:, m * self.k:m * self.k + self.k, n * self.k:n * self.k + self.k].sum([1, 2])
                            self._output[i][j][:, m, n] = div(s, to_config_representation(self.k ** 2, scalar=True, value_type='int'))
                else:
                    for i, _X in enumerate(X):
                        s = _X[:, m * self.k:m * self.k + self.k, n * self.k:n * self.k + self.k].sum([1, 2])
                        self._output[i][:, m, n] = div(s, to_config_representation(self.k ** 2, scalar=True, value_type='int'))

        return self._output

    async def backward(self, partials_prev, batch):
        """
        Calculates and returns the partial derivatives of the loss with respect to the input of the layer.

        Parameters
        ----------
        partials_prev:
            A share of the partial derivatives of the loss with respect to the output.

        Returns
        -------
        new_partials:
            A share of the partial derivatives of the loss with respect to the input.
        """

        if batch:
            new_partials = [[self._empty(self.in_dim) for _ in _partials_prev] for _partials_prev in partials_prev]
            partials_prev = [[__partials_prev.reshape(self.output_dim) for __partials_prev in _partials_prev] for _partials_prev in partials_prev]
        else:
            new_partials = [self._empty(self.in_dim) for _ in partials_prev]
            partials_prev = [_partials_prev.reshape(self.output_dim) for _partials_prev in partials_prev]

        for i in range(self.in_dim[0]):
            for m in range(int(self.in_dim[1] / self.k)):
                for n in range(int(self.in_dim[2] / self.k)):
                    if batch:
                        for _partials_prev, _new_partials in zip(partials_prev, new_partials):
                            for __partials_prev, __new_partials in zip(_partials_prev, _new_partials):
                                __new_partials[i, m * self.k:(m + 1) * self.k, n * self.k:(n + 1) * self.k] = div(__partials_prev[i][m][n], to_config_representation(self.k ** 2, scalar=True, value_type='int'))
                    else:
                        for _partials_prev, _new_partials in zip(partials_prev, new_partials):
                            _new_partials[i, m * self.k:(m + 1) * self.k, n * self.k:(n + 1) * self.k] = div(_partials_prev[i][m][n], to_config_representation(self.k ** 2, scalar=True, value_type='int'))

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
            attributes (batch size, learning rate, ...) as this layer, except for the group number.
        """

        return AvgPooling(self._batch_size, self._lr, self.k, self.in_dim, self.output_dim, self._empty, self._empty_like, group_num, node_id, byzantine)
