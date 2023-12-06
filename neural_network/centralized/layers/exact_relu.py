from neural_network.centralized.layers import Layer


class ExactReLU(Layer):
    """
    A layer that applies the exact Rectified Linear Unit (ReLU) activation function.
    """

    def __init__(self, batch_size, lr, clone) -> None:
        """
        Parameters
        ----------
        batch_size: int
            The number of samples in a batch.
        lr: float
            The learning rate to be used for training the layer.
        clone: callable
            A function that clones the input array to prevent modifying the original data.
        """

        super().__init__(batch_size, lr)
        self._clone = clone
        self._input = None

    def forward(self, x):
        """
        Computes the forward pass of the ExactReLU layer.

        Parameters
        ----------
        x:
            The input tensor to be processed by the layer. If a list is provided, it is treated as a batch.

        Returns
        -------
        output:
            The output data after applying the ExactReLU activation function to the input data.
            If ``x`` is a list of tensors (batched input), ``output`` is a list of tensors with the same length as ``x``.
        """

        if type(x) == list:  # batch inference
            self._input = []
            for i in range(len(x)):
                x[i] = x[i].reshape(-1)
                self._input.append(self._clone(x[i]))
                x[i][x[i] < 0] = 0

        else:
            x = x.reshape(-1)
            self._input = self._clone(x)
            x[x < 0] = 0

        return x

    def backward(self, partials_prev):
        """
        Computes the backward pass of the ExactReLU layer.

        Parameters
        ----------
        partials_prev:
            The partial derivatives of the loss with respect to the output of this layer in the previous iteration
            of backpropagation. If a list is provided, it is treated as a batch.

        Returns
        -------
        new_partials:
            The partial derivatives of the loss with respect to the input data, computed by backpropagating the partial
            derivatives of the loss with respect to the output of this layer.
        """

        if type(partials_prev) == list:
            for i, (_in, _partials_prev) in enumerate(zip(self._input, partials_prev)):
                partials_prev[i] = _partials_prev.reshape(-1)
                partials_prev[i][_in < 0] = 0

        else:
            partials_prev = partials_prev.reshape(-1)
            partials_prev[self._input < 0] = 0

        return partials_prev
