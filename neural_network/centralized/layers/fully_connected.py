from neural_network.centralized.layers import Layer


class FullyConnected(Layer):
    """
    A fully connected layer of a neural network.
    """

    def __init__(self, batch_size, lr, W, b, calc_partials, empty_like, diag, zeros_like) -> None:
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
        """

        super().__init__(batch_size, lr)
        self.W = W  # weight tensor
        self.b = b  # bias tensor
        self._gradient = None
        self._empty_like = empty_like
        self._zeros_like = zeros_like
        self._diag = diag
        self._input = None
        self.calc_partials = calc_partials

    def forward(self, x):
        """
        Compute the forward pass of the fully connected layer.

        Parameters
        ----------
        x:
            The input data for the forward pass. Might be a tensor or a list of tensors (batched input). The input
            tensor(s) can have any dimensions.

        Returns
        -------
        output:
            The output of the forward pass. If ``x`` is a list of tensors (batched input),
            ``output`` is a list of tensors with the same length as ``x``.
        """

        if type(x) == list:
            self._input = []
            result = []
            for _x in x:
                _x = _x.reshape(-1)
                self._input.append(_x)
                result.append(self.W @ _x + self.b)

        else:
            x = x.reshape(-1)
            self._input = x
            result = self.W @ x + self.b

        return result

    def backward(self, partials_prev):
        """
        Compute the backward pass of the fully connected layer.

        Parameters
        ----------
        partials_prev:
            The partial derivatives of the loss with respect to the output of the layer. Might be a tensor or a list of
            tensors (batched backward pass). The given tensor(s) can have any dimensions.


        Returns
        -------
        new_partials:
            The partial derivatives of the loss with respect to the input of the layer. If ``partials_prev`` is a list
            of tensors, ``new_partials`` is a list of tensors with the same length as ``partials_prev``.
        """

        if type(partials_prev) == list:
            W_update, b_update = None, None
            update_init = False
            new_partials = []
            for _partials_prev, _in in zip(partials_prev, self._input):
                _partials_prev = _partials_prev.reshape(-1, 1)

                if not update_init:
                    W_update = _in * _partials_prev
                    b_update = _partials_prev.reshape(-1)
                    update_init = True
                else:
                    W_update += _in * _partials_prev
                    b_update += _partials_prev.reshape(-1)

                if self.calc_partials:
                    new_partials.append(self.W.T @ _partials_prev)

            self.W -= self._lr * W_update / len(partials_prev)
            self.b -= self._lr * b_update / len(partials_prev)

        else:
            partials_prev = partials_prev.reshape(-1, 1)

            W_update = self._input * partials_prev
            b_update = partials_prev.reshape(-1)

            if self.calc_partials:
                new_partials = self.W.T @ partials_prev

            self.W -= self._lr * W_update
            self.b -= self._lr * b_update

        return new_partials if self.calc_partials else None

    def update_parameters(self):
        """
        Update the parameters of the layer based on the accumulated gradients.
        """
        pass
