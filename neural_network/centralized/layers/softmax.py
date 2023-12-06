from neural_network.centralized.layers import Layer


class Softmax(Layer):
    """
    A layer that applies the softmax activation function (https://en.wikipedia.org/wiki/Softmax_function).
    """

    def __init__(self, batch_size, lr, exp) -> None:
        """
        Parameters
        ----------
        batch_size: int
            The number of samples in a batch.
        lr: float
            The learning rate to be used for training the layer.
        exp: callable
            A function that returns a tensor of the exponential values of another tensor.
        """

        super().__init__(batch_size, lr)
        self._exp = exp
        self._output = None

    def forward(self, x):
        """
        Computes the forward pass of the Softmax layer (https://en.wikipedia.org/wiki/Softmax_function).

        Parameters
        ----------
        x:
            The input tensor to be processed by the layer. If a list is provided, it is treated as a batch.

        Returns
        -------
        output:
            The output data after applying the softmax activation function to the input data.
            If ``x`` is a list of tensors (batched input), ``output`` is a list of tensors with the same length as ``x``.
        """

        if type(x) == list:
            self._output = []
            for _x in x:
                _x = _x.reshape(-1)
                _s = self._exp(_x).sum()
                self._output.append(self._exp(_x) / _s)

        else:
            x = x.reshape(-1)
            s = self._exp(x).sum()
            self._output = self._exp(x) / s

        return self._output

    def backward(self, label):
        """
        Computes the backward pass of the Softmax layer (https://en.wikipedia.org/wiki/Softmax_function).

        Parameters
        ----------
        label:
            The label or target output of the layer. If a list is provided, batch training is performed.

        Returns
        -------
        new_partials:
            The partial derivatives of the loss with respect to the input data.
        """

        if type(label) == list:
            result = []
            for _label, _out in zip(label, self._output):
                result.append(_out - _label)

        else:
            result = self._output - label

        return result
