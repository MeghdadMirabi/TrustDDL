class Layer:
    """
    Abstract base class for neural network layers.
    """

    _batch_size: int
    _lr: float
    iteration = 0

    def __init__(self, batch_size, lr):
        """
        Initializes the Layer object.

        Parameters
        ----------
        batch_size : int
            The size of the batches that will be passed through the network.
        lr : float
            The learning rate to use during training.
        """

        self._batch_size = batch_size
        self._lr = lr

    def forward(self, layer_input):
        """
        Performs a forward pass of the layer.

        Parameters
        ----------
        layer_input:
            The input to the layer, with shape (batch_size, ...).

        Returns
        -------
        layer_output:
            The output of the layer, with shape (batch_size, ...).
        """

        pass

    def backward(self, partials_prev):
        """
        Performs a backward pass of the layer.

        Parameters
        ----------
        partials_prev:
            The partial derivatives of the loss with respect to the output of the layer in the previous layer, with shape ``(batch_size, ...)``.

        Returns
        -------
        partials:
            The partial derivatives of the loss with respect to the output of the layer, with shape ``(batch_size, ...)``.
        """

        pass

    def update_parameters(self):
        pass
