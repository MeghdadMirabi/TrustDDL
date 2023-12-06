from neural_network.distributed.layers import Convolution, ExactReLU, FullyConnected, Layer, AvgPooling, Softmax
from util import tensor_function


class DistributedNeuralNetwork:
    """
    A neural network class that supports the addition of specific layers and provides methods for a forward
    and backward pass of a neural network based on additive secret sharing.
    """

    layers: [Layer]
    _batch_size: int

    def __init__(self, auxiliary_nodes, group_num, node_id, ran_group, byzantine=False, batch_size=100, lr=0.05, tensor_package='torch') -> None:
        """
        Initializes the DistributedNeuralNetwork object.

        Parameters
        ----------
        auxiliary_nodes:
            A list of :class:`actors.AuxiliaryNode` ray stubs.
        group_num:
            The group number of the node using this network (used during secure computations).
        ran_group: int
            A randomly selected group (used during secure computations).
        batch_size: int, optional
            The input batch size, by default 100.
        lr: float, optional
            The learning rate, by default 0.05.
        tensor_package: str, optional
            The tensor package used, by default 'torch'.
        """

        super().__init__()

        self._batch_size = batch_size
        self.lr = lr
        self.auxiliary_nodes = auxiliary_nodes
        self.group_num = group_num
        self.node_id = node_id
        self.ran_group = ran_group
        self.tensor_package = tensor_package
        self.byzantine = byzantine

        self.layers = []
        self._diag = tensor_function('diag')
        self._empty_like = tensor_function('empty_like')
        self._empty = tensor_function('empty')
        self._zeros = tensor_function('zeros')
        self._zeros_like = tensor_function('zeros_like')
        self._rand = tensor_function('randn')
        self._exp = tensor_function('exp')
        self._dot = tensor_function('dot')
        self._clone = tensor_function('clone')
        self._pad = tensor_function('pad')

    def add_conv(self, F_dim, in_dim, output_dim, padding, stride, calc_partials):
        """
        Adds a Convolution layer to the network.
    
        Parameters
        ----------
        F_dim:
            The dimensions of the filter.
        in_dim:
            The dimensions of the input tensor.
        output_dim:
            The dimensions of the output tensor.
        """
        
        F = self._rand(F_dim) / (F_dim[1] * F_dim[2])
        b = self._zeros(output_dim)
        self.layers.append(Convolution(self._batch_size, self.lr, F, b, F_dim, in_dim, output_dim, padding, stride, calc_partials, self._empty_like, self._empty, self._zeros_like, self._pad, self._dot, self.auxiliary_nodes, self.group_num, self.node_id, self.byzantine))

    def copy_conv(self, other_layer, clone):
        F = clone(other_layer.F)
        b = clone(other_layer.b)
        self.layers.append(Convolution(self._batch_size, self.lr, F, b, other_layer.F_dim, other_layer.in_dim, other_layer.output_dim, other_layer.padding, other_layer.stride, other_layer.calc_partials, self._empty_like, self._empty, self._zeros_like, self._pad, self._dot, self.auxiliary_nodes, self.group_num, self.node_id, self.byzantine))

    def add_exact_relu(self):
        """
        Adds an ExactReLU layer to the network.
        """
        self.layers.append(ExactReLU(self._batch_size, self.lr, self._clone, self._zeros_like, self.group_num, self.node_id, self.byzantine))

    def add_fully_connected(self, W_dim, calc_partials=True):
        """
        Adds a FullyConnected layer to the network.

        Parameters
        ----------
        W_dim:
            The dimensions of the weight matrix.
        """
        
        W = self._rand(W_dim) / W_dim[1]
        b = self._zeros(W_dim[0])
        self.layers.append(FullyConnected(self._batch_size, self.lr, W, b, calc_partials, self._empty_like, self._diag, self._zeros_like, self.group_num, self.node_id, self.byzantine))

    def copy_fully_connected(self, other_layer, clone):
        """
        Copies a FullyConnected layer to the network.

        Parameters
        ----------
        W:
            The weight matrix of the FullyConnected layer.
        b:
            The bias vector of the FullyConnected layer.
        clone: callable
            Function to use to clone tensors.
        """
        
        W = clone(other_layer.W)
        b = clone(other_layer.b)
        self.layers.append(FullyConnected(self._batch_size, self.lr, W, b, other_layer.calc_partials, self._empty_like, self._diag, self._zeros_like, self.auxiliary_nodes, self.group_num, self.byzantine))

    def add_avg_pooling(self, k, in_dim, output_dim):
        """
        Add an average pooling layer to the network.

        Parameters
        ----------
        k: int
            The size of the pooling window.
        in_dim:
            The input dimension of the layer.
        output_dim:
            The output dimension of the layer.
        """
        
        self.layers.append(AvgPooling(self._batch_size, self.lr, k, in_dim, output_dim, self._empty, self._empty_like, self.group_num, self.node_id, self.byzantine))

    def add_softmax(self):
        """
        Adds a softmax layer to the network.
        """
        
        self.layers.append(Softmax(self._batch_size, self.lr, self.group_num, self.node_id, self.byzantine))

    async def forward_pass(self, layer_input, batch):
        """
        Performs a forward pass through the network.

        Parameters
        ----------
        layer_input:
            The input data for the network. Its shape must match the input dimension of the first layer.

        Returns
        -------
        output:
            The output of the forward pass. If ``layer_input`` is a list of tensors (batched input), ``output`` is a list
            of tensors with the same length as ``output``.
        """

        for i, layer in enumerate(self.layers):
            layer_input = await layer.forward(layer_input, batch)

        return layer_input

    async def backward_pass(self, label, batch):
        """
        Perform a backward pass through the network to compute the gradients.

        Parameters
        ----------
        label:
            The label of the input data. Its shape must match the output dimension of the last layer.
        """

        partials = await self.layers[-1].backward(label, batch)

        for i, layer in enumerate(self.layers[-2::-1]):
            partials = await layer.backward(partials, batch)

    def update_parameters(self):
        """
        Update the parameters of all layers in the network using their accumulated gradients.
        """

        for layer in self.layers:
            layer.update_parameters()
