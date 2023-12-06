from neural_network.centralized.layers import Convolution, ExactReLU, FullyConnected, Layer, AvgPooling, Softmax
from neural_network.centralized.layers.pooling import MaxPooling
from util import tensor_function


class CentralizedNeuralNetwork:
    """
    A neural network class that supports the addition of specific layers and provides methods for a forward
    and backward pass of a neural network.
    """

    layers: [Layer]
    _batch_size: int

    def __init__(self, batch_size, lr) -> None:
        """
        Initializes the CentralizedNeuralNetwork object.

        Parameters
        ----------
        batch_size: int, optional
            The input batch size, by default 100.
        lr: float, optional
            The learning rate, by default 0.05.
        """

        self._batch_size = batch_size
        self.lr = lr
        self.layers = []
        self._diag = tensor_function('diag', fpr=False)
        self._empty_like = tensor_function('empty_like', fpr=False)
        self._empty = tensor_function('empty', fpr=False)
        self._zeros = tensor_function('zeros', fpr=False)
        self._zeros_like = tensor_function('zeros_like', fpr=False)
        self._rand = tensor_function('randn', fpr=False)
        self._exp = tensor_function('exp', fpr=False)
        self._dot = tensor_function('dot', fpr=False)
        self._clone = tensor_function('clone', fpr=False)
        self._pad = tensor_function('pad', fpr=False)

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
        b = self._zeros(output_dim[0])
        self.layers.append(Convolution(self._batch_size, self.lr, F, b, F_dim, in_dim, output_dim, padding, stride, calc_partials, self._empty_like, self._empty, self._zeros_like, self._pad, self._dot))

    def copy_conv(self, F, b, F_dim, in_dim, output_dim, padding, stride, calc_partials):
        self.layers.append(Convolution(self._batch_size, self.lr, F, b, F_dim, in_dim, output_dim, padding, stride, calc_partials, self._empty_like, self._empty, self._zeros_like, self._pad, self._dot))

    def add_exact_relu(self):
        """
        Adds an ExactReLU layer to the network.
        """

        self.layers.append(ExactReLU(self._batch_size, self.lr, self._clone))

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
        self.layers.append(FullyConnected(self._batch_size, self.lr, W, b, calc_partials, self._empty_like, self._diag, self._zeros_like))

    def copy_fully_connected(self, W, b, calc_partials):
        """
        Copies a FullyConnected layer to the network.

        Parameters
        ----------
        W:
            The weight matrix of the FullyConnected layer.
        b:
            The bias vector of the FullyConnected layer.
        """

        self.layers.append(FullyConnected(self._batch_size, self.lr, W, b, calc_partials, self._empty_like, self._diag, self._zeros_like))

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

        self.layers.append(AvgPooling(self._batch_size, self.lr, k, in_dim, output_dim, self._empty_like, self._empty))

    def add_max_pooling(self, k, in_dim, output_dim):
        """
        Add a max pooling layer to the network.

        Parameters
        ----------
        k: int
            The size of the pooling window.
        in_dim:
            The input dimension of the layer.
        output_dim:
            The output dimension of the layer.
        """

        self.layers.append(MaxPooling(self._batch_size, self.lr, k, in_dim, output_dim, self._empty_like, self._empty, self._zeros))

    def add_softmax(self):
        """
        Adds a softmax layer to the network.
        """

        self.layers.append(Softmax(self._batch_size, self.lr, self._exp))

    def forward_pass(self, layer_input):
        """
        Performs a forward pass through the network.

        Parameters
        ----------
        layer_input:
            The input data for the network. Its shape must match the input dimension of the first layer.

        Returns
        -------
        layer_output:
            The output of the network after the forward pass.
        """

        for i, layer in enumerate(self.layers):
            layer_input = layer.forward(layer_input)

        return layer_input

    def backward_pass(self, label):
        """
        Perform a backward pass through the network to compute the gradients.

        Parameters
        ----------
        label:
            The label of the input data. Its shape must match the output dimension of the last layer.
        """

        partials = self.layers[-1].backward(label)

        for i, layer in enumerate(self.layers[-2::-1]):
            partials = layer.backward(partials)

    def update_parameters(self):
        """
        Update the parameters of all layers in the network using their accumulated gradients.
        """

        for layer in self.layers:
            layer.update_parameters()

    def save_to_files(self, save_fct, parameter_file_name, layer_types_file_name):
        parameters = {}
        layer_types_f = open(layer_types_file_name, 'w')

        for i, layer in enumerate(self.layers):
            layer_types_f.write(str(type(layer)) + '\n')
            if type(layer) == FullyConnected:
                parameters[i] = (layer.W, layer.b)
            elif type(layer) == Convolution:
                parameters[i] = (layer.F, layer.b, layer.F_dim, layer.in_dim, layer.output_dim, layer.padding, layer.stride)
            elif type(layer) == AvgPooling or type(layer) == MaxPooling:
                parameters[i] = (layer.k, layer.in_dim, layer.output_dim)

        save_fct(parameters, parameter_file_name)

    def load_from_files(self, load_fct, parameter_file_name, layer_types_file_name):
        parameters = load_fct(parameter_file_name)
        layer_types_f = open(layer_types_file_name, 'r')

        for i, line in enumerate(layer_types_f):
            if line.strip() == str(FullyConnected):
                W, b = parameters[i]
                self.copy_fully_connected(W, b, calc_partials=i != 0)
            elif line.strip() == str(Convolution):
                F, b, F_dim, in_dim, output_dim, padding, stride = parameters[i]
                self.copy_conv(F, b, F_dim, in_dim, output_dim, padding, stride, calc_partials=i != 0)
            elif line.strip() == str(ExactReLU):
                self.add_exact_relu()
            elif line.strip() == str(AvgPooling):
                k, in_dim, output_dim = parameters[i]
                self.add_avg_pooling(k, in_dim, output_dim)
            elif line.strip() == str(MaxPooling):
                k, in_dim, output_dim = parameters[i]
                self.add_max_pooling(k, in_dim, output_dim)
            elif line.strip() == str(Softmax):
                self.add_softmax()
