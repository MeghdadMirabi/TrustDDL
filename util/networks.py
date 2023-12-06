import neural_network
import util
from neural_network import CentralizedNeuralNetwork, DistributedNeuralNetwork


def construct_network_centralized_secureml(batch_size, lr):
    nn = CentralizedNeuralNetwork(batch_size=batch_size, lr=lr)
    nn.add_fully_connected(W_dim=[128, 28 * 28 * 1], calc_partials=False)
    nn.add_exact_relu()
    nn.add_fully_connected(W_dim=[128, 128])
    nn.add_exact_relu()
    nn.add_fully_connected(W_dim=[10, 128])
    nn.add_softmax()
    return nn


def construct_network_centralized_chameleon(batch_size, lr):
    nn = CentralizedNeuralNetwork(batch_size=batch_size, lr=lr)
    nn.add_conv(F_dim=[5, 1, 5, 5], in_dim=[1, 28, 28], output_dim=[5, 14, 14], padding=(2, 2), stride=(2, 2), calc_partials=False)
    nn.add_exact_relu()
    nn.add_fully_connected(W_dim=[100, 5*14*14])
    nn.add_exact_relu()
    nn.add_fully_connected(W_dim=[10, 100])
    nn.add_softmax()
    return nn


def construct_network_centralized_sarda(batch_size, lr):
    nn = CentralizedNeuralNetwork(batch_size=batch_size, lr=lr)
    nn.add_conv(F_dim=[5, 1, 2, 2], in_dim=[1, 28, 28], output_dim=[5, 14, 14], padding=(0, 0), stride=(2, 2), calc_partials=False)
    nn.add_exact_relu()
    nn.add_fully_connected(W_dim=[100, 5*14*14])
    nn.add_exact_relu()
    nn.add_fully_connected(W_dim=[10, 100])
    nn.add_softmax()
    return nn


def construct_network_centralized_minionn(batch_size, lr):
    nn = CentralizedNeuralNetwork(batch_size=batch_size, lr=lr)
    nn.add_conv(F_dim=[16, 1, 5, 5], in_dim=[1, 28, 28], output_dim=[16, 24, 24], padding=(0, 0), stride=(1, 1), calc_partials=False)
    nn.add_exact_relu()
    nn.add_max_pooling(k=2, in_dim=[16, 24, 24], output_dim=[16, 12, 12])
    nn.add_conv(F_dim=[16, 16, 5, 5], in_dim=[16, 12, 12], output_dim=[16, 8, 8], padding=(0, 0), stride=(1, 1), calc_partials=True)
    nn.add_exact_relu()
    nn.add_max_pooling(k=2, in_dim=[16, 8, 8], output_dim=[16, 4, 4])
    nn.add_fully_connected(W_dim=[100, 256])
    nn.add_exact_relu()
    nn.add_fully_connected(W_dim=[10, 100])
    nn.add_softmax()
    return nn


def construct_network_centralized_lenet(batch_size, lr):
    nn = CentralizedNeuralNetwork(batch_size=batch_size, lr=lr)
    nn.add_conv(F_dim=[20, 1, 5, 5], in_dim=[1, 28, 28], output_dim=[20, 24, 24], padding=(0, 0), stride=(1, 1), calc_partials=False)
    nn.add_max_pooling(k=2, in_dim=[20, 24, 24], output_dim=[20, 12, 12])
    nn.add_conv(F_dim=[50, 20, 5, 5], in_dim=[20, 12, 12], output_dim=[50, 8, 8], padding=(0, 0), stride=(1, 1), calc_partials=True)
    nn.add_max_pooling(k=2, in_dim=[50, 8, 8], output_dim=[50, 4, 4])
    nn.add_exact_relu()
    nn.add_fully_connected(W_dim=[500, 800])
    nn.add_exact_relu()
    nn.add_fully_connected(W_dim=[10, 500])
    nn.add_softmax()
    return nn


def construct_network_centralized(name, batch_size, lr):
    if name == 'SecureML':
        return construct_network_centralized_secureml(batch_size, lr)
    elif name == 'Chameleon':
        return construct_network_centralized_chameleon(batch_size, lr)
    elif name == 'Sarda':
        return construct_network_centralized_sarda(batch_size, lr)
    elif name == 'MiniONN':
        return construct_network_centralized_minionn(batch_size, lr)
    elif name == 'LeNet':
        return construct_network_centralized_lenet(batch_size, lr)
    else:
        raise ValueError(f'Unknown network \'{name}\'')


def construct_network_distributed_secureml(batch_size, lr, auxiliary_nodes, ran_group):
    nn = DistributedNeuralNetwork(auxiliary_nodes, None, None, ran_group, batch_size=batch_size, lr=lr)
    nn.add_fully_connected(W_dim=[128, 28 * 28 * 1], calc_partials=False)
    nn.add_exact_relu()
    nn.add_fully_connected(W_dim=[128, 128])
    nn.add_exact_relu()
    nn.add_fully_connected(W_dim=[10, 128])
    nn.add_softmax()
    return nn


def construct_network_distributed_chameleon(batch_size, lr, auxiliary_nodes, ran_group):
    nn = DistributedNeuralNetwork(auxiliary_nodes, None, None, ran_group, batch_size=batch_size, lr=lr)
    nn.add_conv(F_dim=[5, 1, 5, 5], in_dim=[1, 28, 28], output_dim=[5, 14, 14], padding=(2, 2), stride=(2, 2), calc_partials=False)
    nn.add_fully_connected(W_dim=[100, 5 * 14 * 14])
    nn.add_exact_relu()
    nn.add_fully_connected(W_dim=[10, 100])
    nn.add_softmax()
    return nn


def construct_network_distributed(name, batch_size, lr, auxiliary_nodes, ran_group):
    if name == 'SecureML':
        return construct_network_distributed_secureml(batch_size, lr, auxiliary_nodes, ran_group)
    elif name == 'Chameleon':
        return construct_network_distributed_chameleon(batch_size, lr, auxiliary_nodes, ran_group)
    else:
        raise ValueError(f'Unknown network \'{name}\'')


def construct_distributed_from_centralized(other_nn, batch_size, lr, auxiliary_nodes, ran_group):
    nn = DistributedNeuralNetwork(auxiliary_nodes, None, None, ran_group, batch_size=batch_size, lr=lr)
    for layer in other_nn.layers:
        from neural_network.centralized import layers
        if type(layer) == neural_network.centralized.layers.ExactReLU:
            nn.add_exact_relu()
        elif type(layer) == neural_network.centralized.layers.Softmax:
            nn.add_softmax()
        elif type(layer) == neural_network.centralized.layers.AvgPooling:
            nn.add_avg_pooling(layer.k, layer.in_dim, layer.output_dim)
        elif type(layer) == neural_network.centralized.layers.FullyConnected:
            nn.copy_fully_connected(layer, util.tensor_function('clone'))
        elif type(layer) == neural_network.centralized.layers.Convolution:
            nn.copy_conv(layer, util.tensor_function('clone'))
    return nn


def construct_network(name, type, batch_size, lr, auxiliary_nodes=None, ran_group=None):
    if type == 'centralized':
        return construct_network_centralized(name, batch_size, lr)
    elif type == 'distributed':
        return construct_network_distributed(name, batch_size, lr, auxiliary_nodes, ran_group)
    else:
        raise ValueError(f'Unknown type \'{type}\'')
