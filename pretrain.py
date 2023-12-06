import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data.preparation.mnist import mnist

torch.set_default_tensor_type(torch.DoubleTensor)

network = 'SecureML'
batch_size = 128
epochs = 15
lr = 1

if network not in ['SecureML', 'Chameleon', 'Sarda']:
    raise ValueError(f'Unsupported network \'{network}\'.')


class SecureML_Model(nn.Module):
    def __init__(self):
        super(SecureML_Model, self).__init__()
        self.layer1 = nn.Linear(784, 128, bias=True)
        self.layer2 = nn.Linear(128, 128, bias=True)
        self.layer3 = nn.Linear(128, 10, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.layer3(self.relu(self.layer2(self.relu(self.layer1(x))))))

    def output(self, x):
        out1 = self.relu(self.layer1(x))
        out2 = self.relu(self.layer2(out1))
        out3 = self.softmax(self.layer3(out2))
        return out1, out2, out3


class Chameleon_Model(nn.Module):
    def __init__(self):
        super(Chameleon_Model, self).__init__()

        self.layer1 = nn.Conv2d(1, 5, kernel_size=5, stride=2, padding=2)
        self.layer2 = nn.Linear(980, 100, bias=True)
        self.layer3 = nn.Linear(100, 10, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)).view(-1, 980)))))

    def output(self, x):
        out1 = self.relu(self.layer1(x)).view(-1, 980)
        out2 = self.relu(self.layer2(out1))
        out3 = self.softmax(self.layer3(out2))
        return out1, out2, out3


class Sarda_Model(nn.Module):
    def __init__(self):
        super(Sarda_Model, self).__init__()

        self.layer1 = nn.Conv2d(1, 5, kernel_size=2, stride=2, padding=0)
        self.layer2 = nn.Linear(980, 100, bias=True)
        self.layer3 = nn.Linear(100, 10, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)).view(-1, 980)))))

    def output(self, x):
        out1 = self.relu(self.layer1(x)).view(-1, 980)
        out2 = self.relu(self.layer2(out1))
        out3 = self.softmax(self.layer3(out2))
        return out1, out2, out3


if network == 'SecureML':
    model = SecureML_Model()
elif network == 'Chameleon':
    model = Chameleon_Model()
elif network == 'Sarda':
    model = Sarda_Model()

optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
lrs = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)


def get_acc(model, loader):
    correct = 0
    total = 0
    for img, label in loader:
        if network == 'SecureML':
            correct += torch.sum(torch.argmax(model(img.view(-1, 784)), -1) == label).item()
        elif network == 'Chameleon' or network == 'Sarda':
            correct += torch.sum(torch.argmax(model(img), -1) == label).item()
        total += len(img)
    return 100 * correct / total


loaders = mnist.prepare_loaders(batch_size=batch_size)

bar1 = tqdm(total=len(loaders['train']), position=0, leave=False)
for e in range(epochs):
    tqdm.write(f"lr {optimizer.param_groups[0]['lr']}")
    for img, label in loaders['train']:
        out = None
        if network == 'SecureML':
            out = model(img.view(-1, 784))
        elif network == 'Chameleon' or network == 'Sarda':
            out = model(img)
        optimizer.zero_grad()
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        bar1.update(1)
    lrs.step()
    bar1.reset()
    tqdm.write(f"Epoch {e}, test accuracy {get_acc(model, loaders['test'])}")


if not os.path.exists(f'models/{network.lower()}/parameters/trained/bs-{batch_size}-e-{epochs}/'):
    os.makedirs(f'models/{network.lower()}/parameters/trained/bs-{batch_size}-e-{epochs}/')

_params = list(model.parameters())
_params = [p.data for p in _params]
parameters = {}

# format for fully connected layers: weights, biases, <not first layer?>
# format for convolutional layers: weights, biases, filter dim, input dimension, output dimension, padding, stride, <not first layer?>

if network == 'SecureML':
    parameters[0] = (_params[0], _params[1])  # fully connected layer 1
    # ReLU
    parameters[2] = (_params[2], _params[3])  # fully connected layer 2
    # ReLU
    parameters[4] = (_params[4], _params[5])  # fully connected layer 3
    # Softmax

elif network == 'Chameleon':
    parameters[0] = (_params[0], _params[1], [5, 1, 5, 5], [1, 28, 28], [5, 14, 14], (2, 2), (2, 2))  # convolutional layer
    # ReLU
    parameters[2] = (_params[2], _params[3])  # fully connected layer 1
    # ReLU
    parameters[4] = (_params[4], _params[5])  # fully connected layer 2
    # Softmax

elif network == 'Sarda':
    parameters[0] = (_params[0], _params[1], [5, 1, 2, 2], [1, 28, 28], [5, 14, 14], (0, 0), (2, 2))  # convolutional layer
    # ReLU
    parameters[2] = (_params[2], _params[3])  # fully connected layer 1
    # ReLU
    parameters[4] = (_params[4], _params[5])  # fully connected layer 2
    # Softmax

torch.save(parameters, f'models/{network.lower()}/parameters/trained/bs-{batch_size}-e-{epochs}/parameters.db')
