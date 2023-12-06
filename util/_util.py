import torch

import util.constants


tensor_functions = {
    'torch': {
        'rand': torch.rand,
        'argmax': torch.argmax,
        'diag': torch.diag,
        'empty_like': torch.empty_like,
        'empty': torch.empty,
        'zeros': torch.zeros,
        'zeros_like': torch.zeros_like,
        'randn': torch.randn,
        'exp': torch.exp,
        'dot': torch.dot,
        'clone': torch.clone,
        'randint': torch.randint,
        'pad': torch.nn.functional.pad
    }
}

fpr_tensor_functions = {
    'torch': {
        'rand': torch.rand,
        'argmax': torch.argmax,
        'diag': torch.diag,
        'empty_like': torch.empty_like,
        'empty': (lambda x: torch.empty(x, dtype=torch.int64 if util.constants.fixedpoint_representation else torch.float64)),
        'zeros': (lambda x: torch.zeros(x, dtype=torch.int64 if util.constants.fixedpoint_representation else torch.float64)),
        'zeros_like': torch.zeros_like,
        'randn': torch.randn,
        'exp': torch.exp,
        'dot': torch.dot,
        'clone': torch.clone,
        'randint': torch.randint,
        'pad': torch.nn.functional.pad
    }
}


def tensor_function(function_name, fpr=util.constants.fixedpoint_representation):
    if fpr:
        return fpr_tensor_functions[util.constants.tensor_package][function_name]
    else:
        return tensor_functions[util.constants.tensor_package][function_name]
