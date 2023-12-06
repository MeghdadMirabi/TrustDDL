import util.constants
from util import fixedpoint_representation, fp_precision


def to_config_representation(value, scalar=False, value_type='float'):
    return to_fixedpoint(value, scalar, value_type) if fixedpoint_representation else value


def to_fixedpoint(value, scalar=False, value_type='float'):
    fp_value = value * (2**fp_precision)
    if value_type == 'float':
        return int(fp_value) if scalar else fp_value.long()
    elif value_type == 'int':
        return fp_value
    else:
        raise ValueError("value_type must be one of {'float', 'int'}.")


def to_float(fp_tensor, fpr=util.constants.fixedpoint_representation):
    return fp_tensor.double() / (2**fp_precision) if fpr else fp_tensor


def mul(x, y, fpr=util.constants.fixedpoint_representation):
    # assume an equal scaling factor of 'fp_precision' for x and y if in fixedpoint representation
    return (x * y) >> fp_precision if fpr else x * y


def div(x, y):
    # assume an equal scaling factor of 'fp_precision' for x and y if in fixedpoint representation
    return to_config_representation(x / y)


def matmul(X, Y, fpr=util.constants.fixedpoint_representation):
    # assume an equal scaling factor of 'fp_precision' for x and y if in fixedpoint representation
    return (X @ Y) >> fp_precision if fpr else X @ Y
