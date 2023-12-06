import torch

import secret_sharing.constants
import util.constants
from fixedpoint.fp_util import to_fixedpoint
from util import tensor_function


def create_shares(secret, num_shares: int, random_float=tensor_function('rand'), random_int=tensor_function('randint'),
                  fpr=util.constants.fixedpoint_representation, num_replicas=util.constants.replication_factor, secret_in_fpr=False):
    """
    Creates additive secret shares for an input ``secret``.

    Parameters
    ----------
    secret:
        Floating point tensor to create additive secret shares for.
    num_shares: int
        Number of secret shares to create.
    random_float:
        Function to create random values or tensors in the representation of the input.
    fpr: bool
        Toggle `floating point` representation (False) or `fixed point` representation (True).

    Returns
    -------
    list
        A list of additive secret shares.
    """

    if fpr and not secret_in_fpr:
        secret = to_fixedpoint(secret)

    shares = []
    for _ in range(num_replicas):
        shares_replica = []
        for _ in range(num_shares-1):
            shares_replica.append(random_share(secret.shape, random_float, random_int, fpr))
        final_share = secret - sum(shares_replica)
        shares_replica.append(final_share)
        shares.append(shares_replica)

    return shares


def random_share(secret_shape, random_float=tensor_function('rand'), random_int=tensor_function('randint'), fpr=util.constants.fixedpoint_representation):
    if fpr:
        return random_int(low=secret_sharing.constants.share_min_fp, high=secret_sharing.constants.share_max_fp, size=secret_shape)
    else:
        return (secret_sharing.constants.share_min - secret_sharing.constants.share_max) * random_float(secret_shape) + secret_sharing.constants.share_max


def approx_equals(x, y, epsilon=1e-8):
    return torch.max(torch.abs((x-y))) < epsilon


def approx_equals_batch(x, y, epsilon=1e-8):
    for _x, _y in zip(x, y):
        if torch.max(torch.abs((_x-_y))) >= epsilon:
            return False
    return True
