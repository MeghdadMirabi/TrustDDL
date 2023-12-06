import util.constants


def consensus_confirmation(x: []):
    """
    Consensus confirmation for input x.
    """
    byzantine_values = 0
    value = None
    for val, byzantine in x:
        if byzantine:
            byzantine_values += 1
        else:
            value = val

    if byzantine_values == 0 \
            or (util.constants.optimize_communication and (len(x) - byzantine_values) >= (
            util.constants.max_assumed_byzantine_nodes + 1))\
            or (not util.constants.optimize_communication and byzantine_values < int(len(x) / 2) + 1):
        result = value
    else:
        result = None

    return result
