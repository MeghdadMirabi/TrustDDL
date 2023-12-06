from fixedpoint.fp_util import to_fixedpoint

share_min = - 2 ** 2
share_max = 2 ** 2

share_min_fp = to_fixedpoint(share_min, scalar=True, value_type='int')
share_max_fp = to_fixedpoint(share_max, scalar=True, value_type='int')
