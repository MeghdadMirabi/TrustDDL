import torch

import configuration as cfg

tensor_package = 'torch'
torch.set_default_tensor_type(torch.DoubleTensor)

fixedpoint_representation = True
fp_precision = 20

replication_factor = 3
replicate_shares_on_worker_id = 0
num_groups = 2
num_workers = replication_factor
num_auxiliaries = replication_factor
num_mediators = replication_factor
group_sizes = [num_workers for _ in range(num_groups)]
total_nodes = num_groups * num_workers + num_auxiliaries + num_mediators + 1

share_placement = [[] for i in range(replication_factor)]
# map node_ids to the (list of) replica-ids maintained on each worker
for replica in range(replication_factor):
    for w_id in range(num_groups):
        share_placement[replica].append([])
        if w_id == replicate_shares_on_worker_id:
            for i in range(max(1, replication_factor - 1)):
                share_placement[replica][w_id].append((replica + i) % replication_factor)
        else:
            share_placement[replica][w_id].append((replication_factor - 1 + replica) % replication_factor)

optimize_communication = False
max_assumed_byzantine_nodes = 1

if cfg.model == 'malicious':
    commit_default = True
    byzantine_setup = {
        'auxiliary': {i: i < max_assumed_byzantine_nodes if num_auxiliaries >= 2*max_assumed_byzantine_nodes+1 else False for i in range(num_auxiliaries)},
        'worker': {i: {j: j < max_assumed_byzantine_nodes if group_sizes[i] >= 2*max_assumed_byzantine_nodes+1 else False for j in range(group_sizes[i])} for i in range(num_groups)}
    }
elif cfg.model == 'semi-honest':
    commit_default = False
    byzantine_setup = {
        'auxiliary': {i: False for i in range(num_auxiliaries)},
        'worker': {i: {j: False for j in range(group_sizes[i])} for i in range(num_groups)}
    }

lr = None
if cfg.network == 'SecureML':
    if cfg.batch_size <= 1:
        lr = 2 ** (-7)
    elif cfg.batch_size <= 10:
        lr = (2 ** (-7)) * cfg.batch_size
    else:
        lr = (2 ** (-7)) * (cfg.batch_size/2)
elif cfg.network == 'Chameleon' or cfg.network == 'Sarda':
    if cfg.batch_size <= 1:
        lr = 0.01
    elif cfg.batch_size <= 10:
        lr = 0.05
    else:
        lr = 0.1
else:
    raise Exception(f'Unsupported network \'{cfg.network}\'.')
