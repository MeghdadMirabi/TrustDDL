import ray
import torch
from tqdm import tqdm

import configuration as cfg
import util.constants
from actors import AuxiliaryNode, DataOwner, ModelOwner, WorkerNode
from neural_network import CentralizedNeuralNetwork
from util.networks import construct_distributed_from_centralized


def get_accumulated_comm_cost(workers, auxiliaries, model_owner):
    comm_costs = []
    comm_costs.extend(ray.get([w.get_comm_cost.remote() for w in workers]))
    comm_costs.extend(ray.get([a.get_comm_cost.remote() for a in auxiliaries]))
    comm_costs.append(ray.get(model_owner.get_comm_cost.remote()))

    msgs, msg_size = 0, 0
    for _msgs, _msg_size in comm_costs:
        msgs += _msgs
        msg_size += _msg_size

    return msgs, msg_size


def main():
    """ local ray start """
    resources = dict()
    for i in range(util.constants.replication_factor):
        resources[f"party{i}"] = util.constants.num_groups
    resources["Owners"] = 1
    if not ray.is_initialized(): ray.init(log_to_driver=cfg.log_warnings, resources=resources)

    print(f'network={cfg.network}, batch_size={cfg.batch_size}, iterations={cfg.iterations}, train={cfg.train}')

    device = 'cpu'

    random_fct = lambda size: torch.rand(size, dtype=torch.float64, device=device)

    ran_group = 0

    model_owner = ModelOwner.options(resources={f"Owners": 0.5}).remote(util.constants.num_groups, torch.exp, torch.empty_like, torch.abs, random_fct)

    data_owner = DataOwner(util.constants.num_groups, random_fct, torch.zeros)

    auxiliaries = []
    for i in range(util.constants.num_auxiliaries):
        auxiliaries.append(AuxiliaryNode.options(resources={f"party{i}": 0.5}).remote(util.constants.group_sizes, node_id=i))

    centralized_nn = CentralizedNeuralNetwork(batch_size=cfg.batch_size, lr=util.constants.lr)
    centralized_nn.load_from_files(torch.load, f'models/{cfg.network.lower()}/parameters/untrained/parameters.db', f'models/{cfg.network.lower()}/layers.txt')
    distributed_nn = construct_distributed_from_centralized(other_nn=centralized_nn, batch_size=cfg.batch_size, lr=util.constants.lr,
                                                            auxiliary_nodes=auxiliaries,
                                                            ran_group=ran_group)

    workers = []
    workers_dict = {}
    group_nums = []
    models = ray.get(model_owner.create_models.remote(distributed_nn, util.constants.num_groups))
    for i in range(util.constants.num_groups):
        workers_dict[i] = []
        for j in range(util.constants.num_workers):
            worker = WorkerNode.options(resources={f"party{j}": 0.25}).remote(group_num=i, node_id=j,
                                                                              auxiliary_nodes=auxiliaries,
                                                                              model_owner=model_owner,
                                                                              nn=models[j][i], random_fct=random_fct)
            workers.append(worker)
            group_nums.append(i)
            workers_dict[i].append(worker)

    X_shares, Y_shares = init_share_dicts(workers_dict)
    for i in tqdm(range(cfg.iterations * cfg.batch_size)):
        if (i + 1) % cfg.batch_size == 0 and i > 0:
            iterate_jobs = []
            for group_num, group_workers in workers_dict.items():
                for replica, worker in enumerate(group_workers):
                    iterate_jobs.append(worker.iterate.remote(X_shares[group_num][replica], Y_shares[group_num][replica], batch=True, train=cfg.train, return_result=False, return_runtime=True))
            ray.get(iterate_jobs)

            X_shares, Y_shares = init_share_dicts(workers_dict)

        _X_shares, _Y_shares = data_owner.mnist_train_shares(i)
        for group_num, group_workers in workers_dict.items():
            for replica, worker in enumerate(group_workers):
                share_ids = util.constants.share_placement[replica][group_num]
                X_shares[group_num][replica].append([_X_shares[share_id][group_num] for share_id in share_ids])
                Y_shares[group_num][replica].append([_Y_shares[share_id][group_num] for share_id in share_ids])

    iterate_jobs = []
    for group_num, group_workers in workers_dict.items():
        for replica, worker in enumerate(group_workers):
            iterate_jobs.append(worker.iterate.remote(X_shares[group_num][replica], Y_shares[group_num][replica], batch=True, train=cfg.train, return_result=False, return_runtime=True))
    ray.get(iterate_jobs)

    msgs, msg_size = get_accumulated_comm_cost(workers, auxiliaries, model_owner)
    tqdm.write(f'Total # of messages: {msgs}')
    tqdm.write(f'Total communication cost: {msg_size} B = {msg_size / (10**6)} MB = {msg_size / (10**9)} GB')


def init_share_dicts(workers_dict):
    X_shares, Y_shares = {}, {}
    for group_num, group_workers in workers_dict.items():
        X_shares[group_num], Y_shares[group_num] = {}, {}
        for replica, worker in enumerate(group_workers):
            X_shares[group_num][replica], Y_shares[group_num][replica] = [], []
    return X_shares, Y_shares


if __name__ == '__main__':
    main()
