import ray
import torch
from tqdm import tqdm

import configuration as cfg
import util
import util.constants
from actors import AuxiliaryNode, DataOwner, ModelOwner, WorkerNode
from data.preparation.mnist import mnist
from neural_network import CentralizedNeuralNetwork
from secret_sharing.additive_secret_sharing import approx_equals
from util.networks import construct_distributed_from_centralized


def main():
    """ local ray start """
    resources = dict()
    for i in range(util.constants.replication_factor):
        resources[f"party{i}"] = util.constants.num_groups
    resources["Owners"] = 1
    if not ray.is_initialized(): ray.init(log_to_driver=cfg.log_warnings, resources=resources)

    print(f'network={cfg.network}, batch_size={cfg.batch_size}, epochs={15}')

    device = 'cpu'

    random_fct = lambda size: torch.rand(size, dtype=torch.float64, device=device)

    ran_group = 0

    model_owner = ModelOwner.options(resources={f"Owners": 0.5}).remote(util.constants.num_groups, torch.exp, torch.empty_like, torch.abs, random_fct)

    data_owner = DataOwner(util.constants.num_groups, random_fct, torch.zeros)
    mnist.prepare_loaders(batch_size=cfg.batch_size)

    auxiliaries = []
    for i in range(util.constants.num_auxiliaries):
        auxiliaries.append(AuxiliaryNode.options(resources={f"party{i}": 0.5}).remote(util.constants.group_sizes, node_id=i))

    centralized_nn = CentralizedNeuralNetwork(batch_size=cfg.batch_size, lr=util.constants.lr)
    centralized_nn.load_from_files(torch.load, f'models/{cfg.network.lower()}/parameters/trained/bs-{cfg.batch_size}-e-15/parameters.db', f'models/{cfg.network.lower()}/layers.txt')
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

    test_size = data_owner.get_mnist_test_size()
    bar1 = tqdm(total=test_size, desc='Testing', position=0, leave=False)

    bar1.reset(test_size)
    bar1.set_description('Centralized Tests')
    accuracy_centralized = run_tests_centralized(centralized_nn, data_owner, test_size=test_size, tqdm_bar=bar1)
    bar1.reset(test_size)
    bar1.set_description('TrustDDL Tests')
    accuracy_distributed = run_tests_distributed(data_owner, workers_dict, test_size=test_size, tqdm_bar=bar1)
    bar1.reset(test_size)
    tqdm.write(f'Accuracy (Centralized, TrustDDL): {accuracy_centralized * 100} {accuracy_distributed * 100}')


def run_tests_distributed(data_owner, workers_dict, test_size, tqdm_bar):
    correct = 0
    for i in range(test_size):
        X_shares, label = data_owner.mnist_test_shares(i)

        inference_jobs = []
        for group_num, group_workers in workers_dict.items():
            for replica, worker in enumerate(group_workers):
                share_ids = util.constants.share_placement[replica][group_num]
                _X_shares = [X_shares[share_id][group_num] for share_id in share_ids]
                inference_jobs.append(worker.inference.remote(_X_shares, batch=False))

        results = ray.get(inference_jobs)

        replica_results = reconstruct_replicas(0, results, workers_dict)

        non_byzantine_result = replica_results[0]
        replica_byzantine = len(replica_results) != 1
        for i in range(util.constants.replication_factor - 1):
            if approx_equals(replica_results[i], replica_results[i + 1]):
                replica_byzantine = False
                non_byzantine_result = replica_results[i]
                break

        if replica_byzantine:
            old_results = replica_results
            replica_results = reconstruct_replicas(1, results, workers_dict)

            for j in range(util.constants.replication_factor):
                for k in range(util.constants.replication_factor):
                    if j != k and approx_equals(replica_results[j], old_results[k]):
                        replica_byzantine = False
                        non_byzantine_result = replica_results[j]
                        break
                if not replica_byzantine:
                    break

        prediction = torch.argmax(non_byzantine_result)
        if prediction == label:
            correct += 1

        tqdm_bar.update(1)

    return correct / test_size


def reconstruct_replicas(i, results, workers_dict):
    result_dict = {}
    for replica in range(util.constants.replication_factor):
        result_dict[replica] = []
    for group_num, group_workers in workers_dict.items():
        for replica, worker in enumerate(group_workers):
            _res = results[group_num * util.constants.replication_factor + replica]
            share_ids = util.constants.share_placement[replica][group_num]
            if group_num == util.constants.replicate_shares_on_worker_id:
                result_dict[share_ids[i]].append(_res[i])
            else:
                result_dict[share_ids[0]].append(_res[0])
    replica_results = []
    for replica in result_dict.keys():
        replica_results.append(sum(result_dict[replica]))
    return replica_results


def run_tests_centralized(nn, data_owner, test_size, tqdm_bar):
    correct = 0
    for i in range(test_size):
        image, label = data_owner.mnist_test_data(i)
        output = nn.forward_pass(image)
        result = torch.argmax(output)

        if result == label:
            correct = correct + 1

        tqdm_bar.update(1)

    return correct / test_size


if __name__ == '__main__':
    main()
