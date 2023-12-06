from asyncio import Queue, Lock
from hashlib import sha256

import util
import util.constants


class Collector:
    """
    Abstract class to collect secret shares from groups of nodes.
    """

    def __init__(self, group_sizes, comm_evaluator) -> None:
        """
        Parameters
        ----------
        group_sizes:
            A list of group sizes if ``collection_mode=='mult'`` or the size of the single group to collect from if
            ``collection_mode=='single'``.
        """

        self.collector_comm_evaluator = comm_evaluator

        self.sec_result = Queue(maxsize=1)
        self.byzantine_exception_queue = Queue(maxsize=1)
        self.sec_shares = {}
        self.commits = {}
        self.flags = []
        self.flags_prev = []
        self.group_sizes = group_sizes
        self.collection_iteration = 0
        self.new_collection_ids = None
        self.iteration_locks = {}
        self.iteration_locks_access = Lock()
        self.results_delivered = 0
        self.shares_collected = 0
        self.commits_collected = 0

    async def reset_or_wait(self, iteration):
        if iteration > self.collection_iteration:
            async with self.iteration_locks_access:
                if iteration not in self.iteration_locks:
                    self.iteration_locks[iteration] = Lock()
                    await self.iteration_locks[iteration].acquire()
            await self.iteration_locks[iteration].acquire()
            self.iteration_locks[iteration].release()

    async def await_result(self):
        # wait until result queue is not empty (-> method can be called in parallel!)
        result = await self.sec_result.get()
        # re-queue result
        await self.sec_result.put(result)
        self.results_delivered += 1
        return result

    async def optional_reset(self, commit_support):
        results_to_deliver = sum(self.group_sizes)
        if self.results_delivered == results_to_deliver:
            await self._reset_helper_vars(commit_support)

    async def _reset_helper_vars(self, commit_support):
        await self.sec_result.get()
        self.shares_collected = 0
        self.results_delivered = 0
        self.collection_iteration += 1
        if not commit_support:
            self.sec_shares = {}
            self.commits = {}
            self.commits_collected = 0
            self.flags_prev = self.flags
            self.flags = []

    async def save_commits(self, commit_value, group_num, node_id):
        """
        Share format (dimensions: outer -> inner):
        - batched:
        """

        if group_num not in self.commits.keys():
            self.commits[group_num] = {}

        self.commits[group_num][node_id] = commit_value

        self.shares_collected += 1
        self.commits_collected += 1

    async def save_shares(self, shares, group_num, node_id, share_ids, batch, commit):
        """
        Share format (dimensions: outer -> inner):
        - batched:
        """

        if group_num not in self.sec_shares.keys():
            self.sec_shares[group_num] = {}

        batch_size = len(shares) if batch else None

        if commit:
            commit_recalc = sha256(str(shares).encode('utf-8')).hexdigest()
            flag = commit_recalc == self.commits[group_num][node_id]
        else:
            self.flags = self.flags_prev

        for share_idx, share_id in enumerate(share_ids):
            if share_id not in self.sec_shares[group_num]:
                self.sec_shares[group_num][share_id] = []

            if batch:
                shares_per_id = []
                for batch_idx in range(batch_size):
                    share = shares[batch_idx][share_idx]
                    shares_per_id.append(share)
            else:
                shares_per_id = shares[share_idx]

            if share_idx == 0:
                self.sec_shares[group_num][share_id].insert(0, shares_per_id)
                if commit and not flag:
                    self.flags.append((share_id, 0))
            else:
                self.sec_shares[group_num][share_id].append(shares_per_id)
                if commit and not flag:
                    self.flags.append((share_id, 1))

        self.shares_collected += 1

    async def check_ready(self):
        """
        Check if shares of every worker in every group have been received.
        """
        return self.shares_collected == util.constants.replication_factor * util.constants.num_groups

    def generic_double_reconstruct(self):
        shares_1, shares_2 = self._organize_reconstruction_sets()

        e, e_hat = [], []
        f, f_hat = [], []
        for share_id in range(util.constants.replication_factor):
            e.append(sum([shares_1[share_id][group_num][0] for group_num in range(util.constants.num_groups)]))
            e_hat.append(sum([shares_2[share_id][group_num][0] for group_num in range(util.constants.num_groups)]))
            f.append(sum([shares_1[share_id][group_num][1] for group_num in range(util.constants.num_groups)]))
            f_hat.append(sum([shares_2[share_id][group_num][1] for group_num in range(util.constants.num_groups)]))

        return self._accept_double_non_byzantine_result(e, f, e_hat, f_hat, batch=False)

    def generic_double_reconstruct_batch(self):
        shares_1, shares_2 = self._organize_reconstruction_sets()

        e, e_hat = [], []
        f, f_hat = [], []
        batch_size = len(shares_1[0][0])
        for share_id in range(util.constants.replication_factor):
            e.append([sum([shares_1[share_id][group_num][i][0] for group_num in range(util.constants.num_groups)]) for i in range(batch_size)])
            e_hat.append([sum([shares_2[share_id][group_num][i][0] for group_num in range(util.constants.num_groups)]) for i in range(batch_size)])
            f.append([sum([shares_1[share_id][group_num][i][1] for group_num in range(util.constants.num_groups)]) for i in range(batch_size)])
            f_hat.append([sum([shares_2[share_id][group_num][i][1] for group_num in range(util.constants.num_groups)]) for i in range(batch_size)])

        return self._accept_double_non_byzantine_result(e, f, e_hat, f_hat, batch=True)

    def _organize_reconstruction_sets(self):
        shares_1, shares_2 = {}, {}
        for group_num in range(util.constants.num_groups):
            for share_id in range(util.constants.replication_factor):
                if share_id not in shares_1.keys():
                    shares_1[share_id] = []
                    shares_2[share_id] = []
                share_1 = self.sec_shares[group_num][share_id][0]
                if group_num == util.constants.replicate_shares_on_worker_id:
                    share_2 = self.sec_shares[group_num][share_id][1]
                    shares_1[share_id].append(share_1)
                    shares_2[share_id].append(share_2)
                else:
                    shares_1[share_id].append(share_1)
                    shares_2[share_id].append(share_1)
        return shares_1, shares_2

    def _accept_double_non_byzantine_result(self, e, f, e_hat, f_hat, batch):
        min_dist = None
        accepted = None
        for i in range(util.constants.replication_factor):
            for j in range(util.constants.replication_factor):
                if i != j and ((i, 0) not in self.flags) and ((j, 1) not in self.flags):
                    if batch:
                        dist = sum([(_e - _e_hat).abs().sum() + (_f - _f_hat).abs().sum() for _e, _e_hat, _f, _f_hat in zip(e[i], e_hat[j], f[i], f_hat[j])])
                    else:
                        dist = (e[i] - e_hat[j]).abs().sum() + (f[i] - f_hat[j]).abs().sum()
                    if min_dist is None or dist < min_dist:
                        accepted = i
                        min_dist = dist
        return e[accepted], f[accepted]

    def generic_single_reconstruct(self):
        rs_1_shares, rs_2_shares = self._organize_reconstruction_sets()

        x = [sum(rs_1_shares[share_id]) for share_id in range(util.constants.replication_factor)]
        x_hat = [sum(rs_2_shares[share_id]) for share_id in range(util.constants.replication_factor)]

        return self._accept_single_non_byzantine_result(x, x_hat, batch=False)

    def generic_single_reconstruct_batch(self):
        x_shares_1, x_shares_2 = self._organize_reconstruction_sets()

        x = []
        x_hat = []
        batch_size = len(x_shares_1[0][0])
        for share_id in range(util.constants.replication_factor):
            x.append([sum([x_shares_1[share_id][group_num][i] for group_num in range(util.constants.num_groups)]) for i in range(batch_size)])
            x_hat.append([sum([x_shares_2[share_id][group_num][i] for group_num in range(util.constants.num_groups)]) for i in range(batch_size)])

        return self._accept_single_non_byzantine_result(x, x_hat, batch=True)

    def _accept_single_non_byzantine_result(self, x, x_hat, batch):
        min_dist = None
        accepted = None
        for i in range(util.constants.replication_factor):
            for j in range(util.constants.replication_factor):
                if i != j and ((i, 0) not in self.flags) and ((j, 1) not in self.flags):
                    if batch:
                        dist = sum([(_x - _x_hat).abs().sum() for _x, _x_hat in zip(x[i], x_hat[j])])
                    else:
                        dist = (x[i] - x_hat[j]).abs().sum()
                    if min_dist is None or dist < min_dist:
                        accepted = i
                        min_dist = dist
        return x[accepted]
