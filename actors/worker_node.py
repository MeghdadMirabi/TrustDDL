import time

import ray
from ray.util.queue import Queue

import configuration as cfg
import util
import util.constants
from actors.abstract import AbstractNode
from evaluation import CommunicationEvaluator
from neural_network import DistributedNeuralNetwork
from secret_sharing.secure_computations import SecComp

cpu_res = cfg.threads_available / util.constants.total_nodes


@ray.remote(num_cpus=(cpu_res if cpu_res < 1 else int(cpu_res)))
class WorkerNode(AbstractNode):

    nn: DistributedNeuralNetwork

    _auxiliary_nodes: []
    _sec_comp: SecComp
    _group_num: int

    def __init__(self, group_num: int, node_id: int, auxiliary_nodes: [], model_owner, nn: DistributedNeuralNetwork, random_fct) -> None:
        self.comm_evaluator = CommunicationEvaluator()

        AbstractNode.__init__(self, node_id, util.constants.byzantine_setup['worker'][group_num][node_id])

        self.iteration_auxiliaries = 0
        self.iteration_mo = 0

        self._auxiliary_nodes = auxiliary_nodes
        self._sec_comp = SecComp(auxiliary_nodes, model_owner, group_num, node_id, self, random_fct)
        self._group_num = group_num

        if nn is not None:
            self.nn = nn
            self.nn.sec_comp = self._sec_comp
            for layer in self.nn.layers:
                layer.sec_comp = self._sec_comp

        self.replicas_to_maintain = util.constants.share_placement[self.node_id][self._group_num]
        self.model_owner_receive_queue = Queue(maxsize=1)

    def update_model(self, nn: DistributedNeuralNetwork):
        self.nn = nn

    async def iterate(self, X_i, y_i, batch=True, train=True, return_result=True, return_runtime=False):
        start = time.time_ns()

        shares_sent = len(X_i[0]) if batch else len(X_i)

        if shares_sent != len(self.replicas_to_maintain):
            raise ValueError(f'The number of shares sent to Worker(node_id={self.node_id}, group_num={self._group_num}) '
                             f'does not match the number of shares to maintain.')

        output = await self.nn.forward_pass(X_i, batch)

        if train:
            await self.nn.backward_pass(y_i, batch)

        end = time.time_ns()

        if return_result:
            if return_runtime:
                return output, start, end
            else:
                return output
        elif return_runtime:
            return start, end
        else:
            return None

    async def inference(self, X_i, batch):
        shares_sent = len(X_i[0]) if batch else len(X_i)

        if shares_sent != len(self.replicas_to_maintain):
            raise ValueError(
                f'The number of shares sent to Worker(node_id={self.node_id}, group_num={self._group_num}) '
                f'does not match the number of shares to maintain.')

        return await self.nn.forward_pass(X_i, batch)

    def get_comm_cost(self):
        return self._sec_comp.online_comm_evaluator.msgs, self._sec_comp.online_comm_evaluator.msg_size

    def increase_auxiliary_iteration(self):
        self.iteration_auxiliaries += 1

    def increase_mo_iteration(self):
        self.iteration_mo += 1
