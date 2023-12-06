class AbstractNode:

    def __init__(self, node_id, byzantine) -> None:
        self.node_id = node_id
        self.byzantine = byzantine
