import time


class Runtime:

    def __init__(self, runtime_int=0):
        self.runtime_int = runtime_int

    def add(self, other):
        self.runtime_int += other

    def get_runtime(self):
        return self.runtime_int


class RuntimeRecorder:

    def __init__(self, runtime: Runtime, record_runtime=True):
        self.runtime = runtime
        self.record_runtime = record_runtime
        self.start = None

    def __enter__(self):
        if self.record_runtime:
            self.start = time.process_time_ns()

    def __exit__(self, *args):
        if self.record_runtime:
            self.runtime.add(time.process_time_ns() - self.start)
