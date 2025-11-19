import time

class Profiler:

    def __init__(self):
        self._start_time = None
        self._start_cpu_time = None

    def start(self):
        self._start_time = time.time()
        self._start_cpu_time = time.process_time()

    def step(self, label=None):

        if self._start_time is None:
            raise RuntimeError("[PROFILE] Error: profiler has not been started")

        msg = "[PROFILE] Step"
        if label is not None:
            msg += "(%s)" % label
        msg += ", time: %.3f s" % (time.time() - self._start_time)
        msg += ", cpu-time: %.3f s" % (time.process_time() - self._start_cpu_time)
        print(msg)

        self._start_time = time.time()
        self._start_cpu_time = time.process_time()

profiler = Profiler()