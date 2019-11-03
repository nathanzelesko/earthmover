import sys
import time
import numpy as np
import scipy.sparse

def boolmatrix_any(A):
    assert A.dtype == np.bool

    if scipy.sparse.issparse(A):
        return A.nnz > 0
    else:
        return A.any()


def is_symmetric(A):
    return not boolmatrix_any(A != A.transpose())


class Timer(object):
    def __init__(self, text = None):
        if text != None:
            print ('%s:' % text), 
        sys.stdout.flush()
        self.start_clock = time.clock()
        self.start_time = time.time()
        self.measured_time = None
        self.measured_clock = None

    def stop(self):
        self.measured_time = time.time() - self.start_time
        self.measured_clock = time.clock() - self.start_clock
        self.stop_time = time.time()
        self.stop_clock = time.clock()
        print('Wall time: %.3f seconds.  CPU time: %.3f seconds.' % (self.measured_time, self.measured_clock))

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.stop()

class Profiler(object):
    def __init__(self, text = None):
        if text != None:
            print ('%s:' % text), 
        import cProfile
        self.pr = cProfile.Profile()
        self.pr.enable()
    def stop(self):
        self.pr.disable()
        import pstats
        ps = pstats.Stats(self.pr)
        ps.sort_stats('cumtime').print_stats(50)
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.stop()
