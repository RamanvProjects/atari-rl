from collections import deque
from random import sample

class PolicyNetwork(object):
    """
    Abstract policy network class type
    """
    def __init__(self):
        pass
    
    def partial_fit_step(self):
        pass

    def get_action(self, X):
        pass

    def update_memory(self, s, a, r, gradients=None):
        pass

class Buffer(object):
    def __init__(self, max_size):
        self._buffer = deque()
        self._size = 0
        self._max_size = max_size
    
    @property
    def size(self):
        return self._size

    def add_event(self, s, a, r, t, s_):
        if self._size > self._max_size:
            self._buffer.popleft()
        else:
            self._size += 1
        
        self._buffer.append((s, a, r, t, s_))
    
    def next_batch(self, bsz):
        bsz = bsz if bsz < self.num_examples else self.num_examples
        batch = sample(self._buffer, bsz)
        s_b, a_b, r_b, t_b, s_next_b = zip(*batch)

        return s_b, a_b, r_b, t_b, s_next_b

    def reset_memory(self):
        self.deque.clear()
        self._size = 0