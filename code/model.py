from abc import ABC, abstractmethod
from constants import RNG
from copy import deepcopy
import matplotlib.pyplot as plt

class Model(ABC):
    @abstractmethod
    def transition_rate(self, state):
        raise NotImplementedError

    @abstractmethod
    def sample_next(self, state):
        raise NotImplementedError
    
    @abstractmethod
    def hire(self, state):
        raise NotImplementedError
    
    @abstractmethod
    def promote(self, state, level):
        raise NotImplementedError
    
    def run(self, state_init, n_steps=256, log_interval=10):
        self.time = 0.0
        path = [(self.time, deepcopy(state_init))]
        last_logged_time = 0  # Tracks the last logged time for intervals
        
        while self.time <= n_steps:
            current_state = path[-1][1]
            rate = self.transition_rate(current_state)
            time_delta = RNG.exponential(1 / rate)
            self.time += time_delta

            # Log the current time at regular intervals
            if self.time >= last_logged_time + log_interval:
                print(f"Simulation time: {self.time:.2f}")
                last_logged_time = self.time

            next_state = self.sample_next(current_state, time_delta)
            path.append((self.time, deepcopy(next_state)))

        # Return a list of (time, state) pairs
        return path

    

