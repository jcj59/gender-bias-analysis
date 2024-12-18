from model import Model
import numpy as np

RNG = np.random.default_rng()


def run_dtmc(model: Model, state_init, n_steps=256):
    path = [state_init]
    for i in range(n_steps):
        path.append(model.sample_next(path[-1]))
    # Return a list of (time, state) pairs
    # This simulation is discrete-time, so time is an integer
    return list(enumerate(path))

def run_ctmc(model : Model, state_init, n_steps=256):
    "Continuous-time version of `run_dtmc`"""
    current_time = 0.0
    path = [(current_time, state_init)]
    
    while current_time <= n_steps:
        current_state = path[-1][1]
        rate = model.transition_rate(current_state) 
        current_time += RNG.exponential(1/rate)
        path.append((current_time, model.sample_next(current_state)))
    # Return a list of (time, state) pairs
    return path