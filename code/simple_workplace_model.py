from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

# your additional imports here
from dataclasses import field
from typing import Dict, List


@dataclass(frozen=True)
class SimpleWorkplaceState:
    levels: List[str] = field(default_factory=list)
    identities: List[str] = field(default_factory=list)
    state: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    def __post_init__(self):
        n = len(self.levels)
        m = len(self.identities)
                
        # Validate State
        if len(self.state.keys()) != n:
            raise ValueError(f"State does not match levels: {self.state.keys()} (# levels {n})")
        for level in self.levels:
            if len(self.state[level].keys()) != m:
                raise ValueError(f"State does not match identities: ({self.state[level].keys()} {level}) (# identities {m})")
    

@dataclass(frozen=True)
class SimpleWorkplaceParams:
    levels: List[str] = field(default_factory=list) # Levels must be in order of lowest to highest within the list
    identities: List[str] = field(default_factory=list)
    employee_targets: Dict[str, int] = field(default_factory=dict)
    weights: Dict[str, Dict[str, int]] = field(default_factory=dict)
    lambdas: Dict[str, Dict[str, int]] = field(default_factory=dict)
        
    def __post_init__(self):
        n = len(self.levels)
        m = len(self.identities)
        
        # Validate employee targets
        if len(self.employee_targets.keys()) != n:
            raise ValueError(f"Targets do not match levels: {self.employee_targets.keys()} (# levels {n})")
          
        # Validate weights
        if len(self.weights.keys()) != n:
            raise ValueError(f"Weights do not match levels: {self.weights.keys()} (# levels {n})")
        for level in self.levels:
            if len(self.weights[level].keys()) != m:
                raise ValueError(f"Weights do not match identities: ({self.weights[level].keys()} {level}) (# identities {m})")
                
        # Validate lambdas
        if len(self.lambdas.keys()) != n:
            raise ValueError(f"Lambdas do not match levels: {self.lambdas.keys()} (# levels {n})")
        for level in self.levels:
            if len(self.lambdas[level].keys()) != m:
                raise ValueError(f"Lambdas do not match identities: ({self.lambdas[level].keys()} {level}) (# identities {m})")
                
        # Do more checks to ensure that values are all positive
        
        
@dataclass(frozen=True)
class SimpleWorkplaceModel:
    params: SimpleWorkplaceParams

    def transition_rate(self, state: SimpleWorkplaceState):
        rate = 0
        for level in self.params.levels:
            for identity in self.params.identities:
                if state.state[level][identity] > 0:
                    rate += self.params.lambdas[level][identity]
        return rate

    def sample_next(self, state: SimpleWorkplaceState):
        transitions = []
        for level in self.params.levels:
            for identity in self.params.identities:
                if state.state[level][identity] > 0:
                    transitions.append((level, identity, self.params.lambdas[level][identity]))
        rates = np.array([transition[2] for transition in transitions])
        probabilities = rates / rates.sum()
        sampled_index = np.random.choice(len(transitions), p=probabilities)
        sampled_level, sampled_identity, _ = transitions[sampled_index]
        return self.update_state(state, sampled_level, sampled_identity)

    
    def update_state(self, state: SimpleWorkplaceState, departure_level: str, departure_identity: str):
        if departure_level not in self.params.levels:
            raise ValueError(f"Invalid departure level: {departure_level}")
        if departure_identity not in self.params.identities:
            raise ValueError(f"Invalid departure identity: {departure_identity}")
        
        new_state = {level: dict(identities) for level, identities in state.state.items()}
        new_state[departure_level][departure_identity] -= 1
        
        departure_level_index = self.params.levels.index(departure_level)
        if departure_level_index == 0:
            weights = np.array([self.params.weights[departure_level][identity] for identity in self.params.identities])
            probabilities = weights / weights.sum()
            new_identity = np.random.choice(self.params.identities, p=probabilities)
            new_state[departure_level][new_identity] += 1
            return SimpleWorkplaceState(state.levels, state.identities, new_state)
        else:
            prev_departure_level = self.params.levels[departure_level_index - 1]
            weights = np.array([self.params.weights[departure_level][identity] for identity in self.params.identities])
            populations = np.array([state.state[prev_departure_level][identity] for identity in self.params.identities])
            total_weighted_population = weights * populations
            probabilities = total_weighted_population / total_weighted_population.sum()
            new_identity = np.random.choice(self.params.identities, p=probabilities)
            new_state[departure_level][new_identity] += 1
            state = SimpleWorkplaceState(state.levels, state.identities, new_state)
            return self.update_state(state, prev_departure_level, new_identity)
        
def test_update_state(
    params: SimpleWorkplaceParams,
    state: SimpleWorkplaceState,
    departure_level: str,
    departure_identity: str
):
    model = SimpleWorkplaceModel(params)
    return model.update_state(state, departure_level, departure_identity)

def visualize_path_percentage(path, identities, levels):
    """
    Visualizes the state throughout the path as a step graph with filled regions.

    Parameters:
        path (list of tuples): A list where each element is a tuple (time, state),
                               with state being a SimpleWorkplaceState.
        identities (list): List of identities (e.g., ["F", "M"]).
        levels (list): List of levels (e.g., ["0", "1", "2"]).
    """
    times = [entry[0] for entry in path]
    states = [entry[1] for entry in path]  # Each state is a SimpleWorkplaceState
    
    # Prepare data for plotting percentages
    percentages = {identity: [] for identity in identities}
    for state in states:
        total_employees = sum(
            sum(state.state[level][identity] for identity in identities) for level in levels
        )
        for identity in identities:
            total_identity_count = sum(state.state[level][identity] for level in levels)
            percentages[identity].append(total_identity_count / total_employees)
    
    # Prepare cumulative percentages for filling
    cumulative_percentages = np.zeros(len(times))
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    for identity in identities:
        identity_percentages = np.array(percentages[identity])
        plt.step(times, cumulative_percentages + identity_percentages, where="post", label=f"{identity}")
        plt.fill_between(times, cumulative_percentages, cumulative_percentages + identity_percentages, step="post", alpha=0.7)
        cumulative_percentages += identity_percentages
    
    plt.xlim(0, times[-1])
    plt.ylim(0, 1)
    
    # Add labels and legend
    plt.xlabel("Time")
    plt.ylabel("Percentage of Employees")
    plt.title("Overall Employee Identity Distribution Over Time")
    plt.legend(title="Identities", loc="upper right")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    
    plt.show()

def visualize_path_percentage_by_level(path, identities, levels):
    """
    Visualizes the state throughout the path as a step graph with filled regions
    for each level individually.

    Parameters:
        path (list of tuples): A list where each element is a tuple (time, state),
                               with state being a SimpleWorkplaceState.
        identities (list): List of identities (e.g., ["F", "M"]).
        levels (list): List of levels (e.g., ["0", "1", "2"]).
    """
    times = [entry[0] for entry in path]
    states = [entry[1] for entry in path]  # Each state is a SimpleWorkplaceState
    
    # Create a plot for each level
    for level in levels:
        # Prepare data for plotting percentages at the current level
        percentages = {identity: [] for identity in identities}
        for state in states:
            total_employees_at_level = sum(state.state[level][identity] for identity in identities)
            for identity in identities:
                identity_count = state.state[level][identity]
                if total_employees_at_level > 0:
                    percentages[identity].append(identity_count / total_employees_at_level)
                else:
                    percentages[identity].append(0)  # Avoid division by zero
        
        # Prepare cumulative percentages for filling
        cumulative_percentages = np.zeros(len(times))
        
        # Create the plot for this level
        plt.figure(figsize=(12, 8))
        for identity in identities:
            identity_percentages = np.array(percentages[identity])
            plt.step(times, cumulative_percentages + identity_percentages, where="post", label=f"{identity}")
            plt.fill_between(times, cumulative_percentages, cumulative_percentages + identity_percentages, step="post", alpha=0.7)
            cumulative_percentages += identity_percentages
        
        # Customize the plot for this level
        plt.xlim(0, times[-1])
        plt.ylim(0, 1)
        plt.xlabel("Time")
        plt.ylabel("Percentage of Employees")
        plt.title(f"Employee Identity Distribution Over Time at Level {level}")
        plt.legend(title="Identities", loc="upper right")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()

def test_simple_workplace_model(
    params: SimpleWorkplaceParams,
    initial_state: SimpleWorkplaceState,
    n_iters=64
):
    model = SimpleWorkplaceModel(params)
    path = run_ctmc(model, initial_state, n_iters)
    visualize_path_percentage(path, params.identities, params.levels)
    visualize_path_percentage_by_level(path, params.identities, params.levels)
    return path
