from base_model import BaseModel
from state import State
from base_functions import *
from interventions import *
from constants import *

def main(num_steps=100):
    # Define parameters
    level_populations=[25, 15, 10, 5]
    identities=["M", "F"]
    identity_probabilities=[0.6, 0.4]
    leave_rate=0.5

    # Define the model
    model = BaseModel(
        leave_rate=LEAVE_RATE,
        maternity_leave_rate=MATERNITY_LEAVE,
        identities=IDENTITIES,
        quit_func=base_quit_func,
        fire_func=base_fire_func,
        bias_func=base_bias_func,
        identity_probabilities_func=base_hire_func, 
        promotion_probability_func=base_promotion_func, 
        num_levels=NUM_LEVELS, 
        level_populations=LEVEL_POPULATIONS,
        population_percentages=IDENTITY_POPULATION_PERCENTAGES,
        quotas=None
    )

    # Define the initial state
    initial_state = State.generate_initial_state(
        level_populations=level_populations,
        identities=identities,
        identity_probabilities=identity_probabilities,
    )

    # Run the simulation
    path = model.run(initial_state, num_steps)
    return path, model

if __name__ == "__main__":
    path = main()
    print("Simulation complete.")
    print(len(path))
