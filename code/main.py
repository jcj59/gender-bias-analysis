from base_model import BaseModel
from state import State
from probability_functions import base_bias_func, base_fire_func, promotion_probability_func

def main(num_steps=100):
    # Define parameters
    level_populations=[25, 15, 10, 5]
    identities=["M", "F"]
    identity_probabilities=[0.6, 0.4]
    leave_rate=0.5

    # Define the model
    model = BaseModel(
        leave_rate=leave_rate,
        bias_func=base_bias_func,
        fire_func=base_fire_func,
        identities=identities,
        identity_probabilities=identity_probabilities,
        promotion_probability_func=promotion_probability_func,
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
