import numpy as np

# Random Number Generator
RNG = np.random.default_rng()

# Employee Constants
PERFORMANCE_INCREASE_RATE = 0.01
PERFORMANCE_DECREASE_RATE = 0.05

# Promotion Weights for Combining Factors
PERFORMACE_LEVEL_WEIGHT = 1/3
POSITION_EXPERIENCE_WEIGHT = 1/3
IDENTITY_SIMILARITY_WEIGHT = 1/3