import numpy as np

# Random Number Generator
RNG = np.random.default_rng()

# Employee Constants
PERFORMANCE_INCREASE_RATE = 0.01
PERFORMANCE_DECREASE_RATE = 0.01

# Promotion Weights for Combining Factors
PERFORMACE_LEVEL_WEIGHT = 1/3
POSITION_EXPERIENCE_WEIGHT = 1/3
IDENTITY_SIMILARITY_WEIGHT = 1/3

# Bias Constants
MAN_BIAS = 0.5
WOMAN_BIAS = 2
LEVEL_BIAS_COEFFICIENT = 0.25

# Maternity Leave Constants
MATERNITY_LEAVE = 0.5
MATERNITY_RETURN = 0.8
MATERNITY_BIAS = 0.5