import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def probabilities_from_weights(weights):
    weights = np.array(weights, dtype=np.float64)
    total_weight = np.sum(weights)
    
    if total_weight == 0:
        # If all weights are zero, return uniform probabilities
        return [float(x) for x in np.ones_like(weights) / len(weights)]
    
    probabilities = weights / total_weight
    # Ensure sum is exactly 1 by correcting the final element
    probabilities[-1] += 1.0 - np.sum(probabilities)
    
    return probabilities


