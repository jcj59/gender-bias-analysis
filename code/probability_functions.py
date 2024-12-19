from utils import probabilities_from_weights
from constants import PERFORMACE_LEVEL_WEIGHT, POSITION_EXPERIENCE_WEIGHT, IDENTITY_SIMILARITY_WEIGHT, MAN_BIAS, WOMAN_BIAS, LEVEL_BIAS_COEFFICIENT

def base_fire_func(state):
    """
    Provides probabilities of employees being fired based on their performance levels.
    """
    employees = state.employees
    num_employees = len(employees)
    fire_weights = [0] * num_employees

    for i, employee in enumerate(employees):
        fire_weights[i] = employee.performance_level
    fire_rate = sum(fire_weights)
    return fire_rate, probabilities_from_weights(fire_weights)

def base_quit_func(state):
    """
    Provides probabilities of employees quitting based on their bias scores.
    """
    employees = state.employees
    num_employees = len(employees)
    bias_weights = [0] * num_employees

    for i, employee in enumerate(employees):
        bias_weights[i] = employee.bias_score
    quit_rate = sum(bias_weights)
    return quit_rate, probabilities_from_weights(bias_weights)

def promotion_probability_func(state, employees, level, identities):
    """
    Provides probabilities of employees being promoted based on their performance levels, seniority, and identity similarity.
    """
    num_employees = len(employees)
    promotion_weights = [0] * num_employees

    # Calculate identity weights
    identity_weights = {identity: 0 for identity in identities}
    for employee in state.employees:
        if employee.position_level == level:
            identity_weights[employee.identity] += 1
    
    max_identity_weight = max(identity_weights.values()) if identity_weights else 1
    max_experience = max(e.position_experience for e in employees) if employees else 1

    for i, employee in enumerate(employees):
        normalized_experience = employee.position_experience / max_experience
        normalized_identity = identity_weights[employee.identity] / max_identity_weight
        promotion_weights[i] = (
            PERFORMACE_LEVEL_WEIGHT * employee.performance_level +
            POSITION_EXPERIENCE_WEIGHT * normalized_experience +
            IDENTITY_SIMILARITY_WEIGHT * normalized_identity
        )

    return probabilities_from_weights(promotion_weights)

def base_bias_func(employee):
    identity_bias = {
        "M": MAN_BIAS,  
        "F": WOMAN_BIAS,  
    }
    identity_bias_score = identity_bias.get(employee.identity, 0)
    
    # Add bias based on position level (e.g., more bias at higher levels)
    level_bias = LEVEL_BIAS_COEFFICIENT * (employee.position_level + 1)
    total_bias = identity_bias_score * level_bias 
    return total_bias
