"""
Sources of bias in the model:
1. Population skewed towards men (Can never fix)
2. Homophily in hiring and promotion (Can fix)
3. Maternity leave (Can reduce)
4. Bias (including level bias) this plays a role in performance and promotion (Can reduce)

We want the following interventions:

General Interventions:
1. Quotas: Enforce a minimum number of employees from each identity group at each level (Model input)
2. Anti Bias Training: Introduce decay to bias scores over time
3. Maternity leave policy changes: Introduce decay in bias scores during maternity leave (Not yet implemented)

Promotion Specific Interventions:
1. Completely random promotion (hire a person completely at random)
2. Performance-based promotion (hire the person with the highest performance level)
3. Seniority-based promotion (hire the most senior person)

Hire Specific Interventions:
1. Completely random hiring (hire a person completely at random)
2. Population-based hiring (hire people based on the population percentages)
"""
from constants import *
from base_functions import base_bias_func

### PROMOTION INTERVENTIONS ###
def random_promotion_func(state, employees, level, identities):
    num_employees = len(employees)
    probability = 1 / num_employees
    promotion_probabilities = [probability] * num_employees
    return promotion_probabilities

def performance_promotion_func(state, employees, level, identities):
    num_employees = len(employees)
    promotion_probabilities = [0] * num_employees
    highest_index = max(enumerate(employees), key=lambda e: e[1].performance_level)[0]
    promotion_probabilities[highest_index] = 1
    return promotion_probabilities

def seniority_promotion_func(state, employees, level, identities):
    num_employees = len(employees)
    promotion_probabilities = [0] * num_employees
    highest_index = max(enumerate(employees), key=lambda e: e[1].position_experience)[0]
    promotion_probabilities[highest_index] = 1
    return promotion_probabilities

### HIRING INTERVENTIONS ###
def uniform_hire_func(state, identities):
    identity_probabilities = [1 / len(identities)] * len(identities)
    return identity_probabilities

def population_hire_func(state, identities, population_percentages):
    identity_probabilities = []
    for identity in identities:
        identity_probabilities.append(population_percentages.get(identity, 0))
    if sum(identity_probabilities) != 1.0:
        raise ValueError("Population percentages do not sum to 1.")
    return identity_probabilities

### GENERAL INTERVENTIONS ###
def decay_bias_func(employee):
    current_bias = employee.bias_score
    decayed_bias = current_bias * BIAS_DECAY_RATE
    employee.bias_score = decayed_bias
    return base_bias_func(employee)
