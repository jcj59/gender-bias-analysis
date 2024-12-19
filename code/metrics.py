import numpy as np

### METRICS ###
def naive_bias_metric(state, identities, level=None):
    """
    Calculate the naive bias for the entire company or a specific level.

    Naive bias is the squared difference between the actual and uniform share of each identity group.
    """
    employees = state.employees if level is None else [e for e in state.employees if e.position_level == level]
    total_employees = len(employees)
    if total_employees == 0:
        return 0  # No employees at this level or in the company

    expected_share = 1 / len(identities)
    identity_counts = {identity: 0 for identity in identities}
    
    for employee in employees:
        identity_counts[employee.identity] += 1
    
    bias = sum(
        ((identity_counts[identity] / total_employees) - expected_share) ** 2
        for identity in identities
    )
    return bias

def population_bias_metric(state, identities, general_population_percentages, level=None):
    """
    Calculate population bias for the entire company or a specific level.

    Population bias is the squared difference between the actual and general population share of each identity group.
    """
    employees = state.employees if level is None else [e for e in state.employees if e.position_level == level]
    total_employees = len(employees)
    if total_employees == 0:
        return 0  # No employees at this level or in the company

    identity_counts = {identity: 0 for identity in identities}
    
    for employee in employees:
        identity_counts[employee.identity] += 1
    
    bias = sum(
        ((identity_counts[identity] / total_employees) - general_population_percentages.get(identity, 0)) ** 2
        for identity in identities
    )
    return bias

def performance_metric(state, level_weights=None, level=None):
    """
    Calculate weighted average performance for the entire company or a specific level.

    Performance is weighted by the position level of each employee.
    """
    employees = state.employees if level is None else [e for e in state.employees if e.position_level == level]
    if level_weights is None:
        level_weights = {level: level + 1 for level in range(max(e.position_level for e in state.employees) + 1)}

    weighted_performance_sum = 0
    total_weight = 0
    
    for employee in employees:
        if employee.position_level is not None:
            weight = level_weights.get(employee.position_level, employee.position_level + 1)
            weighted_performance_sum += weight * employee.performance_level
            total_weight += weight
    
    return weighted_performance_sum / total_weight if total_weight > 0 else 0

def average_company_experience(state, level=None):
    """
    Calculate the average company experience for the entire company or a specific level.
    """
    employees = state.employees if level is None else [e for e in state.employees if e.position_level == level]
    total_employees = len(employees)
    
    if total_employees == 0:
        return 0  # No employees at this level or in the company

    total_experience = sum(employee.company_experience for employee in employees)
    return total_experience / total_employees


def calculate_average_company_experience_over_path(path):
    """
    Calculate the average company experience over the entire path.
    """
    timestamps = [timestamp for timestamp, _ in path]
    experiences = {"company": []}

    # Initialize metrics for each level
    levels = {employee.position_level for _, state in path for employee in state.employees if employee.position_level is not None}
    for level in levels:
        experiences[level] = []

    for timestamp, state in path:
        # Company-wide metric
        experiences["company"].append(average_company_experience(state))

        # Per-level metrics
        for level in levels:
            experiences[level].append(average_company_experience(state, level=level))

    return {
        "timestamps": timestamps,
        "experiences": experiences,
    }

def calculate_metrics_over_path(path, identities, general_population_percentages, level_weights=None, tolerance=0.01):
    timestamps = [timestamp for timestamp, _ in path]
    naive_biases = {"company": []}
    population_biases = {"company": []}
    performances = {"company": []}
    experiences = {"company": []}

    # Initialize metrics for each level
    levels = {employee.position_level for _, state in path for employee in state.employees if employee.position_level is not None}
    for level in levels:
        naive_biases[level] = []
        population_biases[level] = []
        performances[level] = []
        experiences[level] = []

    for timestamp, state in path:
        # Company-wide metrics
        naive_biases["company"].append(naive_bias_metric(state, identities))
        population_biases["company"].append(population_bias_metric(state, identities, general_population_percentages))
        performances["company"].append(performance_metric(state, level_weights))
        experiences["company"].append(average_company_experience(state))

        # Per-level metrics
        for level in levels:
            naive_biases[level].append(naive_bias_metric(state, identities, level=level))
            population_biases[level].append(population_bias_metric(state, identities, general_population_percentages, level=level))
            performances[level].append(performance_metric(state, level_weights, level=level))
            experiences[level].append(average_company_experience(state, level=level))

    return {
        "timestamps": timestamps,
        "naive_biases": naive_biases,
        "population_biases": population_biases,
        "performances": performances,
        "experiences": experiences,
    }

### AVERAGES ###
def compute_weighted_averages(metrics, timestamps, identity_percentages):
    """
    Compute the weighted average of each metric over the entire run.

    Parameters:
        metrics (dict): A dictionary containing all metrics over the path. It should include:
                        "naive_biases", "population_biases", "performances", "experiences".
        timestamps (list): List of timestamps corresponding to the metrics.
        identity_percentages (dict): A dictionary of identity percentages for each timestamp.

    Returns:
        dict: A dictionary containing the weighted average of each metric for the company and each level,
              and the average population percentage of each identity.
    """
    averages = {
        "naive_biases": {},
        "population_biases": {},
        "performances": {},
        "experiences": {},
        "identity_percentages": {"company": {}, "levels": {}}
    }

    # Calculate delta_ts
    delta_ts = np.diff(timestamps, prepend=timestamps[0])

    # Compute weighted averages for each metric
    for metric_name in ["naive_biases", "population_biases", "performances", "experiences"]:
        metric_data = metrics[metric_name]

        for level, values in metric_data.items():
            if len(values) == len(delta_ts):
                weighted_sum = np.sum(np.array(values) * delta_ts)
                total_time = np.sum(delta_ts)
                averages[metric_name][level] = weighted_sum / total_time if total_time > 0 else 0
            else:
                averages[metric_name][level] = 0  # Handle mismatched lengths

    # Compute average identity percentages for the company
    identity_totals = {identity: 0 for identity in identity_percentages["company"].keys()}
    total_time = np.sum(delta_ts)

    for identity in identity_totals:
        weighted_sum = np.sum(
            np.array(identity_percentages["company"][identity]) * delta_ts
        )
        identity_totals[identity] = weighted_sum / total_time if total_time > 0 else 0

    averages["identity_percentages"]["company"] = identity_totals

    # Compute average identity percentages for each level
    levels = identity_percentages.keys() - {"company"}
    for level in levels:
        level_totals = {identity: 0 for identity in identity_percentages[level].keys()}
        for identity in level_totals:
            weighted_sum = np.sum(
                np.array(identity_percentages[level][identity]) * delta_ts
            )
            level_totals[identity] = weighted_sum / total_time if total_time > 0 else 0
        averages["identity_percentages"]["levels"][level] = level_totals

    return averages

# Example usage
def calculate_metrics_with_weighted_averages(
    path, identities, general_population_percentages, level_weights=None, tolerance=0.01
):
    """
    Calculate all metrics over the path and their weighted averages.

    Parameters:
        path (list): List of (timestamp, state) tuples.
        identities (list): List of identity groups (e.g., ["F", "M"]).
        general_population_percentages (dict): General population percentages for each identity.
        level_weights (dict, optional): Weights for each level. Defaults to None.
        tolerance (float, optional): Convergence tolerance. Defaults to 0.01.

    Returns:
        dict: A dictionary containing all metrics over the path and their weighted averages.
    """
    # Extract timestamps
    timestamps = [timestamp for timestamp, _ in path]

    # Calculate metrics over the path
    metrics = calculate_metrics_over_path(
        path, identities, general_population_percentages, level_weights, tolerance
    )

    # Calculate identity percentages over the path
    identity_percentages = calculate_identity_percentages_over_path(path, identities)

    # Compute weighted averages
    weighted_averages = compute_weighted_averages(metrics, timestamps, identity_percentages["percentages"])

    metrics["weighted_averages"] = weighted_averages
    return metrics

### IDENTITY PERCENTAGES ###
def identity_percentages(state, identities, level=None):
    """
    Calculate the percentage of each identity for the entire company or a specific level.

    Parameters:
        state (State): The current state of the simulation.
        identities (list): List of identity groups (e.g., ["F", "M"]).
        level (int, optional): Level to calculate the metric for. Defaults to None (entire company).

    Returns:
        dict: A dictionary with identity percentages.
    """
    employees = state.employees if level is None else [e for e in state.employees if e.position_level == level]
    total_employees = len(employees)
    
    if total_employees == 0:
        return {identity: 0 for identity in identities}

    identity_counts = {identity: 0 for identity in identities}
    for employee in employees:
        identity_counts[employee.identity] += 1
    
    percentages = {identity: identity_counts[identity] / total_employees for identity in identities}
    return percentages

def calculate_identity_percentages_over_path(path, identities):
    """
    Calculate identity percentages over the entire simulation path.

    Parameters:
        path (list): List of (timestamp, state) tuples.
        identities (list): List of identity groups (e.g., ["F", "M"]).

    Returns:
        dict: Identity percentages for each level and the entire company over time.
    """
    timestamps = [timestamp for timestamp, _ in path]
    percentages = {"company": {identity: [] for identity in identities}}

    # Determine all levels
    levels = {employee.position_level for _, state in path for employee in state.employees if employee.position_level is not None}
    for level in levels:
        percentages[level] = {identity: [] for identity in identities}

    for _, state in path:
        # Company-wide percentages
        company_percentages = identity_percentages(state, identities)
        for identity in identities:
            percentages["company"][identity].append(company_percentages[identity])

        # Per-level percentages
        for level in levels:
            level_percentages = identity_percentages(state, identities, level=level)
            for identity in identities:
                percentages[level][identity].append(level_percentages[identity])

    return {
        "timestamps": timestamps,
        "percentages": percentages,
    }

