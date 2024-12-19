from constants import *

class Employee:
    def __init__(self, id, identity, performance_level, position_level, time):
        self.id = id
        self.identity = identity
        self.performance_level = max(0, min(1, performance_level))
        self.position_level = position_level
        self.start_time = time
        self.end_time = None

        self.position_experience = 0
        self.company_experience = 0
        self.bias_score = 0
        self.position_history = {self.position_level : self.position_experience}
        self.performance_history = [self.performance_level]

    def promote(self):
        self.position_history[self.position_level] = self.position_experience
        self.position_level += 1
        self.position_experience = 0

    def leave(self, time):
        self.end_time = time
        self.position_history[self.position_level] = self.position_experience
        self.position_experience = 0
        self.position_level = None

    def update_performance(self, delta_t):
        self.performance_level += delta_t * (self.position_experience * PERFORMANCE_INCREASE_RATE - self.bias_score * PERFORMANCE_DECREASE_RATE)
        self.performance_level = max(0, min(1, self.performance_level))
        self.performance_history.append(self.performance_level)

    def update_experience(self, delta_t):
        self.position_experience += delta_t
        self.company_experience += delta_t

    def update_bias(self, bias, delta_t):
        self.bias_score += bias * delta_t * BIAS_RATE_COEFFICIENT

    def __str__(self):
        return (
            f"Employee {self.id}: "
            f"Identity: {self.identity}, "
            f"Position Level: {self.position_level}, "
            f"Performance: {self.performance_level:.2f}, "
            f"Experience: {self.position_experience} (Position), {self.company_experience} (Company), "
            f"Bias Score: {self.bias_score:.2f}, "
        )

    @staticmethod
    def generate_employee(id, identities, identity_probabilities, position_level, time, performance_mean=0.5, performance_std=0.1):
        performance_level = RNG.normal(performance_mean, performance_std)
        identity = RNG.choice(identities, p=identity_probabilities)
        return Employee(id, identity, performance_level, position_level, time)