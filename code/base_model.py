from model import Model
from constants import RNG, MATERNITY_RETURN, MATERNITY_BIAS

class BaseModel(Model):
    def __init__(
            self, 
            leave_rate,
            maternity_leave_rate,
            quit_func,
            fire_func,
            bias_func,
            identities, 
            identity_probabilities, 
            promotion_probability_func, 
            next_id=0, 
            num_levels=4, 
            level_populations=[50, 25, 10, 5]
            ):
        self.leave_rate = leave_rate
        self.maternity_leave_rate = maternity_leave_rate
        self.quit_func = quit_func
        self.fire_func = fire_func
        self.bias_func = bias_func

        self.identities = identities
        self.identity_probabilities = identity_probabilities
        self.num_levels = num_levels
        self.level_populations = level_populations
        self.promotion_probability_func = promotion_probability_func

        self.next_id = next_id
        self.all_employees = []
        self.time = 0
        self.log = []

    def transition_rate(self, state):
        return sum(self.get_rates(state))
    
    def get_rates(self, state):
        fire_rate, _ = self.fire_func(state)
        quit_rate, _ = self.quit_func(state)
        leave_rate = self.leave_rate * len(state.employees)
        maternity_leave_rate = self.maternity_leave_rate * sum(state.get_count(level, "F") for level in range(self.num_levels))
        return fire_rate, quit_rate, leave_rate, maternity_leave_rate

    def sample_next(self, state, time_delta):
        # Update the state
        state.update(time_delta, self.bias_func)
        if self.time != state.time:
            raise ValueError("Model time does not match state time.")
        
        fire_rate, quit_rate, leave_rate, maternity_leave_rate = self.get_rates(state)
        rate = fire_rate + quit_rate + leave_rate + maternity_leave_rate
        rate_details = (fire_rate, quit_rate, leave_rate, maternity_leave_rate)

        # Determine the event type
        event_prob = RNG.random()
        if event_prob < fire_rate / rate:
            event_type = "fire"
            event_details = self.fire(state)
        elif event_prob < (fire_rate + quit_rate) / rate:
            event_type = "quit"
            event_details = self.quit(state)
        elif event_prob < (fire_rate + quit_rate + leave_rate) / rate:
            event_type = "leave"
            event_details = self.leave(state)
        else:
            event_type = "maternity_leave"
            event_details = self.maternity_leave(state)

        self.log_event(event_type, state.time, event_details, rate_details)
        return state

    def hire(self, state):
        new_id = self.next_id
        self.next_id += 1
        new_employee = state.hire_employee(new_id, self.identities, self.identity_probabilities) # Consider coming up with different ways to assign performance levels
        return [(new_employee, 0)]

    def promote(self, state, level, event_details=[]):
        if level < 1:
            raise ValueError("Cannot promote to lowest level.")
        
        promotable_employees = [e for e in state.employees if e.position_level is not None and e.position_level == level-1]
        if not promotable_employees:
            return None

        # Generate promotion probabilities
        promotion_probabilities = self.promotion_probability_func(state, promotable_employees, level, self.identities)
        employee = RNG.choice(promotable_employees, p=promotion_probabilities)
        state.promote_employee(employee)

        event_details.append((employee, level))

        return event_details + self.hire(state) if level == 1 else self.promote(state, level - 1, event_details) # This might not work as intended
    
    def remove_employee(self, state, employee):
        level = employee.position_level
        state.remove_employee(employee, state.time)
        return self.hire(state) if level == 0 else self.promote(state, level)

    def fire(self, state):
        _, fire_probs = self.fire_func(state)
        for prob in fire_probs:
            if prob < 0 or prob > 1:
                raise ValueError(f"Probabilities must be between 0 and 1 ({prob}).")
        employee_id = RNG.choice(state.employee_ids, p=fire_probs)
        employee = state.get_employee(employee_id)
        return self.remove_employee(state, employee)
    
    def quit(self, state):
        _, quit_probs = self.quit_func(state)
        employee_id = RNG.choice(state.employee_ids, p=quit_probs)
        employee = state.get_employee(employee_id)
        return self.remove_employee(state, employee)
    
    def leave(self, state):
        employee = RNG.choice(state.employees)
        return self.remove_employee(state, employee)
    
    def maternity_leave(self, state):
        female_employees = [employee for employee in state.employees if employee.identity == "F"]
        employee = RNG.choice(female_employees)
        employee.update_bias(MATERNITY_BIAS, 1)
        return self.remove_employee(state, employee) if RNG.random() > MATERNITY_RETURN else [] 
    
    def log_event(self, event_type, time, event_details, rate_details):
        self.log.append((event_type, time, event_details, rate_details))
        


