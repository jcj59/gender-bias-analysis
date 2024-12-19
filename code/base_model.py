from model import Model
from constants import RNG

class BaseModel(Model):
    def __init__(
            self, 
            leave_rate,
            bias_func,
            fire_func,
            identities, 
            identity_probabilities, 
            promotion_probability_func, 
            next_id=0, 
            num_levels=4, 
            level_populations=[50, 25, 10, 5]
            ):
        self.leave_rate = leave_rate
        self.bias_func = bias_func
        self.fire_func = fire_func

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
        quit_rate, _ = self.bias_func(state)
        leave_rate = self.leave_rate * len(state.employees)
        return fire_rate, quit_rate, leave_rate

    def sample_next(self, state, time_delta):
        # Update the state
        state.update(time_delta)
        if self.time != state.time:
            raise ValueError("Model time does not match state time.")
        
        fire_rate, quit_rate, _ = self.get_rates(state)
        rate = fire_rate + quit_rate + self.leave_rate * len(state.employees)
        rate_details = (fire_rate, quit_rate, self.leave_rate * len(state.employees))

        # Determine the event type
        event_prob = RNG.random()
        if event_prob < fire_rate / rate:
            event_type = "fire"
            event_details = self.fire(state)
        elif event_prob < (fire_rate + quit_rate) / rate:
            event_type = "quit"
            event_details = self.quit(state)
        else:
            event_type = "leave"
            event_details = self.leave(state)

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
        employee_id = RNG.choice(state.employee_ids, p=fire_probs)
        employee = state.get_employee(employee_id)
        return self.remove_employee(state, employee)
    
    def quit(self, state):
        _, quit_probs = self.bias_func(state)
        employee_id = RNG.choice(state.employee_ids, p=quit_probs)
        employee = state.get_employee(employee_id)
        return self.remove_employee(state, employee)
    
    def leave(self, state):
        employee = RNG.choice(state.employees)
        return self.remove_employee(state, employee)
    
    def log_event(self, event_type, time, event_details, rate_details):
        self.log.append((event_type, time, event_details, rate_details))
        


