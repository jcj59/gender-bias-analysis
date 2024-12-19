from constants import RNG
from employee import Employee

class State:
    def __init__(self, employees, time=0):
        self.employees = employees
        self.employee_ids = [employee.id for employee in employees]
        self.time = time

    def update(self, delta_t, bias_func):
        self.time += delta_t
        for employee in self.employees:
            employee.update_experience(delta_t)
            employee.update_performance(delta_t)
            employee.update_bias((bias_func(employee)), delta_t)

    def add_employee(self, employee):
        self.employees.append(employee)
        self.employee_ids.append(employee.id)

    def remove_employee(self, employee, time):
        if employee not in self.employees:
            raise ValueError(f"Employee with ID {employee.id} not found in the state.")
        employee.leave(time)
        self.employees.remove(employee)
        self.employee_ids.remove(employee.id)

    def promote_employee(self, employee):
        if employee not in self.employees:
            raise ValueError(f"Employee with ID {employee.id} not found in the state.")
        employee.promote()
    
    def hire_employee(self, new_id, identities, identity_probabilities, position_level=0, performance_mean=0.5, performance_std=0.1):        
        new_employee = Employee.generate_employee(
            id=new_id,
            identities=identities,
            identity_probabilities=identity_probabilities,
            position_level=position_level,
            time=self.time,
            performance_mean=performance_mean,
            performance_std=performance_std,
        )
        
        self.add_employee(new_employee)
        return new_employee
    
    def get_employee(self, id):
        for employee in self.employees:
            if employee.id == id:
                return employee
        raise ValueError(f"Employee with ID {employee.id} not found in the state.")

    def get_count(self, position, identity):
        count = 0
        for employee in self.employees:
            if employee.position_level == position and employee.identity == identity:
                count += 1
        return count

    def get_summary(self):
        summary = {}
        for employee in self.employees:
            position = employee.position_level
            identity = employee.identity
            summary.setdefault(position, {}).setdefault(identity, 0)
            summary[position][identity] += 1
        return summary

    def __str__(self):
        summary = self.get_summary()
        summary_str = "\n".join(
            f"Position Level {level}: " + ", ".join(f"{identity}: {count}" for identity, count in identities.items())
            for level, identities in summary.items()
        )
        return f"Time: {self.time}\nWorkforce Summary:\n{summary_str}"
    
    @staticmethod
    def generate_initial_state(level_populations, identities, identity_probabilities, performance_mean=0.5, performance_std=0.1):
        employees = []
        for level, population in enumerate(level_populations):
            for _ in range(population):
                employees.append(Employee.generate_employee(
                    id=len(employees),
                    identities=identities,
                    identity_probabilities=identity_probabilities,
                    position_level=level,
                    time=0,
                    performance_mean=performance_mean,
                    performance_std=performance_std,
                ))
        return State(employees)

        
