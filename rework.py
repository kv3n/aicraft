import random
from math import exp, log
import time


class Plane:
    def __init__(self, plane_config):
        config_split = plane_config.split(' ')
        self.R = int(config_split[0])
        self.M = int(config_split[1])
        self.S = int(config_split[2])
        self.O = int(config_split[3])
        self.C = int(config_split[4])

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '{} {} {} {} {}'.format(self.R, self.M, self.S, self.O, self.C)


class Airport:
    def __init__(self):
        with open('input.txt', 'rU') as fp:
            airport_config = fp.readlines()

            config_split = airport_config[0].split(' ')
            self.L = int(config_split[0])
            self.G = int(config_split[1])
            self.T = int(config_split[2])

            self.max_time = 0

            self.N = int(airport_config[1])
            self.planes = []
            for line in airport_config[2:]:
                plane = Plane(line)
                self.planes.append(plane)
                self.max_time = max(self.max_time, plane.R + plane.M + plane.C + plane.O)

            self.min_resource = min(self.L, self.G, self.T)

    def __str__(self):
        return 'L:{} G:{} T:{}'.format(self.L, self.G, self.T) + '\n' + str(self.planes)


GlobalAirport = Airport()


class Thermostat:
    def __init__(self, initial_temp, alpha, update_schedule):
        self.temperature = initial_temp
        self.initial_temp = initial_temp
        self.alpha = alpha
        self.update_schedule = update_schedule

        self.rotations = 0
        self.update_count = 0
        self.tabu_rotations = 0

    def update(self, velocity):
        self.rotations += 1

        if velocity == 0:
            self.tabu_rotations += 1
        else:
            self.tabu_rotations = 0

        if self.tabu_rotations > 1000:
            self.reset()

        if self.rotations % self.update_schedule == 0:
            self.update_count += 1
            self.temperature = self.alpha * self.temperature

    def reset(self):
        self.rotations = 0
        self.tabu_rotations = 0
        self.temperature = self.initial_temp


class Solver:
    def __init__(self, do_assignment=False, solver_type=0):
        self.land_slots = [0] * GlobalAirport.max_time
        self.gate_slots = [0] * GlobalAirport.max_time
        self.takeoff_slots = [0] * GlobalAirport.max_time
        self.schedule = [(0, 0, 0, 0) for _ in xrange(GlobalAirport.N)]

        self.thermostat = Thermostat(initial_temp=10000, alpha=0.88, update_schedule=4)

        if solver_type == 0:
            self.solver_type = 'Combinatorial'
            self.fitness_type = self.combinatorial_slot_fn
        elif solver_type == 1:
            self.solver_type = 'Slot Per Resource'
            self.fitness_type = self.one_slot_per_resource_fn
        elif solver_type == 2:
            self.solver_type = 'Conflict per Slot'
            self.fitness_type = self.oneconflict_perslot_fn
        elif solver_type == 3:
            self.solver_type = 'Extra Resource'
            self.fitness_type = self.extra_resource_fn

        self.fitness_score = 0

        if do_assignment:
            self.do_initial_assignment()
            self.get_fitness_score()

    def do_initial_assignment(self):
        for idx, _ in enumerate(self.schedule):
            plane = GlobalAirport.planes[idx]

            stl = random.randint(0, plane.R)
            time_at_gate = stl + plane.M
            tot = random.randint(time_at_gate + plane.S, time_at_gate + plane.C)
            takeoff = tot + plane.O

            self.register_plane_schedule(idx=idx,
                                         stl=stl,
                                         time_at_gate=time_at_gate,
                                         tot=tot,
                                         takeoff=takeoff)

    def do_pre_assigned_schedule(self):
        solution_schedule = []
        with open('schedule.txt', 'rU') as fp:
            schedule_lines = fp.readlines()
            for schedule_line in schedule_lines:
                solution_schedule.append((int(schedule_line.split(' ')[0]), int(schedule_line.split(' ')[1])))

        for idx, _ in enumerate(self.schedule):
            plane = GlobalAirport.planes[idx]

            stl = solution_schedule[idx][0]
            tot = solution_schedule[idx][1]
            time_at_gate = stl + plane.M
            takeoff = tot + plane.O

            self.register_plane_schedule(idx=idx,
                                         stl=stl,
                                         time_at_gate=time_at_gate,
                                         tot=tot,
                                         takeoff=takeoff)

    def register_plane_schedule(self, idx, stl, time_at_gate, tot, takeoff):
        self.schedule[idx] = stl, time_at_gate, tot, takeoff

        self.land_slots[stl:time_at_gate] = [x+1 for x in self.land_slots[stl:time_at_gate]]
        self.gate_slots[time_at_gate:tot] = [x+1 for x in self.gate_slots[time_at_gate:tot]]
        self.takeoff_slots[tot:takeoff] = [x+1 for x in self.takeoff_slots[tot:takeoff]]

    def unregister_plane_schedule(self, idx):
        stl, time_at_gate, tot, takeoff = self.schedule[idx]

        self.land_slots[stl:time_at_gate] = [x-1 for x in self.land_slots[stl:time_at_gate]]
        self.gate_slots[time_at_gate:tot] = [x-1 for x in self.gate_slots[time_at_gate:tot]]
        self.takeoff_slots[tot:takeoff] = [x-1 for x in self.takeoff_slots[tot:takeoff]]

    def oneconflict_perslot_fn(self, minute):
        slot_conflict_score = 0
        num_landing = self.land_slots[minute]
        num_gate = self.gate_slots[minute]
        num_takeoff = self.takeoff_slots[minute]

        if num_landing > GlobalAirport.L or num_gate > GlobalAirport.G or num_takeoff > GlobalAirport.T:
            slot_conflict_score += 1

        return slot_conflict_score

    def one_slot_per_resource_fn(self, minute):
        slot_conflict_score = 0

        num_landing = self.land_slots[minute]
        num_gate = self.gate_slots[minute]
        num_takeoff = self.takeoff_slots[minute]

        if num_landing > GlobalAirport.L:
            slot_conflict_score += 1
        if num_gate > GlobalAirport.G:
            slot_conflict_score += 1
        if num_takeoff > GlobalAirport.T:
            slot_conflict_score += 1

        return slot_conflict_score

    def extra_resource_fn(self, minute):
        slot_conflict_score = 0

        num_landing = self.land_slots[minute]
        num_gate = self.gate_slots[minute]
        num_takeoff = self.takeoff_slots[minute]

        slot_conflict_score += max(num_landing - GlobalAirport.L, 0)
        slot_conflict_score += max(num_gate - GlobalAirport.G, 0)
        slot_conflict_score += max(num_takeoff - GlobalAirport.T, 0)

        return slot_conflict_score

    def combinatorial_slot_fn(self, minute):
        slot_conflict_score = 0
        num_landing = self.land_slots[minute]
        num_gate = self.gate_slots[minute]
        num_takeoff = self.takeoff_slots[minute]

        if num_landing > GlobalAirport.L:
            slot_conflict_score += ((num_landing * (num_landing - 1)) // 2)

        if num_gate > GlobalAirport.G:
            slot_conflict_score += ((num_gate * (num_gate - 1)) // 2)

        if num_takeoff > GlobalAirport.T:
            slot_conflict_score += ((num_takeoff * (num_takeoff - 1)) // 2)

        return slot_conflict_score

    def get_fitness_score(self):
        num_total_conflicts = 0

        for minute in xrange(GlobalAirport.max_time):
            num_total_conflicts += self.fitness_type(minute)

        self.fitness_score = num_total_conflicts

    def update(self):
        plane_to_update = random.randint(0, GlobalAirport.N-1)
        plane = GlobalAirport.planes[plane_to_update]

        stl, time_at_gate, tot, takeoff = self.schedule[plane_to_update]
        current_score = self.fitness_score

        self.unregister_plane_schedule(plane_to_update)

        new_stl = random.randint(0, plane.R)
        new_time_at_gate = new_stl + plane.M
        new_tot = random.randint(new_time_at_gate + plane.S, new_time_at_gate + plane.C)
        new_takeoff = new_tot + plane.O

        self.register_plane_schedule(idx=plane_to_update,
                                     stl=new_stl,
                                     time_at_gate=new_time_at_gate,
                                     tot=new_tot,
                                     takeoff=new_takeoff)

        self.get_fitness_score()

        rejected = False
        delta = current_score - self.fitness_score
        if delta <= 0:
            acceptance_prob = exp(delta / self.thermostat.temperature)
            if random.random() > acceptance_prob:
                rejected = True

        if rejected:
            self.unregister_plane_schedule(plane_to_update)
            self.register_plane_schedule(idx=plane_to_update,
                                         stl=stl,
                                         time_at_gate=time_at_gate,
                                         tot=tot,
                                         takeoff=takeoff)
            self.fitness_score = current_score

        delta = current_score - self.fitness_score
        velocity = delta / abs(delta) if delta != 0 else 0
        self.thermostat.update(velocity=velocity)

    def output_schedule(self):
        with open('output.txt', 'w') as fp:
            for plane_schedule in self.schedule[:-1]:
                fp.write('{} {}\n'.format(plane_schedule[0], plane_schedule[2]))
            fp.write('{} {}'.format(self.schedule[-1][0], self.schedule[-1][2]))


class SolutionManager:
    def __init__(self):
        self.solvers = [Solver(do_assignment=True, solver_type=0),
                        Solver(do_assignment=True, solver_type=1),
                        Solver(do_assignment=True, solver_type=2),
                        Solver(do_assignment=True, solver_type=3)]

    def solve(self):
        start_time = time.time()
        num_iterations = 0
        while True:
            for idx, solver in enumerate(self.solvers):
                print('iteration {} - solver {} - score {}'.format(num_iterations, idx, solver.fitness_score))
                if solver.fitness_score == 0:
                    print('Finished in {} seconds by solver type {}'.format(time.time() - start_time, solver.solver_type))
                    solver.output_schedule()
                    return

                solver.update()
            num_iterations += 1


solution_manager = SolutionManager()
solution_manager.solve()
