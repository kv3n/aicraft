import random
import copy
from math import exp, log
from collections import defaultdict
from itertools import chain

from matplotlib import pyplot as plt


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

    def __str__(self):
        return 'L:{} G:{} T:{}'.format(self.L, self.G, self.T) + '\n' + str(self.planes)


GlobalAirport = Airport()


class Citizen:
    def __init__(self, is_mutation=True):
        self.schedule = [(0, 0)] * GlobalAirport.N
        self.fitness_score = 0

        self.landing = defaultdict(int)
        self.gate = defaultdict(int)
        self.takeoff = defaultdict(int)

        if not is_mutation:
            self.do_complete_random_assignment()

    def do_random_assignment(self, idx):
        plane = GlobalAirport.planes[idx]
        stl = random.randint(0, plane.R)
        time_at_gate = stl + plane.M
        tot = random.randint(time_at_gate + plane.S, time_at_gate + plane.C)

        self.schedule[idx] = (stl, tot)

    def do_complete_random_assignment(self):
        for idx, _ in enumerate(GlobalAirport.planes):
            self.do_random_assignment(idx)

        self.get_fitness_score()

    def do_mutation(self):
        mutation_idx = random.randint(0, GlobalAirport.N - 1)

        plane = GlobalAirport.planes[mutation_idx]
        num_total_conflicts = self.fitness_score

        # To Undo previous conflicts
        old_stl = self.schedule[mutation_idx][0]
        old_tot = self.schedule[mutation_idx][1]
        old_gate = old_stl + plane.M

        self.do_random_assignment(mutation_idx)

        # To Redo new conflicts
        stl = self.schedule[mutation_idx][0]
        tot = self.schedule[mutation_idx][1]
        gate = stl + plane.M

        # Landing
        for minute in chain(xrange(old_stl, stl), xrange(gate, old_gate)):
            self.landing[minute] -= 1
            if self.landing[minute] >= GlobalAirport.L:
                num_total_conflicts -= self.landing[minute]

        for minute in chain(xrange(old_gate, gate), xrange(stl, old_stl)):
            self.landing[minute] += 1
            if self.landing[minute] > GlobalAirport.L:
                num_total_conflicts += self.landing[minute] - 1

        # Gate
        for minute in xrange(old_gate, old_tot):
            self.gate[minute] -= 1
            if self.gate[minute] >= GlobalAirport.G:
                num_total_conflicts -= self.gate[minute]

        for minute in xrange(gate, tot):
            self.gate[minute] += 1
            if self.gate[minute] > GlobalAirport.G:
                num_total_conflicts += self.gate[minute] - 1

        # Takeoff
        for minute in chain(xrange(old_tot, tot), xrange(tot + plane.O, old_tot + plane.O)):
            self.takeoff[minute] -= 1
            if self.takeoff[minute] >= GlobalAirport.T:
                num_total_conflicts -= self.takeoff[minute]

        for minute in chain(xrange(old_tot + plane.O, tot + plane.O), xrange(tot, old_tot)):
            self.takeoff[minute] += 1
            if self.takeoff[minute] > GlobalAirport.T:
                num_total_conflicts += self.takeoff[minute] - 1

        self.fitness_score = num_total_conflicts

    def get_fitness_score(self):
        num_total_conflicts = 0

        # Count number of conflicting minutes
        for idx, plane in enumerate(GlobalAirport.planes):
            stl = self.schedule[idx][0]
            tot = self.schedule[idx][1]
            for minute in xrange(stl, stl + plane.M):
                self.landing[minute] += 1
                if self.landing[minute] > GlobalAirport.L:
                    num_total_conflicts += self.landing[minute] - 1

            for minute in xrange(stl + plane.M, tot):
                self.gate[minute] += 1
                if self.gate[minute] > GlobalAirport.G:
                    num_total_conflicts += self.gate[minute] - 1

            for minute in xrange(tot, tot + plane.O):
                self.takeoff[minute] += 1
                if self.takeoff[minute] > GlobalAirport.T:
                    num_total_conflicts += self.takeoff[minute] - 1

        self.fitness_score = num_total_conflicts

    def __lt__(self, other):
        return self.fitness_score < other.fitness_score

    def output_schedule(self):
        with open('output.txt', 'w') as fp:
            for plane_schedule in self.schedule[:-1]:
                fp.write('{} {}\n'.format(plane_schedule[0], plane_schedule[1]))
            fp.write('{} {}'.format(self.schedule[-1][0], self.schedule[-1][1]))

    def get_key(self):
        key = ''
        for plane_schedule in self.schedule[:-1]:
            key += '{}-{}-'.format(plane_schedule[0], plane_schedule[1])
        key += '{}-{}'.format(self.schedule[-1][0], self.schedule[-1][1])

        return key


class GASolver:
    def __init__(self, max_population_size):
        self.population = [Citizen(is_mutation=False) for _ in xrange(max_population_size)]
        self.max_population_size = max_population_size

    def __str__(self):
        printstr = ''
        for population in self.population:
            printstr += str(population.fitness_score) + '\n'
        return printstr

    def solve(self):
        num_iterations = 0
        # This should change according to new finding
        bad_citizen_tolerance = GlobalAirport.max_time * GlobalAirport.N * (GlobalAirport.N - 1) * 0.5  # 0.25 because 50% conflicts
        alpha = 0.86
        while True:
            for idx, citizen in enumerate(self.population):
                if citizen.fitness_score == 0:
                    return citizen

                mutated_citizen = copy.deepcopy(citizen)
                mutated_citizen.do_mutation()

                # The new score should be less than the old score. For us 0 => Good
                mutation_score = citizen.fitness_score - mutated_citizen.fitness_score

                if mutation_score > 0:  # Should this be greater than or greater than equal to?
                    self.population[idx] = mutated_citizen
                else:
                    acceptance_prob = exp(mutation_score / bad_citizen_tolerance)   # * num_iterations  # Change this
                    if random.random() < acceptance_prob:
                        self.population[idx] = mutated_citizen

            print 'Ran {} with {}'.format(num_iterations, self.population[0].fitness_score)

            bad_citizen_tolerance = bad_citizen_tolerance * alpha
            num_iterations += 1

        return None

    def evaluate_fitness_fn(self):
        num_iterations = 100000
        citizen_set = set()
        score_dict = defaultdict(int)
        while num_iterations > 0:
            print num_iterations
            citizen = Citizen(is_mutation=False)

            key = citizen.get_key()
            if key not in citizen_set:
                citizen_set.add(key)
                score_dict[citizen.fitness_score] += 1

            num_iterations -= 1

        x = score_dict.keys()
        y = score_dict.values()

        max_idx = y.index(max(y))
        print('Max: {} of {} scores found'.format(y[max_idx], x[max_idx]))

        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

        # We can set the number of bins with the `bins` kwarg
        #axs[0].hist(x, bins=len(score_dict))
        #axs[1].hist(y, bins=100)

        axs.hist2d(x, y)

        plt.show()



solver = GASolver(GlobalAirport.N)
solver.evaluate_fitness_fn()

#solution = solver.solve()
#solution.output_schedule()

"""
airport.planes[0].STL = 0
airport.planes[0].TOT = 60
airport.planes[1].STL = 10
airport.planes[1].TOT = 80
airport.planes[2].STL = 50
airport.planes[2].TOT = 130
airport.planes[3].STL = 70
airport.planes[3].TOT = 150
"""

