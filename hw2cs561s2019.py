import random
from math import exp, log
from collections import defaultdict


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
        return '{} {} {} {} {} {}'.format(self.R, self.M, self.S, self.O, self.C)


class Airport:
    def __init__(self):
        with open('input.txt', 'rU') as fp:
            airport_config = fp.readlines()

            config_split = airport_config[0].split(' ')
            self.L = int(config_split[0])
            self.G = int(config_split[1])
            self.T = int(config_split[2])

            self.N = int(airport_config[1])
            self.planes = []
            for line in airport_config[2:]:
                self.planes.append(Plane(line))

    def __str__(self):
        return 'L:{} G:{} T:{}'.format(self.L, self.G, self.T) + '\n' + str(self.planes)


class Citizen:
    def __init__(self, airport, is_mutation=True):
        self.airport = airport
        self.schedule = [(0, 0)] * airport.N
        self.fitness_score = 0

        if not is_mutation:
            self.do_complete_random_assignment()

    def do_random_assignment(self, idx):
        plane = self.airport.planes[idx]
        stl = random.randint(0, plane.R)
        time_at_gate = stl + plane.M
        tot = random.randint(time_at_gate + plane.S, time_at_gate + plane.C)

        self.schedule[idx] = (stl, tot)

    def do_complete_random_assignment(self):
        for idx, _ in enumerate(self.airport.planes):
            self.do_random_assignment(idx)

        self.get_fitness_score()

    def do_cross_over(self, parent_x, parent_y):
        cross_over_point = random.randint(1, self.airport.N - 1)
        self.schedule = parent_x.schedule[:cross_over_point] + parent_y.schedule[cross_over_point:]

        self.get_fitness_score()

    def do_mutation(self):
        mutation_idx = random.randint(0, self.airport.N - 1)
        self.do_random_assignment(mutation_idx)

        self.get_fitness_score()

    def get_fitness_score(self):
        landing = defaultdict(int)
        gate = defaultdict(int)
        takeoff = defaultdict(int)

        num_landing_conflicts = 0.0
        num_gate_conflicts = 0.0
        num_takeoff_conflicts = 0.0

        # Count number of conflicting minutes
        for idx, plane in enumerate(self.airport.planes):
            stl = self.schedule[idx][0]
            tot = self.schedule[idx][1]
            for minute in xrange(stl, stl + plane.M):
                if landing[minute] == self.airport.L:
                    num_landing_conflicts += 1.0

                landing[minute] += 1

            for minute in xrange(stl + plane.M, tot):
                if gate[minute] == self.airport.G:
                    num_gate_conflicts += 1.0

                gate[minute] += 1

            for minute in xrange(tot, tot + plane.O):
                if takeoff[minute] == self.airport.T:
                    num_takeoff_conflicts += 1.0

                takeoff[minute] += 1

        self.fitness_score = ((num_landing_conflicts / len(landing)) +
                              (num_gate_conflicts / len(gate)) +
                              (num_takeoff_conflicts / len(takeoff))) / 3

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
        self.airport = Airport()
        self.population = [Citizen(self.airport, is_mutation=False) for _ in xrange(max_population_size)]
        self.max_population_size = max_population_size

        self.cdf = [0.0] * max_population_size + [1.0]
        self.build_cdf()

    def build_cdf(self):
        lambda_factor = -log(1 - 0.7) / (0.4 * self.max_population_size)
        for x, cdf_x in enumerate(self.cdf):
            self.cdf[x] = 1 - exp(-lambda_factor * x)

    def get_random_parents(self):
        def find_idx(val, left, right):
            while left <= right:
                mid = (left + right) // 2
                if val < self.cdf[mid]:
                    right = mid - 1
                else:
                    left = mid + 1

            return right

        dice_roll = random.random()
        parent_x = find_idx(dice_roll, 0, self.max_population_size - 1)

        pre_cdf = self.cdf[parent_x + 1]
        self.cdf[parent_x + 1] = self.cdf[parent_x]

        dice_roll = random.random()
        parent_y = find_idx(dice_roll, 0, self.max_population_size - 1)

        self.cdf[parent_x + 1] = pre_cdf

        return parent_x, parent_y

    def __str__(self):
        printstr = ''
        for population in self.population:
            printstr += str(population.fitness_score) + '\n'
        return printstr

    def do_natural_selection(self):
        self.population.sort()
        self.population = self.population[:self.max_population_size]

        return True

    def solve(self, breeding_factor=5):
        num_iterations = 0
        num_offsprings = self.max_population_size * breeding_factor
        while self.do_natural_selection() and self.population[0].fitness_score > 0.0:
            print 'Running Iteration: {}, lowest: {}'.format(num_iterations, self.population[0].fitness_score)

            offsprings = []
            offsprings_hashing = set()
            while len(offsprings_hashing) < num_offsprings:
                parent_x, parent_y = self.get_random_parents()

                young_citizen = Citizen(self.airport)
                young_citizen.do_cross_over(self.population[parent_x], self.population[parent_y])

                if random.random() > 0.6:
                    young_citizen.do_mutation()

                if young_citizen.get_key() not in offsprings_hashing:
                    offsprings_hashing.add(young_citizen.get_key())
                    offsprings.append(young_citizen)

            self.population = offsprings

            num_iterations += 1

        return self.population[0]


solver = GASolver(300)
solution = solver.solve()
solution.output_schedule()

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

