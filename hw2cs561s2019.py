import random
import time
from collections import defaultdict


class Plane:
    def __init__(self, plane_config):
        config_split = plane_config.split(' ')
        self.R = int(config_split[0])
        self.M = int(config_split[1])
        self.S = int(config_split[2])
        self.O = int(config_split[3])
        self.C = int(config_split[4])

        self.STL = 0
        self.TOT = 0

        self.generate_random_assignment()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '{} {}'.format(self.STL, self.TOT)

    def generate_random_assignment(self):
        self.STL = random.randint(0, self.R)
        time_at_gate = self.STL + self.M
        self.TOT = random.randint(time_at_gate + self.S, time_at_gate + self.C)


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

    def check_schedule_conflicts(self):
        landing = defaultdict(int)
        gate = defaultdict(int)
        takeoff = defaultdict(int)

        landing_log = defaultdict(set)
        gate_log = defaultdict(set)
        takeoff_log = defaultdict(set)

        num_landing_conflicts = 0
        num_gate_conflicts = 0
        num_takeoff_conflicts = 0

        landing_conflicts = set()
        gate_conflicts = set()
        takeoff_conflicts = set()

        for idx, plane in enumerate(self.planes):
            for time in xrange(plane.STL, plane.STL + plane.M):
                landing[time] += 1
                landing_log[time].add(idx)

                if landing[time] > self.L:
                    num_landing_conflicts += landing[time] - 1
                    landing_conflicts = set(list(landing_log[time]))

            for time in xrange(plane.STL + plane.M, plane.TOT):
                gate[time] += 1
                gate_log[time].add(idx)

                if gate[time] > self.G:
                    num_gate_conflicts += gate[time] - 1
                    gate_conflicts = set(list(gate_log[time]))

            for time in xrange(plane.TOT, plane.TOT + plane.O):
                takeoff[time] += 1
                takeoff_log[time].add(idx)

                if takeoff[time] > self.T:
                    num_takeoff_conflicts += takeoff[time] - 1
                    takeoff_conflicts = set(list(takeoff_log[time]))

        print('Potential Conflicts: L: {}, G: {}, O: {}'.format(landing_conflicts, gate_conflicts, takeoff_conflicts))
        print('Lc: {}, Gc: {}, Tc: {}'.format(num_landing_conflicts, num_gate_conflicts, num_takeoff_conflicts))
        return num_landing_conflicts == 0 and num_gate_conflicts == 0 and num_takeoff_conflicts == 0

    def refresh_schedule(self):
        for _, plane in enumerate(self.planes):
            plane.generate_random_assignment()

    def output_schedule(self):
        with open('output.txt', 'w') as fp:
            for plane in self.planes[:-1]:
                fp.write(str(plane) + '\n')
            fp.write(str(self.planes[-1]))

    def __str__(self):
        return 'L:{} G:{} T:{}'.format(self.L, self.G, self.T) + '\n' + str(self.planes)


airport = Airport()
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

print(airport)
itr = 0
start_time = time.time()
while not airport.check_schedule_conflicts():
    print(itr)
    airport.refresh_schedule()
    itr += 1

print('Solution Found at {}'.format(time.time() - start_time))
airport.output_schedule()
