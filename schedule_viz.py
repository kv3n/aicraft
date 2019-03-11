import cv2 as cv
import numpy as np
from schedule_solver import Airport
import random

FRAME_WIDTH = 1024
FRAME_HEIGHT = 1024


class PlaneViz:
    def __init__(self, idx, schedule, plane):
        self.idx = idx
        self.landing = schedule[0]
        self.at_gate = self.landing + plane.M
        self.takeoff = schedule[1]
        self.clear = self.takeoff + plane.O

        self.state = 0
        # 0 - Air, 1 - Land, 2 - Gate, 3 - Takeoff, 4 - Clear

        self.state_resource_idx = -1

        self.plane_ico = cv.imread('viz/plane.png', cv.IMREAD_UNCHANGED)
        self.plane_ico[np.where((self.plane_ico == [0, 0, 0, 255]).all(axis=2))] = [random.randint(0, 255),  # Random B
                                                                                    random.randint(0, 255),  # Random G
                                                                                    random.randint(0, 255),  # Random R
                                                                                    255]

        self.plane_ico[np.where((self.plane_ico == [255, 255, 255, 0]).all(axis=2))] = [0, 0, 0, 0]
        self.plane_ico = self.plane_ico[:, :, :3]
        self.plane_ico = cv.resize(self.plane_ico, (48, 48))


    def update(self, airport_viz):
        cur_time = airport_viz.cur_time

        if cur_time == self.landing:
            assert (self.state == 0)
            self.state = 1
        elif cur_time == self.at_gate:
            assert (self.state == 1)
            self.state = 2
        elif cur_time == self.takeoff:
            assert (self.state == 2)
            self.state = 3
        elif cur_time == self.clear:
            assert(self.state == 3)
            self.state = 4

    def assign_state_resource_idx(self, state_resource_idx):
        self.state_resource_idx = state_resource_idx

    def resign_state_resource(self):
        self.state_resource_idx = -1

    def draw_plane(self, airport_viz, draw_canvas):
        if self.state == 0 or self.state == 4:
            return

        x = 0
        y = 0
        if self.state == 1:
            y = airport_viz.landing_strips[self.state_resource_idx][0]
            x = airport_viz.landing_strips[self.state_resource_idx][1]
            scale = 1.0 / (self.at_gate - self.landing)

            x = x + 24 + (airport_viz.cur_time - self.landing) * scale * (airport_viz.landing_viz_width-48)
        elif self.state == 2:
            y = airport_viz.gates[self.state_resource_idx][0]
            x = airport_viz.gates[self.state_resource_idx][1]
            scale = int(60.0 * (airport_viz.cur_time - self.at_gate) / (self.takeoff - self.at_gate) - 30.0)

            draw_canvas = cv.rectangle(draw_canvas,
                                       (x-30, y-30), (x+scale, y+30),
                                       (126, 226, 128), cv.FILLED)

        elif self.state == 3:
            y = airport_viz.takeoff_strips[self.state_resource_idx][0]
            x = airport_viz.takeoff_strips[self.state_resource_idx][1]
            scale = 1.0 / (self.clear - self.takeoff)

            x = x + 24 + (airport_viz.cur_time - self.takeoff) * scale * (airport_viz.takeoff_viz_width-48)

        x = int(x) - 24
        y = y - 24

        for y_idx in range(0, 48):
            for x_idx in range(0, 48):
                if np.allclose(self.plane_ico[y_idx, x_idx, :], [0, 0, 0]):
                    continue
                draw_canvas[y+y_idx, x+x_idx] = self.plane_ico[y_idx, x_idx]


class AirportViz:
    def __init__(self, airport, schedule_file):
        self.airport = airport
        self.cur_time = 0
        self.sim_done = False
        self.landing = [False for _ in range(self.airport.L)]
        self.gate = [False for _ in range(self.airport.G)]
        self.takeoff = [False for _ in range(self.airport.T)]

        per_segment_width = FRAME_WIDTH // 8
        self.max_landing = 0
        self.landing_offset = 0
        self.landing_viz_width = per_segment_width * 3
        self.landing_viz_end = self.landing_offset + self.landing_viz_width

        self.max_gate = 0
        self.gate_offset = self.landing_viz_end
        self.gate_viz_end = self.gate_offset + per_segment_width * 2

        self.max_takeoff = 0
        self.takeoff_offset = self.gate_viz_end
        self.takeoff_viz_end = FRAME_WIDTH
        self.takeoff_viz_width = self.takeoff_viz_end - self.takeoff_offset

        self.line_spacing = 1024.0 / (max(airport.L, airport.G, airport.T) + 2.0)

        self.plane_viz = []

        self.gates = []  # XY for Gate Center
        self.landing_strips = []  # XY for landing starting
        self.takeoff_strips = []  # XY for takeoff starting
        self.do_viz_init()

        with open(schedule_file, 'rU') as fp:
            for idx, line in enumerate(fp.readlines()):
                plane = self.airport.planes[idx]
                schedule = (int(line.split(' ')[0]), int(line.split(' ')[1]))
                self.plane_viz.append(PlaneViz(idx, schedule, plane))
                self.max_landing = max(self.max_landing, schedule[0] + plane.M)
                self.max_gate = max(self.max_gate, schedule[1])
                self.max_takeoff = max(self.max_takeoff, schedule[1] + plane.O)

    def do_viz_init(self):
        for idx in range(self.airport.L):
            y = int((idx + 1) * self.line_spacing)
            self.landing_strips.append([y, self.landing_offset])

        for idx in range(self.airport.G):
            y = int((idx + 1) * self.line_spacing)
            x = int((self.gate_offset + self.gate_viz_end) // 2)
            self.gates.append([y, x])

        for idx in range(self.airport.T):
            y = int((idx + 1) * self.line_spacing)
            self.takeoff_strips.append([y, self.takeoff_offset])

    def update(self):
        num_planes_clear = 0

        # Update and undo resources here
        for plane in self.plane_viz:
            prev_state = plane.state

            plane.update(self)

            if prev_state == plane.state:
                continue

            if plane.state == 2:
                self.landing[plane.state_resource_idx] = False
                plane.resign_state_resource()
            elif plane.state == 3:
                self.gate[plane.state_resource_idx] = False
                plane.resign_state_resource()
            elif plane.state == 4:
                self.takeoff[plane.state_resource_idx] = False
                plane.resign_state_resource()

        # Assign resources here
        for plane in self.plane_viz:
            if plane.state == 4:
                # is clear
                num_planes_clear += 1
                continue

            if plane.state_resource_idx >= 0:
                continue

            if plane.state == 1:
                # is landing
                resource_idx = self.landing.index(False)
                plane.assign_state_resource_idx(resource_idx)
                self.landing[resource_idx] = True
            elif plane.state == 2:
                # is gating
                resource_idx = self.gate.index(False)
                plane.assign_state_resource_idx(resource_idx)
                self.gate[resource_idx] = True
            elif plane.state == 3:
                # is takeoff
                resource_idx = self.takeoff.index(False)
                plane.assign_state_resource_idx(resource_idx)
                self.takeoff[resource_idx] = True

        self.cur_time += 1

        if num_planes_clear == self.airport.N:
            self.sim_done = True

    def draw(self, draw_canvas):
        draw_canvas = cv.rectangle(draw_canvas,
                                   (self.landing_offset, 0), (self.landing_viz_end, FRAME_HEIGHT),
                                   (160, 160, 160), cv.FILLED)
        draw_canvas = cv.rectangle(draw_canvas,
                                   (self.gate_offset, 0), (self.gate_viz_end, FRAME_HEIGHT),
                                   (200, 200, 200), cv.FILLED)
        draw_canvas = cv.rectangle(draw_canvas,
                                   (self.takeoff_offset, 0), (self.takeoff_viz_end, FRAME_HEIGHT),
                                   (240, 240, 240), cv.FILLED)

        font = cv.FONT_HERSHEY_SIMPLEX
        info = 'L: {} G: {} T: {} N: {} Time: {}'.format(self.airport.L, self.airport.G, self.airport.T, self.airport.N, self.cur_time)
        draw_canvas = cv.putText(draw_canvas, info, (10, 20), font, 0.5, (255, 255, 255), 1, cv.LINE_AA)

        for coords in self.landing_strips:
            x = coords[1]
            y = coords[0]
            draw_canvas = cv.line(draw_canvas,
                                  (x, y), (self.landing_viz_end, y),
                                  (40, 40, 40), 3)

        for coords in self.gates:
            x = coords[1]
            y = coords[0]
            draw_canvas = cv.rectangle(draw_canvas,
                                       (x-30, y-30), (x+30, y+30),
                                       (47, 47, 211), cv.FILLED)

        for coords in self.takeoff_strips:
            x = coords[1]
            y = coords[0]
            draw_canvas = cv.line(draw_canvas,
                                  (x, y), (self.takeoff_viz_end, y),
                                  (80, 80, 80), 3)

        for plane in self.plane_viz:
            plane.draw_plane(self, draw_canvas)


class ScheduleVisualizer:
    def __init__(self, schedule_file, input_file='', show=False):
        if input_file == '':
            input_file = schedule_file.replace('out_', '')
            self.file_name = schedule_file.split('/')[-1].rstrip('.txt').replace('test_', 'viz_')
        else:
            self.file_name = 'viz'

        self.airport = Airport(input_file=input_file)
        self.airport_viz = AirportViz(self.airport, schedule_file)
        self.show_viz = show

    def run(self):
        schedule_video = cv.VideoWriter('viz/{}.mp4'.format(self.file_name),
                                        cv.VideoWriter_fourcc(*'x264'),
                                        30.0,
                                        (FRAME_WIDTH, FRAME_HEIGHT))

        while not self.airport_viz.sim_done:
            frame = np.zeros((FRAME_WIDTH, FRAME_HEIGHT, 3), np.uint8)
            self.airport_viz.update()
            self.airport_viz.draw(frame)

            schedule_video.write(frame)

            if self.show_viz:
                cv.imshow('Animation', frame)
                cv.waitKey(1)

        schedule_video.release()
        cv.destroyAllWindows()


schedule_visualizer = ScheduleVisualizer(schedule_file='output.txt', input_file='input.txt', show=True)
schedule_visualizer.run()