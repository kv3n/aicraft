from PIL import Image, ImageDraw, ImageFont
from schedule_solver import Airport
import random


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

        self.plane_ico = Image.open('viz/plane.png').resize((48, 48))
        self.plane_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

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
        y = (self.state_resource_idx + 1) * airport_viz.line_spacing

        if self.state == 1:
            x = airport_viz.landing_offset + 24 + (
                        airport_viz.cur_time - self.landing) * 1.0 * (airport_viz.landing_viz-48) / (self.at_gate - self.landing)
        elif self.state == 2:
            x = airport_viz.gate_offset + 24 + (
                        airport_viz.cur_time - self.at_gate) * 1.0 * (airport_viz.gate_viz-48) / (self.takeoff - self.at_gate)
        elif self.state == 3:
            x = airport_viz.takeoff_offset + 24 + (
                        airport_viz.cur_time - self.takeoff) * 1.0 * (airport_viz.takeoff_viz-48) / (self.clear - self.takeoff)

        draw_canvas.bitmap(xy=(x-24, y-24), bitmap=self.plane_ico, fill=self.plane_color)


class AirportViz:
    def __init__(self, airport, schedule_file):
        self.airport = airport
        self.cur_time = 0
        self.sim_done = False
        self.landing = [False for _ in xrange(self.airport.L)]
        self.gate = [False for _ in xrange(self.airport.G)]
        self.takeoff = [False for _ in xrange(self.airport.T)]

        self.max_landing = 0
        self.landing_viz = 341
        self.landing_offset = 0

        self.max_gate = 0
        self.gate_viz = 342
        self.gate_offset = 341

        self.max_takeoff = 0
        self.takeoff_viz = 341
        self.takeoff_offset = 683

        self.line_spacing = 1024.0 / (max(airport.L, airport.G, airport.T) + 2.0)

        self.plane_viz = []
        with open(schedule_file, 'rU') as fp:
            for idx, line in enumerate(fp.readlines()):
                plane = self.airport.planes[idx]
                schedule = (int(line.split(' ')[0]), int(line.split(' ')[1]))
                self.plane_viz.append(PlaneViz(idx, schedule, plane))
                self.max_landing = max(self.max_landing, schedule[0] + plane.M)
                self.max_gate = max(self.max_gate, schedule[1])
                self.max_takeoff = max(self.max_takeoff, schedule[1] + plane.O)

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

    def draw(self):
        img = Image.new('RGB', (1024, 1024), (0, 0, 0))
        draw_canvas = ImageDraw.Draw(img)

        draw_canvas.rectangle(xy=[self.landing_offset, 0, self.landing_viz, 1024],
                              fill=(160, 160, 160))
        draw_canvas.rectangle(xy=[self.gate_offset, 0, self.gate_offset+self.gate_viz, 1024],
                              fill=(200, 200, 200))
        draw_canvas.rectangle(xy=[self.takeoff_offset, 0, self.takeoff_offset+self.takeoff_viz, 1024],
                              fill=(240, 240, 240))

        for idx in xrange(self.airport.L):
            y = (idx + 1) * self.line_spacing
            draw_canvas.line(xy=[self.landing_offset, y, self.landing_offset+self.landing_viz, y],
                             fill=(80, 80, 80), width=2)

        for idx in xrange(self.airport.G):
            y = (idx + 1) * self.line_spacing
            draw_canvas.line(xy=[self.gate_offset, y, self.gate_offset+self.gate_viz, y],
                             fill=(0, 0, 0), width=2)

        for idx in xrange(self.airport.T):
            y = (idx + 1) * self.line_spacing
            draw_canvas.line(xy=[self.takeoff_offset, y, self.takeoff_offset+self.takeoff_viz, y],
                             fill=(40, 40, 40), width=2)

        for plane in self.plane_viz:
            plane.draw_plane(self, draw_canvas)

        return img


class ScheduleVisualizer:
    def __init__(self, schedule_file):
        input_file = schedule_file.replace('out_', '')
        self.file_name = schedule_file.split('/')[-1].rstrip('.txt').replace('test_', 'viz_')

        self.airport = Airport(input_file=input_file)
        self.airport_viz = AirportViz(self.airport, schedule_file)

    def run(self):
        frames = []
        while not self.airport_viz.sim_done:
            self.airport_viz.update()
            frames.append(self.airport_viz.draw())

        frames[0].save('viz/' + self.file_name + '.gif',
                       format='GIF', append_images=frames[1:], save_all=True, duration=self.airport.max_time*2, loop=0)


schedule_visualizer = ScheduleVisualizer(schedule_file='test01/out_test_241_28_11_11_9')
schedule_visualizer.run()