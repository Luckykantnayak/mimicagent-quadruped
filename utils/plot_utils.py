import matplotlib.pyplot as plt
from collections import deque
import numpy as np

class FootPlotter:
    """Live plotting of foot positions (X/Z) for 4 legs"""

    def __init__(self, body_names, buffer_len=100, total_time=0.5):
        self.body_names = body_names
        self.buffer_len = buffer_len
        self.total_time = total_time
        self.time_buffer = deque(np.linspace(0, total_time, buffer_len), maxlen=buffer_len)
        self.pos_z_buffers = [deque(np.zeros(buffer_len), maxlen=buffer_len) for _ in body_names]
        self.pos_x_buffers = [deque(np.zeros(buffer_len), maxlen=buffer_len) for _ in body_names]

        self.colors = ["r", "g", "b", "m"]

        self.fig, (self.ax_z, self.ax_x) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        self.lines_z = []
        self.lines_x = []

        for i, name in enumerate(body_names):
            line_z, = self.ax_z.plot(self.time_buffer, self.pos_z_buffers[i], color=self.colors[i], label=f"{name} Z")
            line_x, = self.ax_x.plot(self.time_buffer, self.pos_x_buffers[i], color=self.colors[i], linestyle="--", label=f"{name} X")
            self.lines_z.append(line_z)
            self.lines_x.append(line_x)

        self.ax_z.set_xlim(0, total_time)
        self.ax_z.set_ylim(-0.4, 0.4)
        self.ax_z.set_ylabel("Foot Z height (m)")
        self.ax_z.legend(loc="upper right")

        self.ax_x.set_xlim(0, total_time)
        self.ax_x.set_ylim(-0.5, 0.5)
        self.ax_x.set_xlabel("Time (s)")
        self.ax_x.set_ylabel("Foot X position (m)")
        self.ax_x.legend(loc="upper right")

        plt.tight_layout()
        plt.ion()
        plt.show(block=False)

    def update(self, foot_positions):
        for i in range(len(self.body_names)):
            self.pos_z_buffers[i].append(foot_positions[i, 2])
            self.pos_x_buffers[i].append(foot_positions[i, 0])

        for i in range(len(self.body_names)):
            self.lines_z[i].set_ydata(self.pos_z_buffers[i])
            self.lines_x[i].set_ydata(self.pos_x_buffers[i])

        plt.pause(0.001)
