import numpy as np
import mujoco
from utils import euler_to_quat

# def delta_vector(vx, theta, dt):
#     # TO-DO :  Not the right expression for getting the delta when yaw_rate is  non zero
#     del_x =  vx * np.cos(theta) * dt
#     del_y =  vx * np.sin(theta) * dt
#     return np.array([del_x, del_y, 0])

def delta_vector(vx, theta, yaw_rate, dt):
    if abs(yaw_rate) < 1e-6:
        # Straight-line motion
        del_x = vx * np.cos(theta) * dt
        del_y = vx * np.sin(theta) * dt
    else:
        # Arc motion
        del_x = (vx / yaw_rate) * (np.sin(theta + yaw_rate * dt) - np.sin(theta))
        del_y = -(vx / yaw_rate) * (np.cos(theta + yaw_rate * dt) - np.cos(theta))

    return np.array([del_x, del_y, 0.0])


class SkatingGaitController:
    def __init__(self, base_init_feet_pos, freq=1.0,
                 push_ratio=0.2, recovery_ratio=0.2, glide_ratio=0.6,
                 step_length=0.12, step_height=0.05,
                 style="handstand"):

        self.freq = freq
        self.push_ratio = push_ratio
        self.recovery_ratio = recovery_ratio
        self.glide_ratio = glide_ratio

        self.step_length = step_length
        self.step_height = step_height

        self.base_init = base_init_feet_pos.copy()
        self.base_feet_pos = base_init_feet_pos.copy()

        self.cycle_period = 1.0 / self.freq
        self.skating_style =  style
        if style=="handstand_sync" or style=="front_alt":
            # Only front legs are pushing alternatively and the back legs are frozen
            self.freeze_legs = ["RL_calf", "RR_calf"]
            self.alternating_legs = [["FL_calf"], ["FR_calf"]]
        elif style=="back_alt":
            # Only back legs are pushing alternatively and the front legs are frozen
            self.freeze_legs = ["FL_calf", "FR_calf"]
            self.alternating_legs = [["FL_calf"], ["FR_calf"]]
        elif style=="diagonal_sync":
            # Both diagonal legs are in sync and alternate between other diagonal legs
            self.freeze_legs = []
            self.alternating_legs = [["FL_calf", "RR_calf"], ["FR_calf", "RL_calf"]]
        else:
            # Both side legs are in sync and alternate between left and right
            self.freeze_legs = []
            self.alternating_legs = [["FL_calf", "RL_calf"], ["FR_calf", "RR_calf"]]




    def set_base_init_feet_pos(self, vx=1.0, yaw=0, dt=0.002, yaw_rate=0.0):
        """Shift reference foot positions forward as the body moves."""
        for leg in self.base_feet_pos.keys():
            self.base_feet_pos[leg] += delta_vector(vx=vx, theta=yaw, dt=dt, yaw_rate=yaw_rate)

    def _compute_leg_phase(self, leg_name, t):
        """
        Compute which phase the leg is in for alternating back-leg gait.
        """
        if leg_name in self.freeze_legs:
            return 0.0, False

        period = 2.0 / self.freq
        time_in_period = t % period
        cycle_time = 1.0 / self.freq

        if leg_name in self.alternating_legs[0]:
            if time_in_period < cycle_time:
                phase = time_in_period / cycle_time
                should_execute = True
            else:
                phase = 1.0
                should_execute = False

        elif leg_name in self.alternating_legs[1]:
            if time_in_period >= cycle_time:
                phase = (time_in_period - cycle_time) / cycle_time
                should_execute = True
            else:
                phase = 1.0
                should_execute = False
        else:
            phase = 0.0
            should_execute = False

        return phase, should_execute

    def foot_target(self, leg_name, t):
        """
        Calculate target foot position based on alternating skating gait.
        """
        phase, should_execute = self._compute_leg_phase(leg_name, t)

        foot = self.base_feet_pos[leg_name].copy()


        if not should_execute:
            return foot

        # PUSH PHASE
        if phase < self.push_ratio:
            progress = phase / self.push_ratio
            foot[0] = self.base_feet_pos[leg_name][0] - self.step_length * progress
            foot[2] = self.base_feet_pos[leg_name][2]
            return foot

        # RECOVERY PHASE
        elif phase < self.push_ratio + self.recovery_ratio:
            phase_rel = phase - self.push_ratio
            progress = phase_rel / self.recovery_ratio
            angle = np.pi * progress

            foot[0] = (
                self.base_feet_pos[leg_name][0]
                - self.step_length * (1 - progress)
            )
            foot[2] = (
                self.base_feet_pos[leg_name][2]
                + self.step_height * np.sin(angle)
            )
            return foot

        # GLIDE PHASE
        else:
            return self.base_feet_pos[leg_name]


class WalkingGaitController:
    def __init__(self, base_init_feet_pos, freq=1.0, duty_ratio=0.75, step_length=0.1, step_height=0.05):
        self.freq = freq
        self.duty = duty_ratio
        self.step_length = step_length
        self.step_height = step_height
        self.base_init_feet_pos = base_init_feet_pos.copy()
        self.base_feet_pos = base_init_feet_pos.copy()

        # Phase offsets (radians)
        self.phase_offsets = {
            "FL_calf": 0.0,
            "FR_calf": np.pi,
            "RL_calf": np.pi,
            "RR_calf": 0.0,
        }

    def set_base_init_feet_pos(self, vx=1.0, yaw=0, dt=0.002, yaw_rate=0.0):
        """Shift reference foot positions forward as the body moves."""
        for leg in self.base_feet_pos.keys():
            self.base_feet_pos[leg] += delta_vector(vx=vx, theta=yaw, dt=dt, yaw_rate=yaw_rate)


    def foot_target(self, leg_name, t, mode="moving"):
        """Compute desired foot position for leg_name at time t"""
        phi = 2 * np.pi * self.freq * t + self.phase_offsets[leg_name]
        phase = (phi % (2*np.pi)) / (2*np.pi)
        foot = self.base_feet_pos[leg_name].copy()

        if mode=="moving":
            if phase < self.duty:
                # --- Stance phase (foot on ground, moves backward relative to body)
                progress = phase / self.duty
                foot[0] -= self.step_length * (progress - 0.5)
                foot[2] = self.base_feet_pos[leg_name][2]  # stay on ground
            else:
                # --- Swing phase (foot in air following semicircle)
                progress = (phase - self.duty) / (1 - self.duty)
                angle = np.pi * progress
                foot[0] += self.step_length * (progress - 0.5)
                foot[2] = self.base_feet_pos[leg_name][2] + self.step_height * np.sin(angle)
        else:
            foot =  self.base_feet_pos[leg_name]

        return foot


class PronkGaitController:
    """
    Pronk gait: all four legs move in sync (hop).
    Foot trajectories are defined relative to the COM/base.
    """
    def __init__(self, base_init_feet_pos, freq=1.0, step_length=0.1, step_height=0.08, base_height=0.2):
        self.freq = freq                  # hopping frequency (Hz)
        self.step_length = step_length    # forward distance per hop
        self.step_height = step_height    # foot clearance (relative to base)
        self.base_height = base_height    # nominal base height
        self.base_init_feet_pos = base_init_feet_pos.copy()
        self.base_feet_pos = base_init_feet_pos.copy()

    def set_base_init_feet_pos(self, vx=1.0, yaw=0, dt=0.002, yaw_rate=0.0):
        """Shift reference foot positions forward as the body moves."""
        for leg in self.base_feet_pos.keys():
            self.base_feet_pos[leg] += delta_vector(vx=vx, theta=yaw, dt=dt, yaw_rate=yaw_rate)

    def com_height(self, t):
        """Compute vertical COM offset for hop."""
        phi = 2 * np.pi * self.freq * t
        return self.base_height + self.step_height * np.sin(phi)  # base rises during hop

    def foot_target(self, leg_name, t):
        """
        Compute foot target relative to COM.
        Legs move slightly relative to base for clearance.
        """
        foot = self.base_feet_pos[leg_name].copy()

        # Foot z is relative to COM (small clearance)
        phi = 2 * np.pi * self.freq * t
        foot[2] = foot[2] + 0.02 * np.sin(phi)  # small foot swing

        # Forward motion along x
        foot[0] += self.step_length * (phi / (2*np.pi) - 0.5)

        return foot

class BackflipGaitController:
    def __init__(self, base_init_feet_pos, total_time=1.0, step_height=0.05, base_height=0.35, hop_amp=0.05):
        self.base_feet_pos = base_init_feet_pos.copy()
        self.base_init_feet_pos = base_init_feet_pos.copy()
        self.total_time = total_time
        self.step_height = step_height
        self.base_height = base_height
        self.hop_amp = hop_amp

    # -----------------------------
    # Dummy method to match interface
    # -----------------------------
    def set_base_init_feet_pos(self, vx=0.0, yaw=0.0, dt=0.002, yaw_rate=0.0):
        pass

    def com_height(self, t):
        phi = t / self.total_time * np.pi
        return self.base_height + self.hop_amp * np.sin(phi)

    def foot_target(self, leg_name, t):
        foot = self.base_init_feet_pos[leg_name].copy()
        phi = t / self.total_time * np.pi
        foot[2] = self.com_height(t) - 0.05 + 0.02*np.sin(phi)
        foot[0] += 0.02 * np.sin(phi)
        return foot

    def base_orientation(self, t):
        phi = t / self.total_time * 2*np.pi
        return euler_to_quat(0, phi, 0)




