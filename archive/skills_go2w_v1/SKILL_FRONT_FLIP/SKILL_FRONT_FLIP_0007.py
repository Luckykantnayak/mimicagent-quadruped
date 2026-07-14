from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_FRONT_FLIP_MotionGenerator(BaseMotionGenerator):
    """
    Front flip motion: Complete forward pitch rotation (360°) with airborne phase.
    
    Phase breakdown:
      0.00-0.15: Crouch preparation (all feet grounded, body lowers)
      0.15-0.30: Launch and rotation initiation (explosive upward + pitch rate)
      0.30-0.75: Airborne rotation (all feet off ground, complete ~360° pitch)
      0.75-0.90: Landing preparation (reduce pitch rate, extend legs)
      0.90-1.00: Landing and stabilization (re-establish contact, zero velocities)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Store base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.crouch_depth = 0.10          # Vertical compression during crouch
        self.launch_vz = 3.5              # Peak upward velocity during launch
        self.launch_pitch_rate = 12.0     # Forward pitch angular velocity (rad/s)
        self.flight_pitch_rate = 10.0     # Sustained pitch rate during airborne phase
        self.tuck_retraction = 0.15       # How much feet retract toward body COM during flight
        self.landing_extension = 0.05     # Extra leg extension for landing prep
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Phase 0.00-0.15: Crouch preparation
        if phase < 0.15:
            vx = 0.0
            vz = 0.0
            pitch_rate = 0.0
        
        # Phase 0.15-0.30: Launch and rotation initiation
        elif phase < 0.30:
            local_phase = (phase - 0.15) / 0.15
            # Ramp up upward velocity and pitch rate
            vx = 0.3 * np.sin(np.pi * local_phase)
            vz = self.launch_vz * np.sin(np.pi * local_phase)
            pitch_rate = self.launch_pitch_rate * (0.5 + 0.5 * local_phase)
        
        # Phase 0.30-0.75: Airborne rotation
        elif phase < 0.75:
            local_phase = (phase - 0.30) / 0.45
            # Ballistic trajectory: vz decreases due to gravity simulation
            # Peak at start, negative at end
            vx = 0.1
            vz = self.launch_vz * 0.8 * (1.0 - 2.0 * local_phase)
            pitch_rate = self.flight_pitch_rate
        
        # Phase 0.75-0.90: Landing preparation
        elif phase < 0.90:
            local_phase = (phase - 0.75) / 0.15
            # Descending, reduce pitch rate to zero
            vx = 0.0
            vz = -1.5 * (1.0 - local_phase)
            pitch_rate = self.flight_pitch_rate * (1.0 - local_phase)
        
        # Phase 0.90-1.00: Landing and stabilization
        else:
            local_phase = (phase - 0.90) / 0.10
            # Rapidly zero out all velocities
            vx = 0.0
            vz = -0.5 * (1.0 - local_phase)
            pitch_rate = 0.0
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on phase.
        All legs move synchronously through crouch, tuck, and landing.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front = leg_name.startswith('F')
        
        # Phase 0.00-0.15: Crouch preparation
        if phase < 0.15:
            local_phase = phase / 0.15
            # Lower body by extending feet downward in body frame
            foot[2] += self.crouch_depth * local_phase
            if is_front:
                foot[0] += 0.02 * local_phase  # Slightly forward
            else:
                foot[0] -= 0.02 * local_phase  # Slightly backward
        
        # Phase 0.15-0.30: Launch (feet push off, then break contact)
        elif phase < 0.30:
            local_phase = (phase - 0.15) / 0.15
            # Feet extend maximally at start, then lift off
            foot[2] += self.crouch_depth * (1.0 - local_phase)
            if is_front:
                foot[0] += 0.02 * (1.0 - local_phase)
            else:
                foot[0] -= 0.02 * (1.0 - local_phase)
        
        # Phase 0.30-0.75: Airborne rotation (tuck feet toward body)
        elif phase < 0.75:
            local_phase = (phase - 0.30) / 0.45
            # Retract feet toward body center to reduce rotational inertia
            tuck_factor = np.sin(np.pi * min(local_phase * 1.5, 1.0))
            foot[0] *= (1.0 - self.tuck_retraction * tuck_factor)
            foot[1] *= (1.0 - self.tuck_retraction * tuck_factor)
            foot[2] -= self.tuck_retraction * tuck_factor
        
        # Phase 0.75-0.90: Landing preparation (extend legs downward)
        elif phase < 0.90:
            local_phase = (phase - 0.75) / 0.15
            # Transition from tucked to extended landing position
            tuck_factor = 1.0 - local_phase
            foot[0] *= (1.0 - self.tuck_retraction * tuck_factor)
            foot[1] *= (1.0 - self.tuck_retraction * tuck_factor)
            foot[2] -= self.tuck_retraction * tuck_factor
            # Add landing extension
            foot[2] += self.landing_extension * local_phase
        
        # Phase 0.90-1.00: Landing and stabilization
        else:
            local_phase = (phase - 0.90) / 0.10
            # Compress slightly on impact, then return to neutral
            compression = 0.05 * np.sin(np.pi * local_phase)
            foot[2] += self.landing_extension * (1.0 - local_phase) + compression
        
        return foot