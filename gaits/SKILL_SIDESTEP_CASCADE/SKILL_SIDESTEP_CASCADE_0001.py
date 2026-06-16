from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_SIDESTEP_CASCADE_MotionGenerator(BaseMotionGenerator):
    """
    Lateral sidestep cascade gait with sequential leg swings from front to rear.
    
    - Each leg steps laterally rightward in sequence: FL → FR → RL → RR
    - Creates a visible wave propagating from front to rear
    - Base drifts rightward through velocity integration during active swing phases
    - All legs grounded during reset/stabilization phase [0.8, 1.0]
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Lateral step parameters
        self.lateral_step_distance = 0.12  # Rightward displacement per leg step (body frame +y)
        self.step_height = 0.06  # Vertical clearance during swing
        
        # Base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Swing phase timing for each leg (non-overlapping sequential cascade)
        # FL: [0.0, 0.2], FR: [0.2, 0.4], RL: [0.4, 0.6], RR: [0.6, 0.8]
        self.swing_phases = {
            leg_names[0]: (0.0, 0.2),  # FL
            leg_names[1]: (0.2, 0.4),  # FR
            leg_names[2]: (0.4, 0.6),  # RL
            leg_names[3]: (0.6, 0.8),  # RR
        }
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Rightward velocity during active cascade phases
        self.lateral_velocity = 0.25  # m/s rightward (world frame +y during cascade)
        self.forward_drift = 0.02  # Small forward velocity for balance stability
        
    def update_base_motion(self, phase, dt):
        """
        Update base using rightward lateral velocity during cascade phases [0.0, 0.8],
        settling to zero during reset phase [0.8, 1.0].
        """
        if phase < 0.8:
            # Active cascade: rightward drift synchronized with leg swings
            vx = self.forward_drift
            vy = self.lateral_velocity
        else:
            # Reset/stabilization: zero velocity, all legs grounded
            vx = 0.0
            vy = 0.05  # Small residual for smooth cycle transition
        
        self.vel_world = np.array([vx, vy, 0.0])
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
    
    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on swing/stance state.
        
        During swing phase:
          - Parabolic arc lifting vertically and sweeping laterally rightward
        During stance phase:
          - Foot remains fixed in world frame, drifts leftward in body frame
            as base moves rightward
        """
        foot_base = self.base_feet_pos_body[leg_name].copy()
        
        swing_start, swing_end = self.swing_phases[leg_name]
        swing_duration = swing_end - swing_start
        
        if swing_start <= phase < swing_end:
            # SWING PHASE: lift and step laterally right
            swing_progress = (phase - swing_start) / swing_duration
            
            # Lateral (y) displacement: sweep rightward
            foot_base[1] += self.lateral_step_distance * (swing_progress - 0.5)
            
            # Vertical (z) displacement: parabolic arc
            foot_base[2] += self.step_height * np.sin(np.pi * swing_progress)
            
        else:
            # STANCE PHASE: foot grounded in world, drifts left in body frame
            # Compute how much phase has elapsed since this leg's swing ended
            if phase >= swing_end:
                stance_elapsed = phase - swing_end
            else:
                # Before this leg's swing: stance from previous cycle
                stance_elapsed = (1.0 - swing_end) + phase
            
            # Approximate leftward drift in body frame as base moves right
            # Integrate lateral velocity relative to body
            drift_distance = self.lateral_velocity * stance_elapsed / self.freq
            foot_base[1] -= drift_distance
            
            # After one full cycle, foot returns to base position for next swing
            # Reset logic embedded in phase wrapping
        
        return foot_base