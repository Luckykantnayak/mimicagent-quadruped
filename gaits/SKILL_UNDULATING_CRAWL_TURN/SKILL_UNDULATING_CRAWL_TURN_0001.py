from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_UNDULATING_CRAWL_TURN_MotionGenerator(BaseMotionGenerator):
    """
    Undulating crawl with rightward turning through diagonal wave propagation.
    
    - Body wave propagates diagonally: RR → FR → FL → RL
    - Sequential leg lifting creates crawl gait with asymmetric timing
    - Right-side legs lead in phase, producing continuous rightward turn
    - Maintains at least three feet in contact at all times
    - Low body posture with smooth undulation
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slow crawl frequency for stability
        
        # Gait parameters
        self.swing_duration = 0.25  # Each leg swings for 1/4 of cycle
        self.step_length = 0.12  # Forward step distance
        self.step_height = 0.06  # Moderate clearance height
        self.lateral_bias = 0.03  # Lateral step bias for turn asymmetry
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase offsets for sequential diagonal wave: RR → FR → FL → RL
        # Each leg swings centered at these phases
        self.swing_centers = {}
        for leg in leg_names:
            if leg.startswith('RR'):
                self.swing_centers[leg] = 0.125  # RR swings [0.0, 0.25]
            elif leg.startswith('FR'):
                self.swing_centers[leg] = 0.375  # FR swings [0.25, 0.50]
            elif leg.startswith('FL'):
                self.swing_centers[leg] = 0.625  # FL swings [0.50, 0.75]
            elif leg.startswith('RL'):
                self.swing_centers[leg] = 0.875  # RL swings [0.75, 1.0]
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Base velocity parameters
        self.vx_base = 0.15  # Forward velocity
        self.yaw_rate_base = 0.4  # Continuous right turn rate
        
        # Undulation parameters
        self.undulation_amplitude_z = 0.02  # Vertical undulation amplitude
        self.roll_amplitude = 0.08  # Body roll during wave
        self.pitch_amplitude = 0.06  # Body pitch during wave

    def update_base_motion(self, phase, dt):
        """
        Update base pose with forward motion, continuous yaw, and undulation.
        
        The undulation wave propagates diagonally:
        - phase 0.0-0.25: wave at rear-right (RR)
        - phase 0.25-0.50: wave at front-right (FR)
        - phase 0.50-0.75: wave at front-left (FL)
        - phase 0.75-1.0: wave at rear-left (RL)
        """
        
        # Forward velocity (constant positive)
        vx = self.vx_base
        
        # Lateral velocity (oscillates to create undulation, net rightward drift)
        vy = 0.02 * np.sin(2 * np.pi * phase)
        
        # Vertical velocity (creates body undulation wave)
        # Wave propagates diagonally: peaks when each leg swings
        wave_phase = phase * 4.0  # Four waves per cycle
        vz = self.undulation_amplitude_z * np.cos(2 * np.pi * phase) * 2.0
        
        # Roll rate (body rocks side-to-side with wave)
        # Negative roll when right side dips, positive when left side dips
        if phase < 0.5:
            # Wave on right side: right side dips then rises
            roll_rate = -self.roll_amplitude * np.sin(4 * np.pi * phase)
        else:
            # Wave on left side: left side dips then rises
            roll_rate = self.roll_amplitude * np.sin(4 * np.pi * phase)
        
        # Pitch rate (body pitches as wave propagates fore-aft)
        # Positive pitch (nose up) when wave at rear, negative when at front
        pitch_rate = self.pitch_amplitude * np.cos(2 * np.pi * phase)
        
        # Yaw rate (continuous right turn)
        yaw_rate = self.yaw_rate_base
        
        # Set velocity commands
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
        Compute foot position in body frame for given leg and phase.
        
        Each leg has:
        - Swing phase: 0.25 duration centered at swing_centers[leg]
        - Stance phase: 0.75 duration, foot sweeps rearward
        
        Swing trajectory: smooth arc (up, forward+lateral, down)
        Stance trajectory: linear rearward sweep
        """
        
        swing_center = self.swing_centers[leg_name]
        swing_half = self.swing_duration / 2.0
        
        # Determine if in swing or stance
        # Compute phase relative to swing center (wrapped to [-0.5, 0.5])
        phase_rel = ((phase - swing_center + 0.5) % 1.0) - 0.5
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine lateral bias based on leg side
        lateral_offset = 0.0
        if leg_name.startswith('FR') or leg_name.startswith('RR'):
            lateral_offset = self.lateral_bias  # Right legs step rightward
        else:
            lateral_offset = -self.lateral_bias  # Left legs step leftward
        
        if abs(phase_rel) < swing_half:
            # SWING PHASE: arc trajectory
            # Map phase_rel [-swing_half, swing_half] to progress [0, 1]
            progress = (phase_rel + swing_half) / self.swing_duration
            
            # Forward motion: arc from rear to front
            foot[0] += self.step_length * (progress - 0.5)
            
            # Lateral motion: bias for turn asymmetry
            foot[1] += lateral_offset * progress
            
            # Vertical motion: smooth arc (sine for smooth lift/land)
            arc_angle = np.pi * progress
            foot[2] += self.step_height * np.sin(arc_angle)
            
        else:
            # STANCE PHASE: foot sweeps rearward in body frame
            # This occurs because body moves forward while foot is planted
            
            # Determine progress through stance phase
            if phase_rel > 0:
                # After swing: progress from 0 to stance_end
                stance_progress = (phase_rel - swing_half) / (1.0 - self.swing_duration)
            else:
                # Before swing: progress continuing from previous cycle
                stance_progress = (phase_rel + 0.5 + swing_half) / (1.0 - self.swing_duration)
            
            # Rearward sweep: from front of stride to rear
            foot[0] -= self.step_length * stance_progress
            
            # Maintain lateral offset from previous swing
            foot[1] += lateral_offset
        
        return foot