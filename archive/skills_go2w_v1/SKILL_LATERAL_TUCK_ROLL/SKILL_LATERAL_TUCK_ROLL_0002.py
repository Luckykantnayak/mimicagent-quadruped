from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_LATERAL_TUCK_ROLL_MotionGenerator(BaseMotionGenerator):
    """
    Lateral tuck roll motion generator.
    
    Executes a continuous sideways rolling motion where the robot rotates 
    about its longitudinal axis while translating laterally to the left.
    All four legs remain tucked close to the body throughout the entire 
    cycle to minimize rotational inertia.
    
    Phase structure:
    - [0.0, 0.25]: Right side lift initiation
    - [0.25, 0.5]: Inverted maximum tuck (all feet off ground)
    - [0.5, 0.75]: Left side lift continuation
    - [0.75, 1.0]: Upright return preparation
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.1  # Full roll cycle frequency
        
        # Base foot positions (extended stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Tuck parameters
        self.tuck_offset_x = 0.0  # Keep x near nominal
        self.tuck_offset_y_scale = 0.3  # Pull toward centerline (reduce lateral distance)
        self.tuck_offset_z = 0.15  # Lift feet upward toward belly
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Base motion parameters
        self.lateral_velocity = -0.5  # Leftward translation (negative y)
        self.roll_rate = -2 * np.pi * self.freq  # -360 deg/cycle for leftward roll
        self.vertical_velocity_amplitude = 0.15  # Modulate z velocity slightly

    def update_base_motion(self, phase, dt):
        """
        Update base using constant roll rate and lateral velocity.
        Roll rate is negative (leftward roll about longitudinal axis).
        Lateral velocity is constant leftward.
        Vertical velocity modulates slightly to assist roll initiation/completion.
        """
        
        # Compute vertical velocity modulation based on phase
        if phase < 0.25:
            # Right side lift: slight upward velocity
            vz = self.vertical_velocity_amplitude * np.sin(np.pi * phase / 0.25)
        elif phase < 0.5:
            # Inverted phase: neutral vertical velocity
            vz = 0.0
        elif phase < 0.75:
            # Left side lift: slight downward velocity
            vz = -self.vertical_velocity_amplitude * np.sin(np.pi * (phase - 0.5) / 0.25)
        else:
            # Upright return: neutral
            vz = 0.0
        
        # Set velocity commands (world frame)
        self.vel_world = np.array([0.0, self.lateral_velocity, vz])
        self.omega_world = np.array([self.roll_rate, 0.0, 0.0])
        
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
        
        Leg motion logic:
        - FL, RL (left legs): Contact [0, 0.25], tucked [0.25, 0.75], extending [0.75, 1.0]
        - FR, RR (right legs): Tucked [0, 0.5], extending/contact [0.5, 1.0]
        """
        
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if left or right leg
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        if is_left:
            # FL, RL logic
            if phase < 0.25:
                # Stance with partial tuck (preparing for lift)
                tuck_progress = phase / 0.25
                foot = self._apply_tuck(base_pos, tuck_progress * 0.5, leg_name)
            elif phase < 0.75:
                # Fully tucked (swing phase through inverted)
                foot = self._apply_tuck(base_pos, 1.0, leg_name)
            else:
                # Extending back to stance
                extension_progress = (phase - 0.75) / 0.25
                foot = self._apply_tuck(base_pos, 1.0 - extension_progress, leg_name)
        
        elif is_right:
            # FR, RR logic
            if phase < 0.25:
                # Rapid tuck as right side lifts
                tuck_progress = min(1.0, phase / 0.15)  # Quick tuck
                foot = self._apply_tuck(base_pos, tuck_progress, leg_name)
            elif phase < 0.5:
                # Fully tucked through inverted
                foot = self._apply_tuck(base_pos, 1.0, leg_name)
            else:
                # Extending and re-establishing contact
                extension_progress = (phase - 0.5) / 0.5
                foot = self._apply_tuck(base_pos, 1.0 - extension_progress, leg_name)
        else:
            # Fallback (should not reach here)
            foot = base_pos
        
        return foot

    def _apply_tuck(self, base_pos, tuck_amount, leg_name):
        """
        Apply tuck transformation to foot position.
        
        tuck_amount: 0.0 = fully extended (base position)
                     1.0 = fully tucked (close to body)
        
        Tucking moves foot:
        - Toward body centerline in y (reduce lateral distance)
        - Upward in z (toward belly)
        - Keep x roughly constant
        """
        foot = base_pos.copy()
        
        # Determine if left or right leg for y-direction tucking
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Y-direction: move toward centerline
        # Left legs have positive y, right legs have negative y
        if is_left:
            # Reduce positive y
            foot[1] = base_pos[1] * (1.0 - self.tuck_offset_y_scale * tuck_amount)
        else:
            # Reduce magnitude of negative y (move toward zero)
            foot[1] = base_pos[1] * (1.0 - self.tuck_offset_y_scale * tuck_amount)
        
        # Z-direction: lift upward
        foot[2] = base_pos[2] + self.tuck_offset_z * tuck_amount
        
        # X-direction: keep near base position (minimal fore-aft movement)
        foot[0] = base_pos[0] + self.tuck_offset_x * tuck_amount
        
        return foot