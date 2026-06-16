from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_CRAB_DIAGONAL_SCUTTLE_MotionGenerator(BaseMotionGenerator):
    """
    Crab diagonal scuttle: Robot maintains sideways orientation (90° yaw) while 
    scuttling diagonally forward-right. Legs perform coordinated sweeping motions:
    front legs push backward, rear legs push forward in body frame, generating 
    diagonal thrust in world frame due to perpendicular body orientation.
    
    Motion phases:
    - [0.0, 0.3]: First scuttle stroke (all legs in contact, coordinated sweep)
    - [0.3, 0.5]: Rapid reset (legs reposition, diagonal pair support)
    - [0.5, 0.8]: Second amplified scuttle (increased sweep amplitude)
    - [0.8, 1.0]: Glide and stabilize (return to neutral, maintain momentum)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Scuttle stroke parameters
        self.first_stroke_length = 0.12
        self.second_stroke_amplification = 1.4
        self.sweep_height_offset = -0.02
        self.reset_lift_height = 0.06
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Perpendicular body orientation (90° yaw relative to travel direction)
        # Maintained throughout the skill
        self.sideways_yaw = np.pi / 2
        self.root_quat = euler_to_quat(0.0, 0.0, self.sideways_yaw)
        
        # Velocity parameters for diagonal scuttle
        # In body frame: +x (forward) and +y (rightward) combine to produce diagonal world motion
        self.first_scuttle_vx = 0.8
        self.first_scuttle_vy = 0.6
        self.second_scuttle_vx = 1.2
        self.second_scuttle_vy = 0.9
        self.glide_vx = 0.4
        self.glide_vy = 0.3

    def update_base_motion(self, phase, dt):
        """
        Update base motion based on current phase.
        Body maintains sideways orientation (yaw locked at 90°) throughout.
        Velocity commands in body frame produce diagonal world-frame motion.
        """
        
        # Always maintain sideways orientation - zero yaw rate
        yaw_rate = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        
        # Phase-dependent velocity commands (body frame)
        if phase < 0.3:
            # First scuttle stroke: moderate diagonal thrust
            vx = self.first_scuttle_vx
            vy = self.first_scuttle_vy
            vz = 0.0
            
        elif phase < 0.5:
            # Rapid reset: coasting on momentum with deceleration
            progress = (phase - 0.3) / 0.2
            vx = self.first_scuttle_vx * (1.0 - 0.4 * progress)
            vy = self.first_scuttle_vy * (1.0 - 0.4 * progress)
            vz = 0.02 * np.sin(np.pi * progress)
            
        elif phase < 0.8:
            # Second amplified scuttle: increased thrust
            progress = (phase - 0.5) / 0.3
            vx = self.first_scuttle_vx * 0.6 + (self.second_scuttle_vx - self.first_scuttle_vx * 0.6) * progress
            vy = self.first_scuttle_vy * 0.6 + (self.second_scuttle_vy - self.first_scuttle_vy * 0.6) * progress
            vz = -0.02 * progress if progress < 0.5 else 0.0
            
        else:
            # Glide and stabilize: smooth transition to cyclic steady state
            progress = (phase - 0.8) / 0.2
            vx = self.second_scuttle_vx * (1.0 - progress) + self.glide_vx * progress
            vy = self.second_scuttle_vy * (1.0 - progress) + self.glide_vy * progress
            vz = 0.0
        
        # Velocity commands are in body frame
        # With 90° yaw, body +x points to world +y, body +y points to world -x
        # This produces diagonal forward-right motion in world frame
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
        # Integrate pose (but override quat to maintain sideways orientation)
        self.root_pos, _ = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        # Lock yaw at 90° - critical constraint for crab motion
        self.root_quat = euler_to_quat(0.0, 0.0, self.sideways_yaw)

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on phase.
        
        Front legs (FL, FR): sweep rearward during scuttle (negative x direction)
        Rear legs (RL, RR): sweep forward during scuttle (positive x direction)
        
        This opposing motion creates the characteristic crab scuttle.
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        is_fl_or_rr = leg_name.startswith('FL') or leg_name.startswith('RR')
        
        if phase < 0.3:
            # First scuttle stroke: all legs in contact, sweep motion
            progress = phase / 0.3
            sweep_offset = self.first_stroke_length * (progress - 0.5)
            
            if is_front:
                # Front legs sweep rearward (negative x)
                foot[0] -= sweep_offset
            else:
                # Rear legs sweep forward (positive x)
                foot[0] += sweep_offset
            
            # Slight downward pressure during power stroke
            foot[2] += self.sweep_height_offset
            
        elif phase < 0.5:
            # Rapid reset: FL and RR swing, FR and RL maintain contact
            progress = (phase - 0.3) / 0.2
            
            if is_fl_or_rr:
                # FL and RR: rapid repositioning with arc trajectory
                if is_front:
                    # FL: move from rear to forward position
                    foot[0] -= self.first_stroke_length * (0.5 - progress)
                else:
                    # RR: move from forward to rear position
                    foot[0] += self.first_stroke_length * (0.5 - progress)
                
                # Arc lift during swing
                foot[2] += self.reset_lift_height * np.sin(np.pi * progress)
            else:
                # FR and RL: diagonal pair maintains ground contact
                # Hold position from end of first scuttle
                if is_front:
                    foot[0] -= self.first_stroke_length * 0.5
                else:
                    foot[0] += self.first_stroke_length * 0.5
                foot[2] += self.sweep_height_offset
                
        elif phase < 0.8:
            # Second amplified scuttle: all legs in contact, larger sweep
            progress = (phase - 0.5) / 0.3
            amplified_stroke = self.first_stroke_length * self.second_stroke_amplification
            sweep_offset = amplified_stroke * (progress - 0.5)
            
            if is_front:
                # Front legs sweep rearward with increased amplitude
                foot[0] -= sweep_offset
            else:
                # Rear legs sweep forward with increased amplitude
                foot[0] += sweep_offset
            
            foot[2] += self.sweep_height_offset
            
        else:
            # Glide and stabilize: smooth return to neutral stance
            progress = (phase - 0.8) / 0.2
            amplified_stroke = self.first_stroke_length * self.second_stroke_amplification
            
            if is_front:
                # Return from rear sweep position to neutral
                final_offset = -amplified_stroke * 0.5
                foot[0] += final_offset * (1.0 - progress)
            else:
                # Return from forward sweep position to neutral
                final_offset = amplified_stroke * 0.5
                foot[0] += final_offset * (1.0 - progress)
            
            # Smoothly return to nominal height
            foot[2] += self.sweep_height_offset * (1.0 - progress)
        
        return foot