from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_REVERSE_HELIX_SINK_MotionGenerator(BaseMotionGenerator):
    """
    Reverse helix sink motion: robot moves backward while rotating counter-clockwise
    and gradually lowering its base height, tracing a descending helical path.
    
    - Base moves backward (negative vx) with sustained velocity
    - Base rotates counter-clockwise (positive yaw rate) completing 360° per cycle
    - Base descends (negative vz) progressively to minimum height
    - All four feet maintain ground contact throughout
    - Legs compress and spread to accommodate the multi-axis motion
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for controlled helical motion
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters
        # Backward velocity profile
        self.vx_phase1 = -0.3  # Initial backward velocity
        self.vx_phase2 = -0.5  # Peak backward velocity
        self.vx_phase3 = -0.4  # Sustained backward velocity
        self.vx_phase4 = -0.35 # Tapering backward velocity
        
        # Yaw rotation parameters (360 degrees per cycle)
        self.yaw_rate_phase1 = 2.0  # Initial counter-clockwise rotation
        self.yaw_rate_phase2 = 2.5  # Peak rotation rate
        self.yaw_rate_phase3 = 2.3  # Sustained rotation
        self.yaw_rate_phase4 = 2.0  # Completion rotation
        
        # Vertical descent parameters
        self.vz_phase1 = -0.15  # Begin descent
        self.vz_phase2 = -0.35  # Maximum descent rate
        self.vz_phase3 = -0.08  # Approaching minimum height
        self.vz_phase4 = -0.02  # Maintain minimum height
        
        # Leg compression parameters
        self.compression_max = 0.12  # Maximum vertical compression (z increase in body frame)
        self.stance_width_expansion = 0.08  # Lateral spreading for stability
        
        # Leg-specific rotation compensation
        # As body rotates, feet appear to rotate in body frame
        self.rotation_compensation_gain = 0.15

    def update_base_motion(self, phase, dt):
        """
        Update base with backward translation, counter-clockwise yaw, and vertical descent.
        Phase-dependent velocity profiles create the helical trajectory.
        """
        # Determine phase-based velocities
        if phase < 0.25:
            # Initiation phase
            sub_phase = phase / 0.25
            vx = self.vx_phase1
            yaw_rate = self.yaw_rate_phase1
            vz = self.vz_phase1
        elif phase < 0.5:
            # Aggressive descent phase
            sub_phase = (phase - 0.25) / 0.25
            vx = self.vx_phase1 + (self.vx_phase2 - self.vx_phase1) * sub_phase
            yaw_rate = self.yaw_rate_phase1 + (self.yaw_rate_phase2 - self.yaw_rate_phase1) * sub_phase
            vz = self.vz_phase1 + (self.vz_phase2 - self.vz_phase1) * sub_phase
        elif phase < 0.75:
            # Deep rotation phase
            sub_phase = (phase - 0.5) / 0.25
            vx = self.vx_phase2 + (self.vx_phase3 - self.vx_phase2) * sub_phase
            yaw_rate = self.yaw_rate_phase2 + (self.yaw_rate_phase3 - self.yaw_rate_phase2) * sub_phase
            vz = self.vz_phase2 + (self.vz_phase3 - self.vz_phase2) * sub_phase
        else:
            # Completion phase
            sub_phase = (phase - 0.75) / 0.25
            vx = self.vx_phase3 + (self.vx_phase4 - self.vx_phase3) * sub_phase
            yaw_rate = self.yaw_rate_phase3 + (self.yaw_rate_phase4 - self.yaw_rate_phase3) * sub_phase
            vz = self.vz_phase3 + (self.vz_phase4 - self.vz_phase3) * sub_phase
        
        # Set velocities in world frame
        self.vel_world = np.array([vx, 0.0, vz])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
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
        Compute foot position in body frame with compression and rotational adjustment.
        All feet remain in contact; legs compress and reposition to accommodate
        the helical motion (backward translation + yaw rotation + height descent).
        """
        base_foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compression profile: gradual increase through phases, stabilizing at minimum
        if phase < 0.25:
            # Initiation: begin compression
            compression_factor = 0.3 * (phase / 0.25)
        elif phase < 0.5:
            # Aggressive descent: peak compression
            sub_phase = (phase - 0.25) / 0.25
            compression_factor = 0.3 + 0.5 * sub_phase
        elif phase < 0.75:
            # Deep rotation: maintain near-maximum compression
            compression_factor = 0.8 + 0.2 * ((phase - 0.5) / 0.25)
        else:
            # Completion: sustain maximum compression
            compression_factor = 1.0
        
        # Apply vertical compression (z increases toward base in body frame)
        z_offset = self.compression_max * compression_factor
        
        # Stance width expansion for stability (lateral spreading)
        lateral_expansion = self.stance_width_expansion * compression_factor
        
        # Adjust x and y based on leg position and rotation phase
        # As body rotates counter-clockwise, foot positions in body frame appear to rotate
        total_rotation = 2 * np.pi * phase  # Accumulated rotation angle
        
        # Rotational adjustment: feet shift in body frame as base rotates
        # This maintains world-frame contact while body rotates
        cos_rot = np.cos(total_rotation * self.rotation_compensation_gain)
        sin_rot = np.sin(total_rotation * self.rotation_compensation_gain)
        
        # Determine leg-specific adjustments
        if leg_name.startswith('FL'):
            # Front-left: expand left and forward
            x_adjust = 0.02 * compression_factor
            y_adjust = lateral_expansion
        elif leg_name.startswith('FR'):
            # Front-right: expand right and forward
            x_adjust = 0.02 * compression_factor
            y_adjust = -lateral_expansion
        elif leg_name.startswith('RL'):
            # Rear-left: expand left and rearward
            x_adjust = -0.02 * compression_factor
            y_adjust = lateral_expansion
        elif leg_name.startswith('RR'):
            # Rear-right: expand right and rearward
            x_adjust = -0.02 * compression_factor
            y_adjust = -lateral_expansion
        else:
            x_adjust = 0.0
            y_adjust = 0.0
        
        # Apply adjustments
        foot_pos = base_foot.copy()
        foot_pos[0] += x_adjust
        foot_pos[1] += y_adjust
        foot_pos[2] += z_offset
        
        # Apply small rotational compensation to maintain contact during yaw
        # This simulates the foot staying planted while body rotates around it
        dx = foot_pos[0] - base_foot[0]
        dy = foot_pos[1] - base_foot[1]
        
        # Rotate adjustment vector slightly to compensate for body rotation
        dx_rot = dx * cos_rot - dy * sin_rot
        dy_rot = dx * sin_rot + dy * cos_rot
        
        foot_pos[0] = base_foot[0] + dx_rot
        foot_pos[1] = base_foot[1] + dy_rot
        
        return foot_pos