from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_ZIGZAG_WEAVE_MotionGenerator(BaseMotionGenerator):
    """
    Zigzag weaving skill with sharp lateral cuts alternating left and right.
    
    Phase structure:
      [0.0, 0.2]: right_cut_1
      [0.2, 0.3]: straight_1
      [0.3, 0.5]: left_cut
      [0.5, 0.6]: straight_2
      [0.6, 0.8]: right_cut_2 (increased amplitude)
      [0.8, 1.0]: final_straight_stabilization
    
    All four feet remain in contact throughout. Lateral stance width modulates
    during cuts: outside legs push outward, inside legs tuck inward.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions in BODY frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Forward velocity (constant throughout)
        self.vx_forward = 1.
        
        # Lateral velocity amplitudes
        self.vy_cut_1 = 0.3  # First right cut
        self.vy_cut_left = 0.3  # Left cut
        self.vy_cut_2 = 0.45  # Second right cut (1.5x amplitude)
        
        # Yaw rate amplitudes
        self.yaw_rate_cut_1 = 0.8
        self.yaw_rate_cut_left = 0.8
        self.yaw_rate_cut_2 = 1.2  # Increased for second cut
        
        # Lateral stance modulation parameters
        self.lateral_push_offset = 0.15  # 15% wider stance when pushing
        self.lateral_tuck_offset = 0.10  # 10% narrower when tucking

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on current phase.
        Implements zigzag pattern with smooth transitions.
        """
        vx = self.vx_forward
        vy = 0.0
        yaw_rate = 0.0
        
        # Phase [0.0, 0.2]: right_cut_1
        if phase < 0.2:
            local_phase = phase / 0.2
            # Smooth ramp up using cosine
            blend = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            vy = self.vy_cut_1 * blend
            yaw_rate = self.yaw_rate_cut_1 * blend
        
        # Phase [0.2, 0.3]: straight_1 (transition)
        elif phase < 0.3:
            local_phase = (phase - 0.2) / 0.1
            # Smooth decay
            blend = 0.5 * (1.0 + np.cos(np.pi * local_phase))
            vy = self.vy_cut_1 * blend
            yaw_rate = self.yaw_rate_cut_1 * blend
        
        # Phase [0.3, 0.5]: left_cut
        elif phase < 0.5:
            local_phase = (phase - 0.3) / 0.2
            # Smooth transition to negative (leftward)
            blend = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            vy = -self.vy_cut_left * blend
            yaw_rate = -self.yaw_rate_cut_left * blend
        
        # Phase [0.5, 0.6]: straight_2 (transition)
        elif phase < 0.6:
            local_phase = (phase - 0.5) / 0.1
            # Decay from left cut
            blend = 0.5 * (1.0 + np.cos(np.pi * local_phase))
            vy = -self.vy_cut_left * blend
            yaw_rate = -self.yaw_rate_cut_left * blend
        
        # Phase [0.6, 0.8]: right_cut_2 (increased amplitude)
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            # Smooth ramp to higher amplitude
            blend = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            vy = self.vy_cut_2 * blend
            yaw_rate = self.yaw_rate_cut_2 * blend
        
        # Phase [0.8, 1.0]: final_straight_stabilization
        else:
            local_phase = (phase - 0.8) / 0.2
            # Smooth decay to zero
            blend = 0.5 * (1.0 + np.cos(np.pi * local_phase))
            vy = self.vy_cut_2 * blend
            yaw_rate = self.yaw_rate_cut_2 * blend
        
        # Set velocity commands in WORLD frame
        self.vel_world = np.array([vx, vy, 0.0])
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
        Compute foot position in BODY frame with lateral stance modulation.
        
        During right cuts: FL and RL push outward (left), FR and RR tuck inward
        During left cuts: FR and RR push outward (right), FL and RL tuck inward
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a left-side or right-side leg
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        lateral_offset = 0.0
        
        # Phase [0.0, 0.2]: right_cut_1
        if phase < 0.2:
            local_phase = phase / 0.2
            blend = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            if is_left:
                # Left legs push outward (increase +y in body frame)
                lateral_offset = self.lateral_push_offset * blend
            elif is_right:
                # Right legs tuck inward (decrease -y magnitude toward centerline)
                lateral_offset = -self.lateral_tuck_offset * blend
        
        # Phase [0.2, 0.3]: straight_1
        elif phase < 0.3:
            local_phase = (phase - 0.2) / 0.1
            blend = 0.5 * (1.0 + np.cos(np.pi * local_phase))
            if is_left:
                lateral_offset = self.lateral_push_offset * blend
            elif is_right:
                lateral_offset = -self.lateral_tuck_offset * blend
        
        # Phase [0.3, 0.5]: left_cut
        elif phase < 0.5:
            local_phase = (phase - 0.3) / 0.2
            blend = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            if is_right:
                # Right legs push outward (decrease -y in body frame, moving away from center)
                lateral_offset = -self.lateral_push_offset * blend
            elif is_left:
                # Left legs tuck inward (decrease +y toward centerline)
                lateral_offset = self.lateral_tuck_offset * blend
        
        # Phase [0.5, 0.6]: straight_2
        elif phase < 0.6:
            local_phase = (phase - 0.5) / 0.1
            blend = 0.5 * (1.0 + np.cos(np.pi * local_phase))
            if is_right:
                lateral_offset = -self.lateral_push_offset * blend
            elif is_left:
                lateral_offset = self.lateral_tuck_offset * blend
        
        # Phase [0.6, 0.8]: right_cut_2 (increased amplitude)
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            blend = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            if is_left:
                # Left legs push outward with increased amplitude
                lateral_offset = self.lateral_push_offset * 1.5 * blend
            elif is_right:
                # Right legs tuck inward with increased depth
                lateral_offset = -self.lateral_tuck_offset * 1.5 * blend
        
        # Phase [0.8, 1.0]: final_straight_stabilization
        else:
            local_phase = (phase - 0.8) / 0.2
            blend = 0.5 * (1.0 + np.cos(np.pi * local_phase))
            if is_left:
                lateral_offset = self.lateral_push_offset * 1.5 * blend
            elif is_right:
                lateral_offset = -self.lateral_tuck_offset * 1.5 * blend
        
        # Apply lateral offset (y-axis in body frame)
        foot[1] += lateral_offset
        
        # Slight rearward shift during cuts to simulate push
        # Compute combined cut intensity for x-axis adjustment
        cut_intensity = 0.0
        if phase < 0.2:
            cut_intensity = 0.5 * (1.0 - np.cos(np.pi * (phase / 0.2)))
        elif phase >= 0.3 and phase < 0.5:
            cut_intensity = 0.5 * (1.0 - np.cos(np.pi * ((phase - 0.3) / 0.2)))
        elif phase >= 0.6 and phase < 0.8:
            cut_intensity = 0.5 * (1.0 - np.cos(np.pi * ((phase - 0.6) / 0.2))) * 1.3
        
        # Push legs move slightly rearward during cuts
        if is_left and (phase < 0.3 or (phase >= 0.6 and phase < 0.8)):
            foot[0] -= 0.03 * cut_intensity
        elif is_right and (phase >= 0.3 and phase < 0.6):
            foot[0] -= 0.03 * cut_intensity
        
        return foot