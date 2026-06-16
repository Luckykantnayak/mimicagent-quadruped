from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_PIVOT_DRAG_CIRCLE_MotionGenerator(BaseMotionGenerator):
    """
    Pivot-drag circular motion skill.
    
    The robot performs a 360-degree circular motion by sequentially pivoting 
    around three different legs (RR → RL → FR) while the other three legs 
    drag along the ground in coordinated arcs.
    
    Phase structure:
    - [0.0, 0.33]: RR pivot, FL/FR/RL drag
    - [0.33, 0.67]: RL pivot, FL/FR/RR drag
    - [0.67, 1.0]: FR pivot, FL/RL/RR drag
    
    Each pivot phase rotates the base ~120° clockwise, totaling 360°.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.3  # Slow rotation for smooth pivot-drag motion
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters
        self.total_yaw = 2 * np.pi  # 360 degrees
        self.yaw_rate = self.total_yaw * self.freq  # Constant yaw rate
        
        # Pivot radius approximation (distance from base center to pivot leg)
        # Used to compute linear velocities that maintain circular motion
        self.pivot_radius = 0.3
        
        # Drag arc amplitude - how much dragging legs sweep in body frame
        self.drag_amplitude = 0.15

    def update_base_motion(self, phase, dt):
        """
        Update base pose with coordinated linear and angular velocities
        to create circular motion around the current pivot leg.
        
        The pivot leg is determined by phase:
        - [0.0, 0.33]: pivot around RR
        - [0.33, 0.67]: pivot around RL
        - [0.67, 1.0]: pivot around FR
        """
        
        # Constant clockwise yaw rate throughout
        yaw_rate = -self.yaw_rate  # Negative for clockwise
        
        # Determine current pivot leg and compute tangential velocity
        # The base needs to move tangentially to orbit around the pivot leg
        if phase < 0.33:
            # Pivot around RR (right-rear)
            # RR is at positive x, negative y in body frame
            # Tangential direction for clockwise rotation: forward and left
            vx = self.pivot_radius * abs(yaw_rate) * 0.5
            vy = self.pivot_radius * abs(yaw_rate) * 0.8
        elif phase < 0.67:
            # Pivot around RL (left-rear)
            # RL is at positive x, positive y in body frame
            # Tangential direction: forward and right
            vx = self.pivot_radius * abs(yaw_rate) * 0.5
            vy = -self.pivot_radius * abs(yaw_rate) * 0.8
        else:
            # Pivot around FR (right-front)
            # FR is at negative x, negative y in body frame
            # Tangential direction: backward and left
            vx = -self.pivot_radius * abs(yaw_rate) * 0.5
            vy = self.pivot_radius * abs(yaw_rate) * 0.8
        
        # Set velocities in world frame
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
        Compute foot position in body frame for given leg and phase.
        
        Logic:
        - If leg is current pivot: adjust body-frame position to compensate 
          for base rotation, keeping it world-fixed
        - If leg is dragging: sweep through arc in body frame that corresponds
          to dragging motion around the pivot leg
        """
        
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine pivot leg for current phase
        if phase < 0.33:
            pivot_leg = "RR"
            sub_phase = phase / 0.33
        elif phase < 0.67:
            pivot_leg = "RL"
            sub_phase = (phase - 0.33) / 0.34
        else:
            pivot_leg = "FR"
            sub_phase = (phase - 0.67) / 0.33
        
        # Check if current leg is the pivot
        is_pivot = leg_name.startswith(pivot_leg)
        
        if is_pivot:
            # Pivot leg: rotate in body frame opposite to base yaw to stay world-fixed
            # Accumulated yaw in current sub-phase
            sub_yaw = -sub_phase * (self.total_yaw / 3.0)  # ~120° per phase, clockwise
            
            # Rotate base foot position by negative of accumulated yaw
            cos_yaw = np.cos(sub_yaw)
            sin_yaw = np.sin(sub_yaw)
            
            x = base_pos[0] * cos_yaw - base_pos[1] * sin_yaw
            y = base_pos[0] * sin_yaw + base_pos[1] * cos_yaw
            z = base_pos[2]
            
            return np.array([x, y, z])
        
        else:
            # Dragging leg: sweep through arc in body frame
            # The arc follows the tangential motion direction around pivot
            
            # Compute arc offset based on sub-phase and pivot configuration
            arc_progress = sub_phase  # [0, 1] within current pivot phase
            
            # Sinusoidal arc sweep for smooth dragging motion
            arc_offset_x = self.drag_amplitude * np.sin(np.pi * arc_progress)
            arc_offset_y = self.drag_amplitude * (arc_progress - 0.5)
            
            # Adjust offset direction based on pivot leg and current leg geometry
            if pivot_leg == "RR":
                # Dragging around RR: legs sweep forward-left
                if leg_name.startswith("FL"):
                    offset_x = -arc_offset_x * 0.5
                    offset_y = arc_offset_y * 0.8
                elif leg_name.startswith("FR"):
                    offset_x = -arc_offset_x * 0.3
                    offset_y = arc_offset_y * 0.5
                else:  # RL
                    offset_x = -arc_offset_x * 0.4
                    offset_y = arc_offset_y * 1.0
                    
            elif pivot_leg == "RL":
                # Dragging around RL: legs sweep forward-right
                if leg_name.startswith("FL"):
                    offset_x = -arc_offset_x * 0.5
                    offset_y = -arc_offset_y * 0.8
                elif leg_name.startswith("FR"):
                    offset_x = -arc_offset_x * 0.3
                    offset_y = -arc_offset_y * 0.5
                else:  # RR
                    offset_x = -arc_offset_x * 0.4
                    offset_y = -arc_offset_y * 1.0
                    
            else:  # FR pivot
                # Dragging around FR: legs sweep backward-left
                if leg_name.startswith("FL"):
                    offset_x = arc_offset_x * 0.3
                    offset_y = arc_offset_y * 0.5
                elif leg_name.startswith("RL"):
                    offset_x = arc_offset_x * 0.4
                    offset_y = arc_offset_y * 1.0
                else:  # RR
                    offset_x = arc_offset_x * 0.5
                    offset_y = arc_offset_y * 0.8
            
            # Apply drag offset to base position
            foot = base_pos.copy()
            foot[0] += offset_x
            foot[1] += offset_y
            foot[2] = base_pos[2]  # Maintain ground contact (z constant)
            
            return foot