from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_REVERBERATING_YAW_PULSE_MotionGenerator(BaseMotionGenerator):
    """
    Reverberating yaw pulse skill: in-place rotational motion with damped oscillatory yaw pulses.
    
    Base motion:
    - Zero linear velocity (no translation)
    - Alternating yaw rate pulses that decay in amplitude
    - Net 45° counterclockwise rotation per cycle
    
    Leg motion:
    - All four feet remain in ground contact (stance only)
    - Radial extension/retraction synchronized with yaw pulses to assist rotation
    - Front and rear pairs move symmetrically to maintain balance
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0

        # Base foot positions (nominal stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Radial extension parameters (distance modulation from nominal position)
        self.extension_amplitudes = {
            "strong_pulse": 0.1,  # initial outward burst
            "moderate_reversal": -0.045,  # snap-back compression
            "diminishing_pulse": -0.028,  # continued inward recoil
            "small_adjustment": -0.012,  # damping correction
            "final_settle": -0.003  # residual compression
        }

        # Yaw rate parameters (rad/s) tuned to achieve target rotations
        # Target rotations: 60°, -40°, 25°, -15°, 5° over 0.2-phase intervals
        # At freq=1.0, each 0.2 phase = 0.2s
        self.yaw_rates = {
            "strong_ccw": np.deg2rad(60) / 0.2,      # ~5.24 rad/s
            "moderate_cw": -np.deg2rad(40) / 0.2,    # ~-3.49 rad/s
            "diminishing_ccw": np.deg2rad(25) / 0.2, # ~2.18 rad/s
            "small_cw": -np.deg2rad(15) / 0.2,       # ~-1.31 rad/s
            "final_ccw": np.deg2rad(5) / 0.2         # ~0.44 rad/s
        }
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
    def update_base_motion(self, phase, dt):
        """
        Update base pose using phase-dependent yaw rate commands.
        Linear velocities remain zero (in-place rotation).
        Smooth transitions between phases using cosine blending.
        """
        # Determine yaw rate based on phase with smooth blending
        blend_width = 0.05  # Blend over 5% of phase at boundaries
        
        if phase < 0.2:
            # Strong CCW pulse
            yaw_rate = self.yaw_rates["strong_ccw"]
            # Ramp down near end
            if phase > 0.2 - blend_width:
                blend = (0.2 - phase) / blend_width
                yaw_rate *= 0.5 * (1 + np.cos(np.pi * (1 - blend)))
                
        elif phase < 0.4:
            # Moderate CW reversal
            yaw_rate = self.yaw_rates["moderate_cw"]
            # Ramp up at start
            if phase < 0.2 + blend_width:
                blend = (phase - 0.2) / blend_width
                yaw_rate *= 0.5 * (1 - np.cos(np.pi * blend))
            # Ramp down at end
            elif phase > 0.4 - blend_width:
                blend = (0.4 - phase) / blend_width
                yaw_rate *= 0.5 * (1 + np.cos(np.pi * (1 - blend)))
                
        elif phase < 0.6:
            # Diminishing CCW correction
            yaw_rate = self.yaw_rates["diminishing_ccw"]
            # Ramp up at start
            if phase < 0.4 + blend_width:
                blend = (phase - 0.4) / blend_width
                yaw_rate *= 0.5 * (1 - np.cos(np.pi * blend))
            # Ramp down at end
            elif phase > 0.6 - blend_width:
                blend = (0.6 - phase) / blend_width
                yaw_rate *= 0.5 * (1 + np.cos(np.pi * (1 - blend)))
                
        elif phase < 0.8:
            # Small CW adjustment
            yaw_rate = self.yaw_rates["small_cw"]
            # Ramp up at start
            if phase < 0.6 + blend_width:
                blend = (phase - 0.6) / blend_width
                yaw_rate *= 0.5 * (1 - np.cos(np.pi * blend))
            # Ramp down at end
            elif phase > 0.8 - blend_width:
                blend = (0.8 - phase) / blend_width
                yaw_rate *= 0.5 * (1 + np.cos(np.pi * (1 - blend)))
                
        else:
            # Final damping settle
            yaw_rate = self.yaw_rates["final_ccw"]
            # Ramp up at start
            if phase < 0.8 + blend_width:
                blend = (phase - 0.8) / blend_width
                yaw_rate *= 0.5 * (1 - np.cos(np.pi * blend))
            # Ramp down to zero at end
            if phase > 0.95:
                blend = (1.0 - phase) / 0.05
                yaw_rate *= blend
        
        # Set velocities (zero translation, only yaw rotation)
        self.vel_world = np.array([0.0, 0.0, 0.0])
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
        Compute foot position in body frame with radial extension/retraction
        synchronized to yaw pulses. All feet remain grounded (z constant).
        """
        nominal_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine radial extension based on phase
        if phase < 0.2:
            # Strong pulse: maximum extension
            extension = self.extension_amplitudes["strong_pulse"]
            progress = phase / 0.2
            # Smooth extension from 0 to max
            extension_factor = 0.5 * (1 - np.cos(np.pi * progress))
            
        elif phase < 0.4:
            # Moderate reversal: retract from max to moderate
            extension = self.extension_amplitudes["moderate_reversal"]
            progress = (phase - 0.2) / 0.2
            # Blend from strong to moderate
            prev_ext = self.extension_amplitudes["strong_pulse"]
            extension = prev_ext + (extension - prev_ext) * progress
            extension_factor = 1.0
            
        elif phase < 0.6:
            # Diminishing pulse: extend to moderate level
            extension = self.extension_amplitudes["diminishing_pulse"]
            progress = (phase - 0.4) / 0.2
            # Blend from moderate to diminishing
            prev_ext = self.extension_amplitudes["moderate_reversal"]
            extension = prev_ext + (extension - prev_ext) * progress
            extension_factor = 1.0
            
        elif phase < 0.8:
            # Small adjustment: retract further
            extension = self.extension_amplitudes["small_adjustment"]
            progress = (phase - 0.6) / 0.2
            # Blend from diminishing to small
            prev_ext = self.extension_amplitudes["diminishing_pulse"]
            extension = prev_ext + (extension - prev_ext) * progress
            extension_factor = 1.0
            
        else:
            # Final settle: return to near-nominal
            extension = self.extension_amplitudes["final_settle"]
            progress = (phase - 0.8) / 0.2
            # Smooth return to nominal
            prev_ext = self.extension_amplitudes["small_adjustment"]
            extension = prev_ext + (extension - prev_ext) * progress
            extension_factor = 1.0 - progress * 0.5
        
        # Calculate radial direction from body center to nominal foot position
        radial_xy = nominal_pos[:2].copy()
        radial_distance = np.linalg.norm(radial_xy)
        
        if radial_distance > 1e-6:
            radial_direction = radial_xy / radial_distance
        else:
            radial_direction = np.array([1.0, 0.0])
        
        # Apply radial extension (outward for front legs, outward for rear legs)
        # Front legs (FL, FR): extend forward and outward
        # Rear legs (RL, RR): extend rearward and outward
        foot_pos = nominal_pos.copy()
        
        if leg_name.startswith('F'):
            # Front legs: extend outward and slightly forward
            foot_pos[0] += extension * extension_factor * radial_direction[0]
            foot_pos[1] += extension * extension_factor * radial_direction[1]
        else:
            # Rear legs: extend outward and slightly rearward
            foot_pos[0] += extension * extension_factor * radial_direction[0]
            foot_pos[1] += extension * extension_factor * radial_direction[1]
        
        # Maintain ground contact (z unchanged)
        foot_pos[2] = nominal_pos[2]
        
        return foot_pos