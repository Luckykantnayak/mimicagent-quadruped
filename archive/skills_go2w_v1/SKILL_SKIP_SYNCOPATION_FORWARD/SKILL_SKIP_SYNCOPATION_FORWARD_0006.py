from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_SKIP_SYNCOPATION_FORWARD_MotionGenerator(BaseMotionGenerator):
    """
    Forward-moving skip with syncopated rhythm.
    
    Features:
    - Rear leg double-bounce (phase 0.0-0.15)
    - Long glide (phase 0.15-0.35)
    - Front leg brief contact and double-bounce (phase 0.35-0.55)
    - Extended asymmetric glide (phase 0.55-0.8)
    - Syncopated diagonal landing: left pair at 0.8, right pair at 0.9
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions in BODY frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.step_length = 0.12
        self.step_height = 0.10
        self.bounce_amplitude = 0.03  # Reduced from 0.06
        self.glide_extension = 0.06
        self.tuck_height = 0.06
        
        # Velocity parameters
        self.rear_bounce_vx = 0.7
        self.rear_bounce_vz = 0.8  # Increased from 0.6
        self.rear_bounce_pitch = -0.3
        
        self.glide_vx = 0.5
        self.glide_pitch = 0.4
        
        self.front_contact_vx = 0.3
        self.front_contact_vz = -0.1  # Reduced magnitude
        
        self.front_bounce_vx = 0.6
        self.front_bounce_vz = 0.7  # Increased from 0.5
        self.front_bounce_pitch = 0.5
        
        self.extended_glide_vx = 0.4
        self.extended_glide_pitch = -0.2
        
        self.landing_decel = 0.2
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        """
        
        # Phase 0.0-0.15: Rear double-bounce
        if phase < 0.15:
            vx = self.rear_bounce_vx
            vy = 0.0
            # Sinusoidal bounce pattern for double-bounce
            bounce_phase = phase / 0.15
            vz = self.rear_bounce_vz * np.sin(2 * np.pi * bounce_phase * 2)
            pitch_rate = self.rear_bounce_pitch
            roll_rate = 0.0
            yaw_rate = 0.0
            
        # Phase 0.15-0.35: Long glide forward
        elif phase < 0.35:
            vx = self.glide_vx
            vy = 0.0
            # Ballistic arc: initially upward, then gradually downward
            glide_progress = (phase - 0.15) / 0.2
            vz = 0.5 * np.cos(np.pi * glide_progress) - 0.1
            pitch_rate = self.glide_pitch
            roll_rate = 0.0
            yaw_rate = 0.0
            
        # Phase 0.35-0.4: Front brief contact
        elif phase < 0.4:
            vx = self.front_contact_vx
            vy = 0.0
            vz = self.front_contact_vz
            pitch_rate = 0.5
            roll_rate = 0.0
            yaw_rate = 0.0
            
        # Phase 0.4-0.55: Front double-bounce
        elif phase < 0.55:
            vx = self.front_bounce_vx
            vy = 0.0
            # Double-bounce pattern
            bounce_phase = (phase - 0.4) / 0.15
            vz = self.front_bounce_vz * np.sin(2 * np.pi * bounce_phase * 2)
            pitch_rate = self.front_bounce_pitch
            roll_rate = 0.0
            yaw_rate = 0.0
            
        # Phase 0.55-0.8: Extended asymmetric glide
        elif phase < 0.8:
            vx = self.extended_glide_vx
            vy = 0.0
            # Ballistic descent with smoother approach to landing
            glide_progress = (phase - 0.55) / 0.25
            vz = 0.4 * np.cos(np.pi * glide_progress) - 0.2
            pitch_rate = self.extended_glide_pitch
            roll_rate = 0.0
            yaw_rate = 0.0
            
        # Phase 0.8-0.9: Left diagonal pair lands
        elif phase < 0.9:
            landing_progress = (phase - 0.8) / 0.1
            vx = self.extended_glide_vx * (1 - 0.5 * landing_progress)
            vy = -0.04 * np.sin(np.pi * landing_progress)
            # Decelerate vertical descent smoothly
            vz = -0.2 * (1 - landing_progress)
            roll_rate = -0.3 * np.sin(np.pi * landing_progress)
            pitch_rate = 0.0
            yaw_rate = 0.08 * np.sin(np.pi * landing_progress)
            
        # Phase 0.9-1.0: Right diagonal pair completes landing
        else:
            landing_progress = (phase - 0.9) / 0.1
            vx = self.landing_decel * (1 - landing_progress)
            vy = 0.04 * np.sin(np.pi * landing_progress)
            # Near zero vertical velocity as landing completes
            vz = -0.1 * (1 - landing_progress)
            roll_rate = 0.3 * np.sin(np.pi * landing_progress)
            pitch_rate = 0.0
            yaw_rate = -0.08 * np.sin(np.pi * landing_progress)
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def smooth_step(self, x):
        """Smooth interpolation function (smoothstep)."""
        return x * x * (3.0 - 2.0 * x)

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in BODY frame for each leg based on phase.
        Feet remain at or near zero Z during stance to avoid ground penetration.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Front Left (FL)
        if leg_name.startswith('FL'):
            if phase < 0.15:
                # Swing: tuck upward and forward
                progress = self.smooth_step(phase / 0.15)
                foot[0] += 0.04 * progress
                foot[2] += self.tuck_height * np.sin(np.pi * progress)
            elif phase < 0.35:
                # Swing: extend forward for contact
                progress = self.smooth_step((phase - 0.15) / 0.2)
                foot[0] += 0.04 + self.glide_extension * progress
                foot[2] += self.tuck_height * (1 - progress * 0.7)
            elif phase < 0.4:
                # Stance: brief contact - zero Z offset
                foot[0] += 0.04 + self.glide_extension
                foot[2] += 0.0
            elif phase < 0.55:
                # Stance: double-bounce - minimal vertical oscillation
                bounce_phase = (phase - 0.4) / 0.15
                foot[0] += 0.04 + self.glide_extension - 0.03 * bounce_phase
                foot[2] += self.bounce_amplitude * (1 - np.abs(np.cos(2 * np.pi * bounce_phase)))
            elif phase < 0.7:
                # Swing: retract with smooth ascent
                progress = self.smooth_step((phase - 0.55) / 0.15)
                foot[0] += 0.01 - self.step_length * progress * 0.3
                foot[2] += self.step_height * np.sin(np.pi * progress * 0.6)
            elif phase < 0.8:
                # Swing: descend gradually toward landing
                progress = self.smooth_step((phase - 0.7) / 0.1)
                foot[0] += 0.01 - self.step_length * 0.3 - self.step_length * progress * 0.2
                foot[2] += self.step_height * 0.6 * (1 - progress)
            else:
                # Stance: landed (left diagonal at 0.8) - zero Z offset
                landing_progress = (phase - 0.8) / 0.2
                foot[0] += 0.01 - self.step_length * 0.5 - 0.02 * landing_progress
                foot[2] += 0.0
        
        # Front Right (FR)
        elif leg_name.startswith('FR'):
            if phase < 0.15:
                # Swing: tuck upward and forward
                progress = self.smooth_step(phase / 0.15)
                foot[0] += 0.04 * progress
                foot[2] += self.tuck_height * np.sin(np.pi * progress)
            elif phase < 0.35:
                # Swing: extend forward for contact
                progress = self.smooth_step((phase - 0.15) / 0.2)
                foot[0] += 0.04 + self.glide_extension * progress
                foot[2] += self.tuck_height * (1 - progress * 0.7)
            elif phase < 0.4:
                # Stance: brief contact - zero Z offset
                foot[0] += 0.04 + self.glide_extension
                foot[2] += 0.0
            elif phase < 0.55:
                # Stance: double-bounce - minimal vertical oscillation
                bounce_phase = (phase - 0.4) / 0.15
                foot[0] += 0.04 + self.glide_extension - 0.03 * bounce_phase
                foot[2] += self.bounce_amplitude * (1 - np.abs(np.cos(2 * np.pi * bounce_phase)))
            elif phase < 0.75:
                # Swing: retract with smooth ascent
                progress = self.smooth_step((phase - 0.55) / 0.2)
                foot[0] += 0.01 - self.step_length * progress * 0.4
                foot[2] += self.step_height * np.sin(np.pi * progress * 0.7)
            elif phase < 0.9:
                # Swing: descend gradually toward delayed landing
                progress = self.smooth_step((phase - 0.75) / 0.15)
                foot[0] += 0.01 - self.step_length * 0.4 - self.step_length * progress * 0.2
                foot[2] += self.step_height * 0.7 * (1 - progress)
            else:
                # Stance: landed (right diagonal at 0.9) - zero Z offset
                landing_progress = (phase - 0.9) / 0.1
                foot[0] += 0.01 - self.step_length * 0.6 - 0.02 * landing_progress
                foot[2] += 0.0
        
        # Rear Left (RL)
        elif leg_name.startswith('RL'):
            if phase < 0.15:
                # Stance: double-bounce - minimal vertical oscillation
                bounce_phase = phase / 0.15
                foot[0] -= 0.02 * bounce_phase
                foot[2] += self.bounce_amplitude * (1 - np.abs(np.cos(2 * np.pi * bounce_phase)))
            elif phase < 0.35:
                # Swing: lift and extend backward in glide
                progress = self.smooth_step((phase - 0.15) / 0.2)
                foot[0] -= 0.02 + self.glide_extension * progress
                foot[2] += self.step_height * np.sin(np.pi * progress * 0.8)
            elif phase < 0.55:
                # Swing: swing forward during front bounce
                progress = self.smooth_step((phase - 0.35) / 0.2)
                foot[0] -= 0.02 + self.glide_extension - (self.glide_extension + self.step_length * 0.6) * progress
                foot[2] += self.step_height * 0.8 * (1 - progress * 0.8)
            elif phase < 0.7:
                # Swing: continue forward
                progress = self.smooth_step((phase - 0.55) / 0.15)
                foot[0] -= self.step_length * 0.6 + 0.02 - self.step_length * progress * 0.2
                foot[2] += self.step_height * 0.16 * (1 - progress * 0.5)
            elif phase < 0.8:
                # Swing: descend gradually toward landing
                progress = self.smooth_step((phase - 0.7) / 0.1)
                foot[0] -= self.step_length * 0.8 + 0.02
                foot[2] += self.step_height * 0.08 * (1 - progress)
            else:
                # Stance: landed (left diagonal at 0.8) - zero Z offset
                landing_progress = (phase - 0.8) / 0.2
                foot[0] -= self.step_length * 0.8 + 0.02 + 0.02 * landing_progress
                foot[2] += 0.0
        
        # Rear Right (RR)
        elif leg_name.startswith('RR'):
            if phase < 0.15:
                # Stance: double-bounce - minimal vertical oscillation
                bounce_phase = phase / 0.15
                foot[0] -= 0.02 * bounce_phase
                foot[2] += self.bounce_amplitude * (1 - np.abs(np.cos(2 * np.pi * bounce_phase)))
            elif phase < 0.35:
                # Swing: lift and extend backward in glide
                progress = self.smooth_step((phase - 0.15) / 0.2)
                foot[0] -= 0.02 + self.glide_extension * progress
                foot[2] += self.step_height * np.sin(np.pi * progress * 0.8)
            elif phase < 0.55:
                # Swing: swing forward during front bounce
                progress = self.smooth_step((phase - 0.35) / 0.2)
                foot[0] -= 0.02 + self.glide_extension - (self.glide_extension + self.step_length * 0.6) * progress
                foot[2] += self.step_height * 0.8 * (1 - progress * 0.8)
            elif phase < 0.75:
                # Swing: continue forward
                progress = self.smooth_step((phase - 0.55) / 0.2)
                foot[0] -= self.step_length * 0.6 + 0.02 - self.step_length * progress * 0.25
                foot[2] += self.step_height * 0.16 * (1 - progress * 0.6)
            elif phase < 0.9:
                # Swing: descend gradually toward delayed landing
                progress = self.smooth_step((phase - 0.75) / 0.15)
                foot[0] -= self.step_length * 0.85 + 0.02
                foot[2] += self.step_height * 0.064 * (1 - progress)
            else:
                # Stance: landed (right diagonal at 0.9) - zero Z offset
                landing_progress = (phase - 0.9) / 0.1
                foot[0] -= self.step_length * 0.85 + 0.02 + 0.02 * landing_progress
                foot[2] += 0.0
        
        return foot