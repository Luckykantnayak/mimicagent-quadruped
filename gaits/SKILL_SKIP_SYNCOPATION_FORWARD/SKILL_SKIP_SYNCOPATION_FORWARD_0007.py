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
        self.step_length = 0.13
        self.step_height = 0.11
        self.glide_extension = 0.07
        self.tuck_height = 0.07
        
        # Velocity parameters
        self.rear_bounce_vx = 0.65
        self.rear_bounce_vz = 0.55
        self.rear_bounce_pitch = -0.25
        
        self.glide_vx = 0.5
        self.glide_pitch = 0.35
        
        self.front_contact_vx = 0.35
        self.front_contact_vz = -0.15
        
        self.front_bounce_vx = 0.55
        self.front_bounce_vz = 0.50
        self.front_bounce_pitch = 0.4
        
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
        Bounce dynamics achieved through base vertical velocity oscillation.
        """
        
        # Phase 0.0-0.15: Rear double-bounce
        if phase < 0.15:
            vx = self.rear_bounce_vx
            vy = 0.0
            # Double-bounce: two complete oscillations
            bounce_phase = phase / 0.15
            vz = self.rear_bounce_vz * np.sin(4 * np.pi * bounce_phase)
            pitch_rate = self.rear_bounce_pitch
            roll_rate = 0.0
            yaw_rate = 0.0
            
        # Phase 0.15-0.35: Long glide forward
        elif phase < 0.35:
            vx = self.glide_vx
            vy = 0.0
            # Ballistic arc
            glide_progress = (phase - 0.15) / 0.2
            vz = 0.4 * (1 - 2 * glide_progress)
            pitch_rate = self.glide_pitch * (1 - glide_progress)
            roll_rate = 0.0
            yaw_rate = 0.0
            
        # Phase 0.35-0.4: Front brief contact
        elif phase < 0.4:
            vx = self.front_contact_vx
            vy = 0.0
            vz = self.front_contact_vz
            pitch_rate = 0.4
            roll_rate = 0.0
            yaw_rate = 0.0
            
        # Phase 0.4-0.55: Front double-bounce
        elif phase < 0.55:
            vx = self.front_bounce_vx
            vy = 0.0
            # Double-bounce: two complete oscillations
            bounce_phase = (phase - 0.4) / 0.15
            vz = self.front_bounce_vz * np.sin(4 * np.pi * bounce_phase)
            pitch_rate = self.front_bounce_pitch
            roll_rate = 0.0
            yaw_rate = 0.0
            
        # Phase 0.55-0.8: Extended asymmetric glide
        elif phase < 0.8:
            vx = self.extended_glide_vx
            vy = 0.0
            # Ballistic descent
            glide_progress = (phase - 0.55) / 0.25
            vz = 0.35 * (1 - 2 * glide_progress)
            pitch_rate = self.extended_glide_pitch * (1 - 0.5 * glide_progress)
            roll_rate = 0.0
            yaw_rate = 0.0
            
        # Phase 0.8-0.9: Left diagonal pair lands
        elif phase < 0.9:
            landing_progress = (phase - 0.8) / 0.1
            vx = self.extended_glide_vx * (1 - 0.6 * landing_progress)
            vy = -0.03 * np.sin(np.pi * landing_progress)
            # Decelerate to small negative (contact force intent)
            vz = -0.2 + 0.15 * landing_progress
            roll_rate = -0.25 * np.sin(np.pi * landing_progress)
            pitch_rate = 0.0
            yaw_rate = 0.06 * np.sin(np.pi * landing_progress)
            
        # Phase 0.9-1.0: Right diagonal pair completes landing
        else:
            landing_progress = (phase - 0.9) / 0.1
            vx = self.landing_decel * (1 - landing_progress)
            vy = 0.03 * np.sin(np.pi * landing_progress)
            # Small negative velocity for contact
            vz = -0.05 * (1 - landing_progress * 0.5)
            roll_rate = 0.25 * np.sin(np.pi * landing_progress)
            pitch_rate = 0.0
            yaw_rate = -0.06 * np.sin(np.pi * landing_progress)
        
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
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in BODY frame for each leg based on phase.
        Stance phases: feet at or slightly below zero Z to maintain ground contact.
        Bounce effect achieved through base motion, not foot Z oscillation.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Front Left (FL)
        if leg_name.startswith('FL'):
            if phase < 0.15:
                # Swing: tuck upward and forward
                progress = self.smooth_step(phase / 0.15)
                foot[0] += 0.04 * progress
                foot[2] += self.tuck_height * np.sin(np.pi * progress)
            elif phase < 0.3:
                # Swing: extend forward for contact
                progress = self.smooth_step((phase - 0.15) / 0.15)
                foot[0] += 0.04 + self.glide_extension * progress
                foot[2] += self.tuck_height * (1 - progress * 0.5)
            elif phase < 0.35:
                # Swing: descend to contact
                progress = self.smooth_step((phase - 0.3) / 0.05)
                foot[0] += 0.04 + self.glide_extension
                foot[2] += self.tuck_height * 0.5 * (1 - progress)
            elif phase < 0.4:
                # Stance: brief contact - steady ground position
                foot[0] += 0.04 + self.glide_extension
                foot[2] -= 0.01
            elif phase < 0.55:
                # Stance: double-bounce - feet stay grounded
                bounce_phase = (phase - 0.4) / 0.15
                foot[0] += 0.04 + self.glide_extension - 0.03 * bounce_phase
                foot[2] -= 0.01
            elif phase < 0.68:
                # Swing: lift and retract
                progress = self.smooth_step((phase - 0.55) / 0.13)
                foot[0] += 0.01 - self.step_length * progress * 0.35
                foot[2] += self.step_height * np.sin(np.pi * progress * 0.7)
            elif phase < 0.8:
                # Swing: descend toward landing
                progress = self.smooth_step((phase - 0.68) / 0.12)
                foot[0] += 0.01 - self.step_length * 0.35 - self.step_length * progress * 0.2
                foot[2] += self.step_height * 0.7 * (1 - progress)
            else:
                # Stance: left diagonal landed - grounded
                landing_progress = (phase - 0.8) / 0.2
                foot[0] += 0.01 - self.step_length * 0.55 - 0.02 * landing_progress
                foot[2] -= 0.015
        
        # Front Right (FR)
        elif leg_name.startswith('FR'):
            if phase < 0.15:
                # Swing: tuck upward and forward
                progress = self.smooth_step(phase / 0.15)
                foot[0] += 0.04 * progress
                foot[2] += self.tuck_height * np.sin(np.pi * progress)
            elif phase < 0.3:
                # Swing: extend forward for contact
                progress = self.smooth_step((phase - 0.15) / 0.15)
                foot[0] += 0.04 + self.glide_extension * progress
                foot[2] += self.tuck_height * (1 - progress * 0.5)
            elif phase < 0.35:
                # Swing: descend to contact
                progress = self.smooth_step((phase - 0.3) / 0.05)
                foot[0] += 0.04 + self.glide_extension
                foot[2] += self.tuck_height * 0.5 * (1 - progress)
            elif phase < 0.4:
                # Stance: brief contact - steady ground position
                foot[0] += 0.04 + self.glide_extension
                foot[2] -= 0.01
            elif phase < 0.55:
                # Stance: double-bounce - feet stay grounded
                bounce_phase = (phase - 0.4) / 0.15
                foot[0] += 0.04 + self.glide_extension - 0.03 * bounce_phase
                foot[2] -= 0.01
            elif phase < 0.72:
                # Swing: lift and retract
                progress = self.smooth_step((phase - 0.55) / 0.17)
                foot[0] += 0.01 - self.step_length * progress * 0.4
                foot[2] += self.step_height * np.sin(np.pi * progress * 0.75)
            elif phase < 0.88:
                # Swing: descend toward delayed landing
                progress = self.smooth_step((phase - 0.72) / 0.16)
                foot[0] += 0.01 - self.step_length * 0.4 - self.step_length * progress * 0.2
                foot[2] += self.step_height * 0.75 * (1 - progress)
            elif phase < 0.9:
                # Swing: final approach
                progress = self.smooth_step((phase - 0.88) / 0.02)
                foot[0] += 0.01 - self.step_length * 0.6
                foot[2] += 0.01 * (1 - progress)
            else:
                # Stance: right diagonal landed - grounded
                landing_progress = (phase - 0.9) / 0.1
                foot[0] += 0.01 - self.step_length * 0.6 - 0.02 * landing_progress
                foot[2] -= 0.015
        
        # Rear Left (RL)
        elif leg_name.startswith('RL'):
            if phase < 0.15:
                # Stance: double-bounce - feet stay grounded
                bounce_phase = phase / 0.15
                foot[0] -= 0.02 * bounce_phase
                foot[2] -= 0.01
            elif phase < 0.3:
                # Swing: lift and extend backward
                progress = self.smooth_step((phase - 0.15) / 0.15)
                foot[0] -= 0.02 + self.glide_extension * progress
                foot[2] += self.step_height * np.sin(np.pi * progress * 0.8)
            elif phase < 0.5:
                # Swing: swing forward during front phases
                progress = self.smooth_step((phase - 0.3) / 0.2)
                foot[0] -= 0.02 + self.glide_extension - (self.glide_extension + self.step_length * 0.5) * progress
                foot[2] += self.step_height * 0.8 * (1 - progress * 0.7)
            elif phase < 0.68:
                # Swing: continue forward
                progress = self.smooth_step((phase - 0.5) / 0.18)
                foot[0] -= self.step_length * 0.5 + 0.02 - self.step_length * progress * 0.25
                foot[2] += self.step_height * 0.24 * (1 - progress * 0.6)
            elif phase < 0.8:
                # Swing: descend toward landing
                progress = self.smooth_step((phase - 0.68) / 0.12)
                foot[0] -= self.step_length * 0.75 + 0.02
                foot[2] += self.step_height * 0.096 * (1 - progress)
            else:
                # Stance: left diagonal landed - grounded
                landing_progress = (phase - 0.8) / 0.2
                foot[0] -= self.step_length * 0.75 + 0.02 + 0.02 * landing_progress
                foot[2] -= 0.015
        
        # Rear Right (RR)
        elif leg_name.startswith('RR'):
            if phase < 0.15:
                # Stance: double-bounce - feet stay grounded
                bounce_phase = phase / 0.15
                foot[0] -= 0.02 * bounce_phase
                foot[2] -= 0.01
            elif phase < 0.3:
                # Swing: lift and extend backward
                progress = self.smooth_step((phase - 0.15) / 0.15)
                foot[0] -= 0.02 + self.glide_extension * progress
                foot[2] += self.step_height * np.sin(np.pi * progress * 0.8)
            elif phase < 0.5:
                # Swing: swing forward during front phases
                progress = self.smooth_step((phase - 0.3) / 0.2)
                foot[0] -= 0.02 + self.glide_extension - (self.glide_extension + self.step_length * 0.5) * progress
                foot[2] += self.step_height * 0.8 * (1 - progress * 0.7)
            elif phase < 0.72:
                # Swing: continue forward
                progress = self.smooth_step((phase - 0.5) / 0.22)
                foot[0] -= self.step_length * 0.5 + 0.02 - self.step_length * progress * 0.3
                foot[2] += self.step_height * 0.24 * (1 - progress * 0.7)
            elif phase < 0.88:
                # Swing: descend toward delayed landing
                progress = self.smooth_step((phase - 0.72) / 0.16)
                foot[0] -= self.step_length * 0.8 + 0.02
                foot[2] += self.step_height * 0.072 * (1 - progress)
            elif phase < 0.9:
                # Swing: final approach
                progress = self.smooth_step((phase - 0.88) / 0.02)
                foot[0] -= self.step_length * 0.8 + 0.02
                foot[2] += 0.01 * (1 - progress)
            else:
                # Stance: right diagonal landed - grounded
                landing_progress = (phase - 0.9) / 0.1
                foot[0] -= self.step_length * 0.8 + 0.02 + 0.02 * landing_progress
                foot[2] -= 0.015
        
        return foot