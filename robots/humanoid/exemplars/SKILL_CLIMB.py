import numpy as np
from motion.base import BaseMotionGenerator


class SkillClimbMotionGenerator(BaseMotionGenerator):
    """
    Hand-authored human-like WALL CLIMB onto a ~0.75 m slab (about half the
    humanoid's height). The feet press on the front FACE (the wall) to drive the
    climb; the body never passes through the block; and the arms stay IK-driven the
    whole time (a fore-aft swing during the walk that flows straight into the reach)
    so there is no jump from walking to reaching.

      0.00-0.30  WALK up to the wall (strides + fore-aft arm swing, hands kept
                 behind the face)
      0.30-0.46  hinge forward, sweep both HANDS up and over onto the top edge
      0.46-0.70  CLIMB: feet smear on the FACE and walk UP it while the arms press
                 and the hips rise IN FRONT of the wall; hands release mid-way as
                 the hips outreach them and the legs take over
      0.70-0.88  hips rise over the top as the feet come off the wall and step up
                 onto it (a slow, single lip transition)
      0.88-1.00  stand up tall on top

    Run:
      python simulator/run.py --robot humanoid --mode SKILL_CLIMB \
             --scene assets/scenes/slab.xml --grounding off
    """

    def __init__(self, init_ee_pos, freq=0.26):
        super().__init__(init_ee_pos, freq=freq)
        self.support_ees = ["right_foot", "left_foot", "right_hand", "left_hand"]

        self.face = 0.65
        self.top = 0.75
        self.foot_off = 0.05
        self.stand = 0.88

        self.walk_x = 0.42          # pelvis x at the end of the walk
        self.wall_foot_x = 0.60     # feet smear on the face
        self.base_foot_x = 0.56     # feet at the base of the wall
        self.hand_x = 0.68          # hands on the top edge
        self.foot_top_x = 0.80      # feet plant on top
        self.stand_x = 0.92
        self.wall_top_foot = 0.50   # how high the feet reach on the face (leg-reach bound)

        self.p_walk = 0.30
        self.p_reach = 0.46
        self.p_climb = 0.70
        self.p_over = 0.88
        self.p_release = 0.52       # let go while the pelvis is still within arm reach
        self.press_pitch = 0.95

    # ------------------------------------------------------------------ base
    def _base(self, p):
        if p < self.p_walk:                          # walk in upright
            s = _smooth(p / self.p_walk)
            return self.walk_x * s, self.stand, 0.0
        if p < self.p_reach:                         # hinge, hands sweep to the top
            s = _smooth((p - self.p_walk) / (self.p_reach - self.p_walk))
            return self.walk_x, self.stand - 0.05 * s, self.press_pitch * s
        if p < self.p_climb:                         # rise UP in FRONT of the face
            s = _smooth((p - self.p_reach) / (self.p_climb - self.p_reach))
            return self.walk_x + 0.14 * s, (self.stand - 0.05) + (1.30 - (self.stand - 0.05)) * s, \
                self.press_pitch * (1 - 0.5 * s)
        if p < self.p_over:                          # hips up over the top; feet top out
            s = _smooth((p - self.p_climb) / (self.p_over - self.p_climb))
            x = (self.walk_x + 0.14) + (self.stand_x - (self.walk_x + 0.14)) * s
            z = 1.30 + (self.top + self.stand - 1.30) * s
            return x, z, self.press_pitch * 0.5 * (1 - s)
        return self.stand_x, self.top + self.stand, 0.0

    def update_base_motion(self, phase, dt):
        x, z, pitch = self._base(phase)
        self.root_pos = np.array([x, 0.0, z])
        self.root_quat = np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0])

    # ------------------------------------------------------------------ limbs
    def compute_ee_position_world_frame(self, ee_name, phase):
        p = phase
        if "hand" in ee_name:
            y = 0.18 if "left" in ee_name else -0.18
            # arms hang naturally at the sides at the end of the walk (below)
            hang = np.array([self.walk_x + 0.05, -0.25 if "left" not in ee_name else 0.25, 0.84])
            if p < self.p_walk:                      # arms swung by JOINT override (natural)
                return None
            if p < self.p_reach:                     # from the hang, sweep UP then OVER
                s = _smooth((p - self.p_walk) / (self.p_reach - self.p_walk))
                up = np.array([0.58, y, 0.92])       # raise in front, above the lip
                goal = np.array([self.hand_x, y, self.top])
                if s < 0.5:
                    t = s / 0.5
                    return hang * (1 - t) + up * t
                t = (s - 0.5) / 0.5
                return up * (1 - t) + goal * t
            press = np.array([self.hand_x, y, self.top])
            if p < self.p_release:                   # planted on the top, pressing
                return press
            # let go and bring the arms to the sides — kept IK and tracking the
            # (rising) base, lifting off quickly so the hands never chase the top
            # once the hips have risen past arm's reach (no release pop)
            s = _smooth(min((p - self.p_release) / 0.14, 1.0))
            bx, bz, _ = self._base(p)
            side = np.array([bx + 0.05, (0.24 if "left" in ee_name else -0.24), bz - 0.05])
            return press * (1 - s) + side * s

        # ---- feet ----
        y = 0.12 if "left" in ee_name else -0.12
        floor = self.foot_off
        top = self.top + self.foot_off
        fx_top = self.foot_top_x + (0.06 if "left" in ee_name else -0.06)

        if p < self.p_walk:                          # WALK: two clear strides
            return self._walk_foot(ee_name, p / self.p_walk)
        if p < self.p_reach:                         # planted at the base of the wall
            return np.array([self.base_foot_x, y, floor])
        if p < self.p_climb:                         # SMEAR + walk UP the front face
            s = _smooth((p - self.p_reach) / (self.p_climb - self.p_reach))
            x = self.base_foot_x + (self.wall_foot_x - self.base_foot_x) * min(s * 2, 1.0)
            z = floor + (self.wall_top_foot - floor) * s
            off = 0.0 if "right" in ee_name else np.pi
            z += 0.03 * max(0.0, np.sin(4 * np.pi * s + off))    # small climbing pushes
            return np.array([x, y, z])
        if p < self.p_over:                          # up over the lip onto the top (slow)
            s = _smooth((p - self.p_climb) / (self.p_over - self.p_climb))
            sz = _smooth(min(s / 0.55, 1.0))         # clear the lip first (height leads)
            sx = _smooth(max((s - 0.4) / 0.6, 0.0))  # then forward onto the top
            x = self.wall_foot_x + (fx_top - self.wall_foot_x) * sx
            z = self.wall_top_foot + (top - self.wall_top_foot) * sz
            return np.array([x, y, z])
        return np.array([fx_top, y, top])            # planted on the top

    def _walk_foot(self, ee_name, s):
        y = 0.12 if "left" in ee_name else -0.12
        floor = self.foot_off
        if "right" in ee_name:
            p0, p1, a, b = 0.06, self.base_foot_x, 0.10, 0.45
        else:
            p0, p1, a, b = 0.10, self.base_foot_x, 0.52, 0.85
        if s < a:
            return np.array([p0, y, floor])
        if s < b:
            pr = (s - a) / (b - a)
            x = p0 + (p1 - p0) * _smooth(pr)
            return np.array([x, y, floor + 0.13 * np.sin(np.pi * pr)])
        return np.array([p1, y, floor])

    def compute_joint_overrides(self, phase):
        # natural fore-aft arm swing during the walk, eased to a hang by walk-end
        if phase < self.p_walk:
            damp = 1.0 - _smooth(phase / self.p_walk)
            sw = 0.5 * damp * np.cos(2 * np.pi * (phase / self.p_walk) * 2.0)
            return {"right_shoulder_z": sw, "left_shoulder_z": sw,
                    "neck_z": 0.1 * np.sin(2 * np.pi * phase)}
        return {}

    def contacts(self, phase):
        return self.support_ees


def _smooth(s):
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * (3 - 2 * s)
