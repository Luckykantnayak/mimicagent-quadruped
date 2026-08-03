import numpy as np
from motion.base import BaseMotionGenerator


class SkillCarryboxMotionGenerator(BaseMotionGenerator):
    """
    Hand-authored PICK UP -> CARRY FORWARD -> PLACE a box — a REFERENCE for
    KINEMATIC OBJECT INTERACTION. The BOX is the primary trajectory (a mocap
    object animated via compute_object_poses); the HANDS grip its two side faces
    (IK end-effectors), so wherever the box goes the hands follow, and vice versa.
    There is no grasp physics — while held, the box simply IS where the hands are.

    Sequence:
      0.00-0.16  step up to the box, hinge down, hands onto its sides
      0.16-0.30  grip and LIFT the box, straightening up to carry height
      0.30-0.68  walk two steps FORWARD carrying the box at the chest
      0.68-0.86  hinge down and PLACE the box on the ground ahead
      0.86-1.00  release, straighten back up

    Run:
      python simulator/run.py --robot humanoid --mode SKILL_CARRYBOX \
             --scene assets/scenes/carry.xml --grounding off
    """

    def __init__(self, init_ee_pos, freq=0.3):
        super().__init__(init_ee_pos, freq=freq)
        self.support_ees = ["right_foot", "left_foot"]

        self.box_half = 0.12          # box half-size
        self.grip_y = 0.15            # hand offset from box centre to its side face
        self.ground_z = 0.12          # box centre height resting on the floor
        self.carry_z = 0.92           # box centre height while carried (at the chest)
        self.carry_fwd = 0.30         # box held this far in FRONT of the pelvis
        self.stand = 0.88

        self.box_x0 = 0.42            # box start x (on the ground)
        self.box_x1 = 0.92            # box destination x
        # pelvis x while squatting so the SHOULDERS sit over the box (reach down)
        self.pick_bx = self.box_x0 - 0.40
        self.place_bx = self.box_x1 - 0.40
        self.squat_z = 0.38           # deep pelvis height when picking / placing
        self.hinge = 1.1              # forward hip hinge (rad) when picking/placing

        self.p_grip = 0.16
        self.p_lift = 0.30
        self.p_carry = 0.68
        self.p_place = 0.86

    # ---- the box trajectory (world frame): held at pelvis_x + carry_fwd ------
    def _box_pos(self, p):
        carry_start = self.pick_bx + self.carry_fwd      # box x at the top of the lift
        carry_end = self.place_bx + self.carry_fwd       # box x before it is set down
        if p < self.p_grip:                              # resting on the ground
            return np.array([self.box_x0, 0.0, self.ground_z])
        if p < self.p_lift:                              # lift up toward the chest
            s = _smooth((p - self.p_grip) / (self.p_lift - self.p_grip))
            x = self.box_x0 + (carry_start - self.box_x0) * s
            return np.array([x, 0.0, self.ground_z + (self.carry_z - self.ground_z) * s])
        if p < self.p_carry:                             # carried forward at the chest
            s = _smooth((p - self.p_lift) / (self.p_carry - self.p_lift))
            return np.array([carry_start + (carry_end - carry_start) * s, 0.0, self.carry_z])
        if p < self.p_place:                             # lower onto the ground ahead
            s = _smooth((p - self.p_carry) / (self.p_place - self.p_carry))
            x = carry_end + (self.box_x1 - carry_end) * s
            return np.array([x, 0.0, self.carry_z + (self.ground_z - self.carry_z) * s])
        return np.array([self.box_x1, 0.0, self.ground_z])   # placed

    def compute_object_poses(self, phase):
        return {"box": (self._box_pos(phase), np.array([1.0, 0.0, 0.0, 0.0]))}

    # ---- hands grip the box's side faces; feet shuffle under the pelvis -----
    def compute_ee_position_world_frame(self, ee_name, phase):
        if "hand" in ee_name:
            if phase >= self.p_place:
                return None                              # released the box -> arms free
            y = self.grip_y if "left" in ee_name else -self.grip_y
            box = self._box_pos(phase)
            grip = np.array([box[0], box[1] + y, box[2]])
            if phase < self.p_grip:                      # reach DOWN as the squat descends
                s = _smooth(phase / self.p_grip)
                bx, bz, _ = self._base(phase)
                ready = np.array([bx + 0.22, y, bz - 0.05])   # in front of the pelvis
                return ready * (1 - s) + grip * s
            return grip

        # feet: authored in WORLD frame. During the pick/place SQUAT they plant
        # under the pelvis at a NARROW width so the reaching hands (at the box
        # sides) clear the thighs. During the CARRY they take two real alternating
        # STEPS — each foot holds a FIXED world x through its stance (no drag as the
        # pelvis advances), then swings ahead — so a planted foot never slides.
        p, right = phase, "right" in ee_name
        bx = self._base(p)[0]
        squatting = p < self.p_lift or p >= self.p_carry
        if squatting:
            y = -0.06 if right else 0.06
            return np.array([bx - 0.04, y, 0.05])
        y = -0.10 if right else 0.10
        return np.array([self._step_x(right, p), y, 0.05 + self._step_lift(right, p)])

    # ---- stepping during the carry: fixed-x stance, arc-forward swing ----------
    def _carry_u(self, p):
        return (p - self.p_lift) / (self.p_carry - self.p_lift)        # 0..1

    def _step_plants(self):
        x0, x1 = self.pick_bx - 0.04, self.place_bx - 0.04
        return [x0, 0.5 * (x0 + x1), x1]                               # 2 steps

    def _step_phase(self, right, p):
        off = 0.0 if right else 0.5
        raw = self._carry_u(p) * 2.0 + off
        return int(raw), (raw % 1.0)                                   # block, cyc

    def _step_x(self, right, p):
        plants = self._step_plants()
        idx, cyc = self._step_phase(right, p)
        p0 = plants[min(idx, 2)]
        p1 = plants[min(idx + 1, 2)]
        if cyc < 0.65:
            return p0                                                  # stance: fixed x
        return p0 + (p1 - p0) * _smooth((cyc - 0.65) / 0.35)           # swing forward

    def _step_lift(self, right, p):
        idx, cyc = self._step_phase(right, p)
        if cyc < 0.65:
            return 0.0
        return 0.07 * np.sin(np.pi * (cyc - 0.65) / 0.35)

    # ---- base: squat+hinge over the box to pick/place; walk between ----------
    def _base(self, p):
        if p < self.p_grip:                              # squat+hinge down over the box
            s = _smooth(p / self.p_grip)
            return self.pick_bx * s, self.stand + (self.squat_z - self.stand) * s, self.hinge * s
        if p < self.p_lift:                              # rise, box to the chest
            s = _smooth((p - self.p_grip) / (self.p_lift - self.p_grip))
            return self.pick_bx, self.squat_z + (self.stand - self.squat_z) * s, self.hinge * (1 - s) + 0.1
        if p < self.p_carry:                             # walk forward, upright
            s = _smooth((p - self.p_lift) / (self.p_carry - self.p_lift))
            return self.pick_bx + (self.place_bx - self.pick_bx) * s, self.stand, 0.1
        if p < self.p_place:                             # squat+hinge down to place
            s = _smooth((p - self.p_carry) / (self.p_place - self.p_carry))
            return self.place_bx, self.stand + (self.squat_z - self.stand) * s, 0.1 + (self.hinge - 0.1) * s
        s = _smooth((p - self.p_place) / (1 - self.p_place))   # straighten up
        return self.place_bx, self.squat_z + (self.stand - self.squat_z) * s, self.hinge * (1 - s)

    def update_base_motion(self, phase, dt):
        x, z, pitch = self._base(phase)
        self.root_pos = np.array([x, 0.0, z])
        self.root_quat = np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0])

    def contacts(self, phase):
        # Both feet planted through the squats; during the carry a foot bears
        # weight only while in stance (not mid-swing), keeping ≥1 support.
        if phase < self.p_lift or phase >= self.p_carry:
            return self.support_ees
        c = []
        for right, name in ((True, "right_foot"), (False, "left_foot")):
            _, cyc = self._step_phase(right, phase)
            if cyc < 0.65:
                c.append(name)
        return c or self.support_ees


def _smooth(s):
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * (3 - 2 * s)
