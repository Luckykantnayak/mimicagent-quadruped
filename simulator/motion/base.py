
from abc import ABC, abstractmethod
import numpy as np
from utils.math_utils import *


class BaseMotionGenerator(ABC):
    """
    Abstract base class for all motion / skill generators.

    1. Time & Phase
       - Internal time variable `t`
       - Scalar phase `phase ∈ [0,1]`
       - Phase representation via `compute_phase(t)`
       - phase = 0 → motion start
       - phase = 1 → motion end

    2. BASE STATE (WORLD FRAME)
       - Base position:    root_pos  ∈ ℝ³
       - Base orientation: root_quat ∈ ℍ  ([w, x, y, z])

    3. BASE MOTION (WORLD FRAME) — two equally valid styles, chosen per skill
       - INTEGRATE a commanded velocity (what the default update_base_motion does):
            set_velocity_command(vx, vy, vz)
            set_angular_velocity_command(roll_rate, pitch_rate, yaw_rate)
       - Or override update_base_motion and write root_pos / root_quat
         ABSOLUTELY. Use this whenever the WORLD fixes the base path — mounting a
         step, squatting to a box on the floor, rising onto a ledge.

    4. END-EFFECTOR TARGETS — THREE CONTROL CHANNELS, PER LIMB PER PHASE
       NOTHING here prescribes a channel for any limb. The discriminator is
       COUPLING — what actually determines where that limb goes. The choice is
       made per limb, per phase, and may change from one phase to the next:

         (a) WORLD-frame IK — compute_ee_position_world_frame(ee_name, phase)
             The WORLD fixes the endpoint: a contact resting on a surface, or a
             limb that must COINCIDE with an object it grips or holds.

         (b) BODY-frame IK — compute_ee_position_body_frame(ee_name, phase)
             The endpoint is fixed RELATIVE TO THE TORSO: a swing foot.

         (c) JOINT — compute_joint_overrides(phase)   [see 5]
             The limb's CONFIGURATION is the point and its endpoint is
             unconstrained: a gesture, a wave, a kick.

       (a) and (b) both solve IK, and BOTH DEFAULT TO None. A None target makes
       IK SKIP that limb — which is how a limb is handed to the joint channel.
       Targets are keyed by name over the UNION IK ROSTER (`self.ee_names`): the
       end-effectors AND every declared contact body, so a knee / forearm /
       pelvis can be PLACED by IK exactly like a foot.

    5. JOINT OVERRIDES (OPTIONAL)
       - step() returns a "joint_overrides" dict {joint_name: angle_radians}
       - Joints listed there are set directly instead of being solved by IK
       - Driving an expressive limb's ENDPOINT through IK dissolves its shape
         into whatever happens to reach that point; drive the ANGLES instead and
         the gesture survives.

    6. DECLARATIONS (what the framework and the quality checks rely on)
       - contacts(phase)             -> load-bearing names this phase; [] = flight
       - compute_object_poses(phase) -> {name: (world_pos, quat)} for mocap objects
       - attachments(phase)          -> {object: [ee, ...]} rigidly HELD this phase

    ----------------------------------------------------------------------
    AVAILABLE SKILL FUNCTIONS
    ----------------------------------------------------------------------

    The following functions are available to be defined or used by a skill:

        - compute_ee_position_world_frame(ee_name, phase)
        - compute_ee_position_body_frame(ee_name, phase)
        - compute_joint_overrides(phase)
        - contacts(phase)
        - compute_object_poses(phase)  /  attachments(phase)
        - compute_phase(t)
        - update_base_motion(phase, dt)
        - set_velocity_command(...)    /  set_angular_velocity_command(...)
        - limb_root_world(...)         /  reach_limited_target(...)
        - advance_base_planar(...)     [translate a PITCHED/ROLLED base correctly]

    The skill decides how (or whether) to use them.

    ----------------------------------------------------------------------
    FRAME CONVENTIONS
    ----------------------------------------------------------------------

    - Base pose is ALWAYS expressed in WORLD frame.
    - End-effector targets are authored in EITHER frame — world (ABSOLUTE: the
      target stays put as the base moves) or body (RELATIVE to the torso). Which
      one is a per-limb, per-phase decision, NOT a fixed convention.
    - A skill may be CYCLIC (phase 1 returns to the phase-0 pose) or ONE-SHOT
      (distinct start and end poses). Neither is privileged.

    ======================================================================
    """

    # ==========================================================
    # INITIALIZATION
    # ==========================================================
    def __init__(self, init_ee_pos, freq=1.0):
        """
        Args:
            init_ee_pos : dict[str, np.ndarray]
                Nominal BODY-frame positions of the UNION IK ROSTER, keyed by name:
                the end-effectors (feet, hands) AND every declared CONTACT BODY
                (knees, thighs, forearms, upper arms, pelvis, torso, head — whatever
                the robot spec registers). Any of these can be given a world/body
                target (compute_ee_position_world_frame / _body_frame) and IK will
                PLACE it like a foot; return None and it is FK-only. So a crawl can
                plant its KNEES on the floor by returning their contact points, a
                plank its FOREARMS, a sit checks its PELVIS — not just feet/hands.
            freq : float
                Phase RATE (Hz): phase advances as (freq * t) % 1, so ONE pass
                through phase 0->1 takes 1/freq SECONDS. This sets the skill's
                WALL-CLOCK DURATION and therefore every joint/base velocity:
                    duration_of_one_pass = 1.0 / freq   seconds.
                freq is a FREE choice per skill — pick it from the motion's real
                tempo, NOT a fixed default. The base default 1.0 means the ENTIRE
                skill plays in ONE SECOND, which is far too fast for anything with
                several events (a multi-step climb, a pick-carry-place): it forces
                huge per-frame joint steps that fail no_teleport / joint_velocity.
                Set self.freq = 1.0 / duration_s, where duration_s is the seconds
                the motion should take (the planner sizes it so joint speeds stay
                under the physical-summary ceilings — e.g. ~1 s per stance step of a
                walk/climb, a few seconds for a pick-carry-place, faster for a whip
                or a kick). A cyclic gait's freq is 1/(one cycle's seconds).
        """
        self.freq = freq

        # End-effector reference positions (BODY frame)
        self.init_ee_pos = {
            k: v.copy() for k, v in init_ee_pos.items()
        }
        self.base_ee_pos = {
            k: v.copy() for k, v in init_ee_pos.items()
        }

        # End-effector list
        self.ee_names = list(init_ee_pos.keys())

        # SUPPORT SET (optional): the end-effectors that bear weight on the ground
        # at ANY point in this skill. The framework grounds the base to the lowest
        # of these and pins them while planted. Default (None) = the robot's
        # spec-declared contact feet. Override to cross postures, e.g.:
        #   walk      -> feet (default)
        #   crawl     -> feet + hands
        #   cartwheel -> feet + hands  (grounding follows whichever is lowest)
        #   quadruped balancing on its rear legs -> rear feet only
        # Any end-effector NOT listed is free to move through the air.
        self.support_ees = None

        # Time
        self.t = 0.0

        # Base state (WORLD frame)
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Per-EE limb geometry {name: {reach, root_offset}}, injected by the
        # controller (reach = how far the limb extends from its own root; root_offset
        # = the limb root's body-frame offset from the base). Powers the reach helpers.
        self._limb_geom = {}

        # Commands (WORLD frame)
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    # ==========================================================
    # RESET
    # ==========================================================
    def reset(self, root_pos, root_quat):
        """
        Reset internal state.

        Args:
            root_pos : np.ndarray (3,)
                Initial base position in WORLD frame.
            root_quat : np.ndarray (4,)
                Initial base orientation [w, x, y, z].
        """
        self.root_pos = root_pos.copy()
        self.root_quat = root_quat.copy()
        self.t = 0.0

    # ==========================================================
    # COMMAND INTERFACE
    # ==========================================================
    def set_velocity_command(self, vx=0.0, vy=0.0, vz=0.0):
        """
        Set desired base linear velocity in WORLD frame.
        """
        self.vel_world = np.array([vx, vy, vz])

    def set_angular_velocity_command(self, roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0):
        """
        Set desired base angular velocity in WORLD frame.
        """
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])

    # ==========================================================
    # MAIN STEP (DO NOT MODIFY)
    # ==========================================================
    def step(self, dt):
        """
        Advance the motion generator by one timestep.

        Pipeline:
            1. Compute phase
            2. Update base pose
            3. Compute end-effector targets in BODY frame
            4. Transform end-effector targets to WORLD frame
        """

        phase = self.compute_phase(self.t)

        self.update_base_motion(phase, dt)

        # Per end-effector, pick the control channel:
        #   world-frame target (ground-anchored)  -> IK
        #   body-frame target                      -> IK
        #   None (neither)                         -> IK SKIPS it; the limb is
        #        driven by compute_joint_overrides (a joint-space gesture).
        ee_body, ee_world, ee_frame = {}, {}, {}
        for ee in self.ee_names:
            w = self.compute_ee_position_world_frame(ee, phase)
            if w is not None:
                ee_world[ee] = np.asarray(w, dtype=float)
                ee_body[ee] = ee_world[ee]                    # unused downstream
                ee_frame[ee] = "world"                        # ABSOLUTE target (stays put)
                continue
            b = self.compute_ee_position_body_frame(ee, phase)
            if b is None:
                ee_body[ee] = None
                ee_world[ee] = None                           # not IK-driven
                ee_frame[ee] = None
                continue
            b = np.asarray(b, dtype=float)
            ee_body[ee] = b
            ee_world[ee] = body_to_world_position(b, self.root_pos, self.root_quat)
            ee_frame[ee] = "body"                             # RELATIVE to the base

        self.t += dt

        return {
            "time": self.t,
            "phase": phase,
            "root_pos": self.root_pos.copy(),
            "root_quat": self.root_quat.copy(),
            "ee_positions_body": ee_body,
            "ee_positions_world": ee_world,
            "ee_frame": ee_frame,   # {ee: "world"|"body"|None} — authoring frame, so the
                                    # controller's reach-seat shifts only BODY-frame free
                                    # targets when it moves the base along z (world targets
                                    # are absolute; contacts are the datum).
            # Optional per-EE orientation targets (None entries are omitted): a
            # DECLARED CONTACT is laid flat on its surface automatically, so this
            # carries only the attitudes a skill asks for EXPLICITLY.
            "ee_orientations": {ee: o for ee in self.ee_names
                                if (o := self.compute_ee_orientation_world_frame(ee, phase))
                                is not None},
            "joint_overrides": self.compute_joint_overrides(phase),
            "contacts": self.contacts(phase),          # per-phase contact EE names, or None
            "object_poses": self.compute_object_poses(phase),   # scene objects (mocap)
            "attachments": self.attachments(phase),    # objects rigidly held by EEs this phase
        }

    # ==========================================================
    # PHASE
    # ==========================================================
    def compute_phase(self, t):
        """
        Compute scalar phase in [0,1].

        Default behavior:
            phase progresses linearly with time based on frequency.
        """
        return (self.freq * t) % 1.0

    # ==========================================================
    # BASE MOTION
    # ==========================================================
    def update_base_motion(self, phase, dt):
        """
        Update base position and orientation.

        Default behavior:
            Integrates commanded WORLD-frame velocities.
        """
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    # ==========================================================
    # REACHABILITY (construct targets the limb can actually reach)
    # ==========================================================
    def limb_root_world(self, ee_name):
        """World position of this end-effector's LIMB ROOT (e.g. a hand's shoulder,
        a foot's hip), from the current base pose. Approximate — it ignores any
        torso joints between the base and the limb root (parity with authoring a
        fixed shoulder offset). Falls back to the base position if geometry is
        unavailable."""
        g = self._limb_geom.get(ee_name)
        if not g:
            return np.asarray(self.root_pos, dtype=float)
        R = quat_to_rotation_matrix(self.root_quat)
        return np.asarray(self.root_pos, dtype=float) + R @ g["root_offset"]

    def reach_limited_target(self, ee_name, desired_world_pos, margin=0.95):
        """Return a WORLD target for `ee_name` that the limb can actually REACH:
        the desired point if it is within `margin * reach` of the limb root, else
        the closest point on that reachable ball (projected along the same
        direction). Use this to CONSTRUCT world-frame IK targets far from the
        standing pose (a hand planted on the floor in a handstand/crawl, a foot in
        a wide split, a hand reaching a ledge) so the arm/leg stays near-straight
        and the contact/target is reachable BY CONSTRUCTION — never an unreachable
        target the IK silently under-reaches (which reads downstream as a floating
        contact). Example — a hand straight down from its own shoulder onto the
        floor:  self.reach_limited_target(ee, self.limb_root_world(ee) * [1,1,0])."""
        desired = np.asarray(desired_world_pos, dtype=float)
        g = self._limb_geom.get(ee_name)
        if not g or g.get("reach", 0.0) <= 1e-6:
            return desired
        root = self.limb_root_world(ee_name)
        d = desired - root
        dist = float(np.linalg.norm(d))
        r = margin * g["reach"]
        if dist <= r or dist < 1e-9:
            return desired
        return root + d * (r / dist)

    # ==========================================================
    # SAFE PLANAR BASE ADVANCE (use this to translate a pitched/rolled base)
    # ==========================================================
    def advance_base_planar(self, forward, lateral, dt, yaw=None):
        """Advance root_pos along the HEADING (yaw-only) world frame — the correct
        way to move the base FORWARD when the torso is PITCHED or ROLLED.

        DO NOT do `vel = quat_to_rotation_matrix(root_quat) @ [v,0,0]` to get a
        forward velocity: root_quat includes the torso PITCH, so for a near-
        horizontal crawl (pitch ~70°) that rotation leaks ~cos(70°) of the speed
        into the (discarded) z-axis and the robot barely advances. "Forward" for
        locomotion means forward ALONG THE GROUND — a function of YAW only, never
        of pitch/roll. This helper does exactly that.

        Args:
            forward, lateral : ground-plane speeds (m/s) in the heading frame.
            dt   : timestep.
            yaw  : heading (rad); defaults to the current root_quat's yaw.
        """
        q = self.root_quat
        if yaw is None:
            yaw = np.arctan2(2.0 * (q[0]*q[3] + q[1]*q[2]),
                             1.0 - 2.0 * (q[2]*q[2] + q[3]*q[3]))
        c, s = np.cos(yaw), np.sin(yaw)
        self.root_pos = self.root_pos.astype(float)
        self.root_pos[0] += (forward * c - lateral * s) * dt
        self.root_pos[1] += (forward * s + lateral * c) * dt

    # ==========================================================
    # END-EFFECTOR MOTION (BODY FRAME)
    # ==========================================================
    def compute_all_ee_positions_body_frame(self, phase):
        """
        Compute BODY-frame targets for all end-effectors.
        """
        return {
            ee: self.compute_ee_position_body_frame(ee, phase)
            for ee in self.ee_names
        }

    def compute_ee_position_body_frame(self, ee_name, phase):
        """
        BODY-frame IK target for a single end-effector (relative to the base).
        Override THIS *or* compute_ee_position_world_frame when the ENDPOINT is
        what matters — a foot's step location, or a hand PLANTED on the floor.

        Default: return None. A None target tells IK to SKIP this end-effector
        this phase; the limb is then shaped by compute_joint_overrides (joint
        angles). Use that for EXPRESSIVE limbs (arms gesturing/punching/waving):
        the joint CONFIGURATION is the gesture, and driving a fingertip endpoint
        through IK would dissolve it into an arbitrary reach that reaches the
        point but does not look like the gesture.
        """
        return None

    def compute_ee_position_world_frame(self, ee_name, phase):
        """
        WORLD-frame target for a single end-effector (ground-anchored). Return
        None to fall back to the body-frame method.

        Override this to author NON-STANDING postures (crawl, handstand, reared
        stance) the easy way: a ground CONTACT target puts the limb's CONTACT POINT
        (sole/palm) ON the surface — i.e. the target z = surface_height + this EE's
        ground_offset (the body-origin-to-sole distance, per the robot spec), NOT
        z = 0 (which would drive the sole a ground_offset BELOW the floor). On the
        flat floor (surface_height 0) that is z = ground_offset; on a tread/ledge it
        is the tread top + ground_offset. A planted contact just keeps its world
        (x, y) fixed — no reach-down arithmetic. Set the base pose (torso height +
        orientation) in world frame via update_base_motion, and IK computes the
        joint angles that connect the base to these world targets.
        """
        return None

    def compute_ee_orientation_world_frame(self, ee_name, phase):
        """
        OPTIONAL world-frame ORIENTATION for an end-effector, as a quaternion
        [w, x, y, z]. Default None.

        You rarely need this. A DECLARED CONTACT is already laid FLAT on whatever
        surface is under it automatically (the framework aligns the sole/palm to that
        surface's normal and leaves the foot's YAW free), so a foot on a stair tread,
        a hand on a vault box and a foot planted on a wall are all handled for you.

        Override it only when the ATTITUDE ITSELF is part of the skill and the
        automatic flat-on-surface rule is not what you want — e.g. gripping an object
        that must be held at a specific angle, or a hand that must meet a surface
        edge-on rather than flat.

        Returning a quaternion pins all THREE rotational DoF of that end-effector, so
        do not use it on an expressive limb: that over-constrains the gesture. For a
        free limb whose shape is the point, use compute_joint_overrides instead.
        """
        return None

    def contacts(self, phase):
        """
        PER-PHASE contact schedule: return a list of end-effector NAMES that are
        load-bearing ground/surface contacts at THIS phase — they get registered
        (seated on their surface) and pinned. Default None -> use the static
        self.support_ees for the whole skill.

        Override for skills whose contact set CHANGES over the trajectory:
          - cartwheel: feet -> (overlap) -> hands -> (overlap) -> feet
          - jump/flip: return [] during the aerial window (base goes ballistic)
        Every end-effector NOT in this list at this phase is free.
        """
        return None

    # ==========================================================
    # SCENE OBJECTS (OPTIONAL) — kinematic object interaction
    # ==========================================================
    def compute_object_poses(self, phase):
        """
        Return a dict {object_name: (pos, quat)} for MOCAP objects in the scene
        that this skill animates (e.g. a box being carried). pos is WORLD-frame
        (3,), quat is [w,x,y,z] (4,). Default: no objects.

        Use this to place an object at rest and while it travels along an authored
        path. To keep a HELD object exactly on the gripping limbs while it is held,
        prefer attachments() below — it removes the guesswork.
        """
        return {}

    def attachments(self, phase):
        """
        Declare which objects are RIGIDLY HELD by which end-effectors this phase:
        {object_name: [ee_name, ...]}. The framework then places the object at the
        MEAN of those end-effectors' ACHIEVED (post-IK / post-FK) world positions —
        so the object coincides with the REAL limbs every frame, with no open-loop
        guessing about where a joint-driven or IK-solved hand ended up. This makes a
        grip correct BY CONSTRUCTION (a two-hand carry: {"box": ["left_hand",
        "right_hand"]} -> box sits at the hands' midpoint).

        Return {} when nothing is held (the object then follows compute_object_poses,
        e.g. resting on the floor before a pick or after a place). Attachments
        OVERRIDE compute_object_poses for the named objects while declared.

        This is OPT-IN and orthogonal to how you drive the holding limbs: you still
        choose their control channel (world-IK to grip, or joint angles to clasp) —
        attachments only guarantees the object tracks wherever those limbs actually go.
        """
        return {}

    # ==========================================================
    # JOINT OVERRIDES (OPTIONAL)
    # ==========================================================
    def compute_joint_overrides(self, phase):
        """
        Return a dict {joint_name: angle_radians} for joints that should be
        set directly instead of being solved by IK (e.g. arms/torso on a
        humanoid). Default: no overrides.
        """
        return {}

    # ==========================================================
    # WORLD TRANSFORM
    # ==========================================================
    def compute_all_ee_positions_world_frame(self, ee_body):
        """
        Transform BODY-frame end-effector positions to WORLD frame.
        """
        return {
            ee: body_to_world_position(pos, self.root_pos, self.root_quat)
            for ee, pos in ee_body.items()
        }
