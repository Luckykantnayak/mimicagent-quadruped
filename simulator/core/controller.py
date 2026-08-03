"""
core/controller.py

Main runner that:
1. Loads the robot spec (end-effectors, joint names, scene path)
2. Uses MotionGenerator to compute root pose and end-effector positions
3. Solves IK to get joint angles
4. Updates simulation
"""
import mujoco
import numpy as np
import os
import sys
from pathlib import Path

from utils.robot_config import load_robot_spec
from utils.reach import base_z_feasible_interval, compute_limb_geometry
from utils.math_utils import quat_to_rotation_matrix
from motion import (get_motion_controller, get_example_controller,
                    get_curated_controller, has_skill_version)
from core.grounding import ContactGrounding
from utils.surfaces import SceneSurfaces


# -------------------------------------------------
# Add project root to PYTHONPATH
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -------------------------------------------------------------------------
# SEAM-BRIDGE constants (GENERAL continuity — NOT morphology/skill tuning).
# A control-channel handoff (joint-override <-> IK) or an IK-target jump snaps
# the joints in a SINGLE frame (a 1-2 rad teleport at the phase boundary) unless
# it is bridged: the joint-override config on one side is not the IK solution on
# the other, and nothing carries state across the seam. These three numbers are
# physical continuity limits applied IDENTICALLY to every robot and skill — the
# same class of general solver default as the IK damping / repulsion gains, never
# a per-skill or per-morphology value.
# -------------------------------------------------------------------------
BRIDGE_WINDOW_S = 0.08     # wall-clock duration of a handoff crossfade (s)
EE_JUMP_M = 0.05           # an IK-target move above this in ONE frame is a teleport
JOINT_JUMP_RAD = 0.20      # an override-angle jump above this in ONE frame is a teleport
OBJ_JUMP_M = 0.05          # an object-pose jump above this in ONE frame is a teleport
GRIP_SLACK_M = 0.12        # how far past an object's own half-extent a holder may still
                           # count as GRIPPING it (about one hand width). Beyond this the
                           # limb never arrived, so `attachments()` must NOT snap the
                           # object to it — see the proximity gate in step().

# -------------------------------------------------------------------------
# BASE-REACH SEAT constants (GENERAL kinematic feasibility — not skill tuning).
# The declared world contacts fix WHERE the limbs must touch; the base must then
# sit where every contact is within its limb's reach, or IK under-reaches and the
# contact FLOATS. On a flat floor the contact grounding already seats the base
# from the contacts; under grounding "off" (raised-surface skills author an
# absolute base + world contacts) that enforcement is switched OFF and delegated
# to the coder, who routinely leaves a contact beyond reach. This seats the base
# ALONG Z onto the reachable set — the same law contact-grounding applies on flat,
# generalized to any surface height. Robot-agnostic (reach comes from the spec).
# -------------------------------------------------------------------------
REACH_GATE = 1.03      # correct only when a contact exceeds 103% of full reach.
                       # A physical standing pose can sit at ~100% extension (the
                       # humanoid home does) — never touch that; only a genuine
                       # over-extension (the float) is corrected.
REACH_TARGET = 0.98    # when correcting, seat the base so the violating contact
                       # comes to 98% of full reach (a slight bend the IK can hit).
REACH_SEAT_CAP = 0.15  # HARD cap on the accumulated base-seat correction (m). Mirrors
                       # the floor-assist's 5 cm cap: the seat is a small honesty
                       # correction, not a free base teleport. Beyond this the authored
                       # base is grossly wrong (the CLIMB dragged the base 0.59 m to hide
                       # an unreachable contact) — STOP correcting, let the contact float
                       # VISIBLY so contact_grounded/support_or_ballistic FAIL and the
                       # repair loop sees it, instead of silently absorbing the error.
REACH_BASE_STEP = 0.01 # (legacy) max change in the base correction between frames (m)
REACH_BASE_RATE = 1.0  # rate limit expressed in m/s (frequency-correct): the per-frame
                       # cap is REACH_BASE_RATE * dt, so the seat eases at the same real
                       # speed at any sim_freq instead of 2 m/s @200 Hz (which sawed the
                       # base up/down and drove the arm IK limit-to-limit on a climb).


def _smoothstep(s):
    s = min(1.0, max(0.0, float(s)))
    return s * s * (3.0 - 2.0 * s)


class MotionControllerRunner:
    """
    Main controller that orchestrates:
    - Motion generation (root pose + end-effector positions)
    - IK solving
    - Simulation updates
    """

    def __init__(self, sim, args):
        self.sim = sim
        self.args = args
        self.model = sim.model
        self.data = sim.data

        # -------------------------------------------------
        # Load robot spec
        # -------------------------------------------------
        robots_dir = getattr(args, "robots_dir", "robots")
        self.spec = load_robot_spec(args.robot, robots_dir=robots_dir)

        print(f"[Controller] Robot: {self.spec.name}")
        print(f"[Controller] End-effectors: {self.spec.ee_names}")

        # -------------------------------------------------
        # Validate: spec joint order must match model qpos[7:] order
        # (the recorder and quality checks depend on this invariant)
        # -------------------------------------------------
        self._check_joint_order()

        # -------------------------------------------------
        # IK Solver — damped least squares by default; IK_BACKEND=mink selects the QP
        # solver with hard position/velocity/collision constraints (see make_ik for the
        # measured trade-off between the two).
        # -------------------------------------------------
        from utils.kinematics import make_ik

        # ONE IK over the UNION roster (end-effectors + declared contact bodies), so
        # ANY declared contact body can be given a world target and PLACED by IK like a
        # foot — not just FK-seated. A union member with NO target this phase is masked
        # OUT (active_mask), so a skill that only drives the 4 EEs solves IDENTICALLY
        # to before (the extra bodies are fully inert — no Jacobian, no residual).
        self.ik = make_ik(
            self.model,
            self.data,
            self.spec.ik_bodies,
            sites=self.spec.ik_sites,
            joint_names=self.spec.actuated_joint_names,
            null_space_gain=self.spec.null_space_gain,
            velocity_ceilings=getattr(self.spec, "joint_speed_ceilings", None),
        )

        # -------------------------------------------------
        # Initial end-effector positions in BODY FRAME
        # -------------------------------------------------
        base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.spec.base_body
        )
        if base_body_id < 0:
            raise ValueError(f"Base body '{self.spec.base_body}' not found in model")

        R_wb = self.data.xmat[base_body_id].reshape(3, 3)
        p_wb = self.data.xpos[base_body_id].copy()

        ee_positions_world = self.ik.get_ee_positions()

        # Body-frame rest position of every UNION member (EEs + contact bodies), so a
        # skill can reference init_ee_pos["right_knee"] and `self.ee_names` inside the
        # skill spans the union — base.step() then queries a world/body target for each
        # contact body too (None -> masked out, FK-only, unchanged for EE-only skills).
        initial_ee_body = {}
        for ee_name, pos_world in zip(self.spec.ik_names, ee_positions_world):
            pos_body = R_wb.T @ (pos_world - p_wb)
            initial_ee_body[ee_name] = pos_body.copy()

        print(f"[Controller] Initial EE positions (body frame):")
        for k, v in initial_ee_body.items():
            print(f"  {k}: {np.round(v, 4)}")

        # -------------------------------------------------
        # Motion Generator — resolution priority:
        #   mode "example"  -> the robot's reference skill (robots/<r>/example.py)
        #   GENERATED       -> skill_output/<robot>/<mode>/<mode>_<N>.py (agent output)
        #   EXEMPLAR        -> robots/<robot>/exemplars/<mode>.py (hand-authored refs)
        #   else            -> example
        # Generated wins so the pipeline always plays its own fresh output; an
        # exemplar is the tracked fallback for a hand-authored reference skill.
        # -------------------------------------------------
        gen_root = os.path.join("skill_output", self.spec.name)
        curated = os.path.join("robots", self.spec.name, "exemplars", f"{args.mode}.py")
        if args.mode == "example":
            print(f"[Controller] Loading reference skill: {self.spec.example_skill}")
            Motion = get_example_controller(self.spec.example_skill)
        elif has_skill_version(gen_root, args.mode):
            Motion = get_motion_controller(args.mode, gaits_dir=gen_root)
        elif os.path.isfile(curated):
            print(f"[Controller] Loading curated skill: {curated}")
            Motion = get_curated_controller(curated)
        else:
            print(f"[Controller] No skill '{args.mode}' found; loading reference: "
                  f"{self.spec.example_skill}")
            Motion = get_example_controller(self.spec.example_skill)

        self.motion_gen = Motion(init_ee_pos=initial_ee_body)

        # --- LIMB REACH GEOMETRY (feeds base.reach_limited_target / limb_root_world)
        # Per EE: its REACH from its own limb root (sum of segment lengths below the
        # last branch point) and that root's body-frame offset from the base at home.
        # A skill can then CONSTRUCT guaranteed-reachable world targets instead of
        # guessing world points the arm/leg can't reach (the #1 cause of "declared
        # contact floats"). Injected as an attribute, keeping the __init__ contract.
        self.motion_gen._limb_geom = self._compute_limb_geom()
        # Same per-EE reach geometry, used by the controller's base-reach seat.
        self._limb_geom = self.motion_gen._limb_geom
        # REACH_SEAT=0 disables the base-reach seat (for before/after measurement).
        self._reach_seat_on = os.environ.get("REACH_SEAT", "1") != "0"
        # FREE-EE reach projection (the dual of the reach-seat): a FREE limb whose world
        # target is beyond reach makes the bridge sawtooth + the IK flip configs (the hand
        # flicker). Project the free target onto the reachable ball so IK gets a feasible,
        # stable, continuous problem and the arm reaches TOWARD the intent. REACH_PROJ=0 to A/B.
        self._reach_proj_on = os.environ.get("REACH_PROJ", "1") != "0"

        # Each skill is self-contained: it owns its base motion (velocity /
        # yaw) inside update_base_motion. The controller does not inject any
        # external velocity command, so nothing overrides the skill.
        root_pos = self.data.qpos[0:3].copy()
        root_quat = self.data.qpos[3:7].copy()
        self.motion_gen.reset(root_pos, root_quat)

        # -------------------------------------------------
        # Timing
        # -------------------------------------------------
        self.dt = 1.0 / args.sim_freq
        self.frame = 0

        # --- SEAM BRIDGE (channel-handoff continuity; see module constants) ---
        # When a limb changes control channel (joint-override <-> IK) or its IK
        # target jumps between frames, crossfade the effective signal from where
        # the limb ACTUALLY IS to the newly-authored value over BRIDGE_WINDOW_S,
        # so IK traces a continuous joint path instead of teleporting. Keys ONLY on
        # "did this limb's control signal change" — no skill/morphology knowledge.
        # SEAM_BRIDGE=0 in the env disables it (for before/after measurement).
        self._bridge_on = (os.environ.get("SEAM_BRIDGE", "1") != "0"
                           and bool(getattr(args, "seam_bridge", True)))
        # per-channel gates (default on; for isolating each bridge's contribution)
        self._bridge_joints = os.environ.get("SEAM_BRIDGE_JOINTS", "1") != "0"
        self._bridge_frames = max(1, int(round(BRIDGE_WINDOW_S * args.sim_freq)))
        self._reset_bridge_state()

        # Resolve the support set ONCE: the skill's declared self.support_ees, else
        # the robot's spec-declared contact feet. This single value drives grounding
        # (below) and is handed to the recorder so the unit tests read the same set.
        default_support = [self.spec.ee_names[i] for i in self.spec.contact_ee_indices]
        declared = getattr(self.motion_gen, "support_ees", None)
        _valid_contacts = self.spec.ee_names + self.spec.contact_body_names
        self.support_ees = [n for n in (declared or default_support)
                            if n in _valid_contacts] or default_support

        # Deterministic contact-grounding (base height from the support feet).
        # "off" for manual playback (examples unchanged); the agent passes
        # "contact" so generated skills can't float.
        grounding_mode = getattr(args, "grounding", "off")
        # QUADRUPED POLICY (no-fork, single choke point): a quadruped is FLAT-terrain,
        # always-contact — it can never author an absolute base or float a foot. This one
        # site covers EVERY entry (run.py, reach_lint, record_video, the agent subprocess),
        # so the policy can't drift. It toggles a MODE VALUE only, not a control path.
        if getattr(self.spec, "kind", "generic") == "quadruped" and grounding_mode != "contact":
            print(f"[grounding] quadruped '{self.spec.name}' → forcing grounding='contact' "
                  f"(was '{grounding_mode}'): flat-terrain, no-float policy.")
            grounding_mode = "contact"
        # `surfaces` is attached below, once the scene-surface query is built (it
        # needs a forward pass to cache world geom poses). Until then the grounding
        # falls back to the floor plane, which is only ever the case during __init__.
        self.grounding = ContactGrounding(
            self.spec, grounding_mode, support_ees=self.support_ees, dt=self.dt,
            # GROUNDING_CLAMP=0 disables the grounding-level surface clamp for A/B.
            # It is partly redundant with the controller's inside()-gated target clamp
            # below; measured identical on the bundled skills, kept because it acts on
            # ee_world BEFORE the reach-seat rather than on the final IK targets.
            clamp_targets=(os.environ.get("GROUNDING_CLAMP", "1") != "0")
        )

        # Robot geoms (exclude the floor / worldbody) for whole-body penetration.
        self._robot_geoms = np.where(self.model.geom_bodyid > 0)[0]
        # End-effector body ids — for recording ACHIEVED (post-IK) EE positions,
        # so the unit tests judge the real pose, not the (maybe unreachable) target.
        self._ee_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            for b in self.spec.ee_bodies
        ]
        # per-EE ground offset (body origin -> sole), for the post-FK re-seat
        self._ee_offsets = np.asarray(self.spec.ee_ground_offsets)

        # Per-EE kinematic-chain joints (base -> EE), for the CONTACT-CHAIN OVERRIDE
        # do-no-harm guard (a contact-chain override that FLOATS its contact off the
        # surface is reverted to the IK solution; see step()).
        self._ee_chain_joints = self._build_ee_chains()
        # SLIDING contacts (from the plan) are EXEMPT from that guard: a moonwalk's feet
        # are override-driven along the floor and legitimately leave their IK target /
        # ride on the toe, so they must NOT be reverted to IK. Read from the skill's plan
        # (lives beside the skill); empty for standalone playback.
        self._sliding_contacts = set()
        try:
            import glob as _glob, json as _json
            _plans = sorted(_glob.glob(os.path.join("skill_output", self.spec.name,
                                                    args.mode, "motion_plan_*.json")))
            if _plans:
                _p = _json.load(open(_plans[-1]))
                self._sliding_contacts = set(_p.get("sliding_contacts")
                    or (_p.get("motion_plan") or {}).get("sliding_contacts") or [])
        except Exception:
            pass

        # UNION CONTACT SET = the IK end-effectors + any spec-declared extra
        # contact bodies (knees/shins/...). Grounding and every contact check
        # operate over this union, so ANY declared body can bear weight on a
        # surface; IK still drives ONLY the end-effectors. The extra bodies have
        # no IK target — their position comes from FK and they are seated post-IK.
        self._contact_names = list(self.spec.ee_names) + list(self.spec.contact_body_names)
        self._contact_body_ids = list(self._ee_body_ids) + [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            for b in self.spec.contact_body_bodies]
        self._contact_offsets = np.concatenate([
            self._ee_offsets,
            np.asarray(self.spec.contact_body_offsets, dtype=float)]) \
            if self.spec.contact_body_offsets else self._ee_offsets
        self.contact_names = self._contact_names   # exposed for the recorder

        # scene MOCAP objects the skill may animate (name -> mocap index) + each
        # object's static half-extent (largest half-size), so the object-grip check
        # can scale its hold distance to the OBJECT size, not an absolute number.
        self._mocap_ids = {}
        self._object_half_extent = {}   # cache for _grip_range()
        self.object_extents = {}
        for bid in range(self.model.nbody):
            mid = int(self.model.body_mocapid[bid])
            if mid >= 0:
                nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
                self._mocap_ids[nm] = mid
                gs = [g for g in range(self.model.ngeom)
                      if self.model.geom_bodyid[g] == bid]
                if gs:
                    self.object_extents[nm] = float(max(
                        float(self.model.geom_size[g].max()) for g in gs))

        # --- FUNDAMENTAL collision checks (surface + self penetration) --------
        # Robot geoms = geoms on the robot's OWN bodies (exclude the worldbody
        # floor/objects AND any mocap object such as a carried box).
        bodyid = self.model.geom_bodyid
        body_mocap = self.model.body_mocapid
        self._robot_geom_set = set(
            g for g in range(self.model.ngeom)
            if bodyid[g] > 0 and body_mocap[bodyid[g]] < 0
        )
        # Representative robot collision-limb RADIUS (median size[0] over round
        # collision geoms — capsule/sphere/cylinder), so the self-collision check can
        # scale its gross-overlap threshold to the robot's limb THICKNESS instead of
        # an absolute human number (0.08 m never fires on a thin-limbed quadruped).
        _round = (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE,
                  mujoco.mjtGeom.mjGEOM_CYLINDER)
        _radii = [float(self.model.geom_size[g, 0]) for g in self._robot_geom_set
                  if self.model.geom_type[g] in _round and self.model.geom_size[g, 0] > 1e-6]
        self.limb_radius = float(np.median(_radii)) if _radii else 0.04
        # Non-adjacent robot geom pairs for SELF-collision: skip same-body and
        # parent-child bodies (they meet at their shared joint by construction).
        # The model filters self-collisions, so mj_collision cannot see them —
        # mj_geomDistance is used per pair (flag-independent), rbound-pruned.
        parent = self.model.body_parentid
        rg = sorted(self._robot_geom_set)
        self._self_pairs = [
            (int(rg[i]), int(rg[j]))
            for i in range(len(rg)) for j in range(i + 1, len(rg))
            if not (bodyid[rg[i]] == bodyid[rg[j]]
                    or parent[bodyid[rg[i]]] == bodyid[rg[j]]
                    or parent[bodyid[rg[j]]] == bodyid[rg[i]])
        ]
        self._rbound = self.model.geom_rbound
        self._fromto = np.zeros(6)
        self._robot_geom_list = sorted(self._robot_geom_set)
        # MOCAP object geoms (a carried box) — for the object-vs-robot penetration
        # measurement (a held object must not clip THROUGH the body; the holder
        # hands legitimately touch it and are excluded per-frame).
        self._mocap_object_geoms = [g for g in range(self.model.ngeom)
                                    if body_mocap[bodyid[g]] >= 0]
        # Static scene BOX geoms (stair treads, slabs — NOT mocap manipulands) for
        # CENTER-based surface penetration: a robot geom's CENTER inside a box (or
        # below the floor plane z=0) is a real penetration; a contact merely resting
        # ON a surface keeps its center outside, so this is false-positive-free
        # (unlike surface-overlap). A carried/mocap box is a MANIPULAND, not a
        # surface — the hands are meant to grip it — so it is excluded here and
        # judged instead by the object checks (grip / continuity / at-rest).
        self._scene_boxes = [g for g in range(self.model.ngeom)
                             if g not in self._robot_geom_set
                             and body_mocap[bodyid[g]] < 0
                             and self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
        # THE scene-surface query (utils/surfaces.py) — exact for every primitive at
        # any orientation, and the single answer to "what is beneath this point?"
        # used by grounding, the target clamp, the reach seat, the floor assist and
        # the recorded clearance. Built after a forward pass so it caches WORLD geom
        # poses (a geom nested in an offset body has geom_pos != its world position).
        mujoco.mj_forward(self.model, self.data)
        self.surfaces = SceneSurfaces(self.model, self.data, self._robot_geom_list)
        self.grounding.surfaces = self.surfaces
        # Static-box AABBs (WORLD center, half-size) precomputed once — static geoms
        # never move. Used by the IK geom-repulsion to keep limbs OUT of every surface
        # (floor + boxes), not just the floor. Read from data.geom_xpos rather than
        # model.geom_pos so a box declared inside an offset <body> is placed correctly.
        self._box_aabb = [(np.asarray(self.data.geom_xpos[g], dtype=float),
                           np.asarray(self.model.geom_size[g], dtype=float))
                          for g in self._scene_boxes]
        # For SURFACE CLEARANCE: the main geom on each end-effector body, and all
        # scene surfaces (floor plane + objects). Clearance = signed distance from
        # an EE to the NEAREST surface (positive = clear, ~0 = resting on it),
        # direction-agnostic (works for a tread top OR a vertical slab face).
        self._ee_geoms = []
        for bid in self._ee_body_ids:
            gs = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == bid]
            self._ee_geoms.append(gs[0] if gs else -1)
        self._scene_surface_geoms = [g for g in range(self.model.ngeom)
                                     if g not in self._robot_geom_set]

    # -------------------------------------------------
    # Limb reach geometry (for base.reach_limited_target)
    # -------------------------------------------------
    def _compute_limb_geom(self):
        """Per-EE {reach, root_offset} at the home pose (shared geometry — same
        definition the pre-sim lint and the onboarding singularity check use)."""
        # EE-ONLY (NOT the union): the reach model (base -> limb TIP at full extension)
        # is end-effector-centric and produces a garbage "reach" for a MID-LIMB contact
        # body (a shin/forearm), which made the reach-seat mis-seat the base for a knee
        # contact. Contact-body IK places those bodies directly (independent of this
        # geometry); the reach-seat/reach-proj stay EE-only. A mid-limb contact that
        # under-reaches simply floats -> contact_grounded catches it (the honest signal).
        mujoco.mj_kinematics(self.model, self.data)
        return compute_limb_geometry(self.model, self.data, self.spec.base_body,
                                     self.spec.ee_names, self.spec.ee_bodies)

    def _build_ee_chains(self):
        """{ee_name: set of ACTUATED joint names on its kinematic chain from the base}.
        Walk body_parentid from each EE body up to the base body, collecting the
        actuated joints along the way — these are the joints IK moves to place that EE.
        Morphology-agnostic (topology from the model)."""
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.spec.base_body)
        act = set(self.spec.joint_names)
        chains = {}
        union_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
                     for b in self.spec.ik_bodies]
        for name, bid in zip(self.spec.ik_names, union_ids):
            joints, b = set(), bid
            while b > 0 and b != base_id:
                adr, num = int(self.model.body_jntadr[b]), int(self.model.body_jntnum[b])
                for j in range(adr, adr + num):
                    jn = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                    if jn in act:
                        joints.add(jn)
                b = int(self.model.body_parentid[b])
            chains[name] = joints
        return chains

    # -------------------------------------------------
    # Spec / model consistency
    # -------------------------------------------------
    def _check_joint_order(self):
        """Assert spec.joints matches the model's qpos[7:] joint order."""
        model_joints = []
        for jid in range(self.model.njnt):
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            model_joints.append(
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            )
        if model_joints != self.spec.joint_names:
            raise ValueError(
                f"Robot spec joint order does not match model qpos order.\n"
                f"  model: {model_joints}\n"
                f"  spec : {self.spec.joint_names}"
            )

    # -------------------------------------------------
    # Commands
    # -------------------------------------------------
    def set_velocity_command(self, vx=0.0, vy=0.0, vz=0.0):
        self.motion_gen.set_velocity_command(vx, vy, vz)

    def set_angular_velocity_command(self, roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0):
        self.motion_gen.set_angular_velocity_command(roll_rate, pitch_rate, yaw_rate)

    def _reset_bridge_state(self):
        """Clear all seam-bridge memory. Called when the phase clock is reset so the
        first recorded frame snaps to the authored pose rather than bridging from a
        stale warmup pose (which would blur the first ~window frames of the clip)."""
        self._prev_ee_active = {}      # ee_name -> was it IK-active last frame
        self._prev_ee_achieved = {}    # ee_name -> achieved world pos last frame (seed)
        self._prev_ee_target = {}      # ee_name -> last authored/effective TARGET (jump reference)
        self._ee_seam = {}             # ee index -> [seed(3), frames_left]
        self._prev_override = {}       # joint -> AUTHORED override value last frame
        self._prev_override_eff = {}   # joint -> APPLIED (bridged) override last frame (seed)
        self._prev_joint_q = None      # qpos[7:] achieved last frame (fallback seed)

        # --- PHYSICAL SLEW LIMIT ---------------------------------------------
        # Per-joint max change per frame = (torque-derived rad/s ceiling) * dt, in
        # MODEL qpos[7:] order so it can be applied straight to the joint vector.
        # PASSIVE joints (a wheel) get no cap — they are not actuated, so a
        # torque-derived speed ceiling is meaningless for them.
        # Absent dynamics data (a model with no torque source) -> feature self-disables,
        # exactly like joint_velocity_envelope does. RATE_LIMIT=0 disables it for A/B.
        self._rate_limit_on = os.environ.get("RATE_LIMIT", "1") != "0"
        # EE_ORIENT=0 disables the rotational IK task (contact-flat + skill-authored).
        self._ee_orient_on = os.environ.get("EE_ORIENT", "1") != "0"
        # Which roster members may be AUTO-oriented flat onto their surface. The
        # auto-align target is "sit the way you sit at HOME", which is only meaningful
        # for a member that IS flat on the ground at the home pose — i.e. the spec's
        # ground-contact end-effectors (feet). It is WRONG for everything else:
        # - a mid-limb contact body (shin/forearm) stands VERTICAL at home but lies
        #   FLAT when kneeling or planking, so aligning its home-up axis to the floor
        #   fights the pose (it broke a crawl outright: 1 -> 5 failing checks);
        # - a `contact: false` EE (the hands) hangs at the side at home, so the same
        #   assumption is meaningless for a handstand's palms.
        # Those limbs are still free to declare an explicit orientation via
        # compute_ee_orientation_world_frame — this only governs the DEFAULT.
        # (The reach-seat and reach-projection are EE-only for the same class of
        # reason: a whole-limb model does not transfer to a mid-limb body.)
        self._auto_orient_idx = set(self.spec.contact_ee_indices)
        ceilings = getattr(self.spec, "joint_speed_ceilings", None) or {}
        if ceilings:
            caps = np.full(len(self.spec.joint_names), np.inf)
            for i, jn in enumerate(self.spec.joint_names):
                if jn in ceilings:
                    caps[i] = float(ceilings[jn]) * self.dt
            self._joint_rate_cap = caps
            n_capped = int(np.isfinite(caps).sum())
            print(f"[Controller] slew limit: {n_capped}/{len(caps)} joints capped "
                  f"at their dynamics ceiling x dt"
                  + ("" if self._rate_limit_on else "  [DISABLED via RATE_LIMIT=0]"))
        else:
            self._joint_rate_cap = None
            print("[Controller] slew limit: OFF (no dynamics data for this robot)")
        self._joint_seam = {}          # joint -> [seed_angle, frames_left]
        self._prev_object_pose = {}    # object -> world pos last frame (seed)
        self._object_seam = {}         # object -> [seed(3), frames_left]
        self._reach_delta = None       # last applied base-reach z correction (rate-limit state)

    def reset_bridge(self):
        """Public hook: reset seam-bridge memory (run.py calls this at record start)."""
        self._reset_bridge_state()

    def reset(self):
        root_pos = self.data.qpos[0:3].copy()
        root_quat = self.data.qpos[3:7].copy()
        self.motion_gen.reset(root_pos, root_quat)
        self.frame = 0
        self._reset_bridge_state()

    def _measure_surface_pen(self, localize=False):
        """Deepest a robot geom CENTER goes below the floor (z<0) OR inside any
        scene box (m). A resting contact keeps its center outside -> ~0; only a part
        driven THROUGH a surface registers. Used both for the recorded metric and
        the joint-bridge do-no-harm guard.

        localize=True also returns (body_name, surface_name) of the deepest geom —
        a blind scalar told the repair loop THAT something clipped but not WHAT or
        WHERE (the CLIMB shin-into-slab was invisible); the localized readout names
        the offending link + surface so the diagnosis is actionable."""
        xp = self.data.geom_xpos
        sp = 0.0
        arg_geom = -1; arg_surf = None
        for rg in self._robot_geom_list:
            d, gid = self.surfaces.inside(xp[rg])
            if d > sp:
                sp = d
                arg_geom = rg
                arg_surf = (self.surfaces.geom_name(gid)
                            or (f"geom#{gid}" if gid is not None else None))
        if not localize:
            return sp
        body = None
        if arg_geom >= 0:
            bid = int(self.model.geom_bodyid[arg_geom])
            body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
        return sp, body, arg_surf

    def _geom_name(self, g):
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(g))

    def _object_here(self, obj, authored):
        """Where object `obj` PHYSICALLY was, as the reference the grip is judged
        against: LAST frame's final pose.

        Deliberately NOT this frame's authored pose. A skill that both animates the
        object (`compute_object_poses`) and declares it held (`attachments`) drives
        the object toward the hands itself, so comparing the hands against that target
        is trivially self-consistent — it would pass however far the limbs actually
        are (measured: the carry-box gate never fired, because the authored box was
        already at the hands). Last frame's ACHIEVED pose is where the thing really
        was before anyone claimed to grab it, which is the only honest datum."""
        prev = getattr(self, "_prev_object_pose", {}).get(obj)
        if prev is not None:
            return np.asarray(prev, dtype=float)[:3]
        if authored is not None:
            return np.asarray(authored[0], dtype=float)
        mid = self._mocap_ids.get(obj)
        if mid is not None:
            return np.asarray(self.data.mocap_pos[mid], dtype=float)
        return None

    def _grip_range(self, obj):
        """How close a holder must get before the object counts as GRIPPED: the
        object's own half-extent (you grip a face, not the centre) plus one hand
        width of slack. Derived from the model, so a small tool and a large crate
        each get a fair bound instead of one hardcoded number."""
        r = self._object_half_extent.get(obj)
        if r is None:
            bid = self.model.body_mocapid
            r = 0.0
            for g in range(self.model.ngeom):
                b = int(self.model.geom_bodyid[g])
                if bid[b] >= 0 and mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, b) == obj:
                    r = max(r, float(np.max(np.abs(self.model.geom_size[g]))))
            self._object_half_extent[obj] = r
        return float(r) + GRIP_SLACK_M

    def _reach_seat_base(self, root_pos, root_quat, ee_world, active):
        """Seat the base ALONG Z so every declared world contact this phase lies
        within its limb's reach. The raised-surface generalization of the contact
        grounding that flat skills already get: the world contacts are the datum
        (a foot on a tread), the authored base is the guess — when the guess leaves
        a contact beyond leg reach the IK under-reaches and the contact floats, so
        move the BASE (not the contact) minimally onto the reachable band.

        Both directions, base-z only. GATED so a physical standing pose (up to full
        extension) is never corrected, and RATE-LIMITED so a contact-schedule change
        eases the base instead of popping it. Returns the (maybe z-shifted) root_pos.
        A contact beyond HORIZONTAL reach can't be fixed by z — it is reported (needs
        a base xy/orientation move, the later joint solve) and skipped here."""
        geom = self._limb_geom
        names = active if active is not None else self.support_ees
        names = [n for n in (names or [])
                 if ee_world.get(n) is not None and n in geom
                 and geom[n].get("reach", 0.0) > 1e-6]
        delta_target = 0.0
        if names:
            base_z = float(root_pos[2])
            # Reachability against the SURFACE-CLAMPED contact positions, NOT the raw
            # authored targets: the coder routinely authors a contact a few cm INTO its
            # surface (a foot at the tread's z instead of tread+ground_offset), and the
            # target clamp fixes that LATER in step() — but the seat runs BEFORE the clamp,
            # so on the raw (too-deep) targets it saw a permanent over-reach and sagged the
            # base every frame (the CLIMB penetration cascade). Clamp here so the seat
            # reasons about where the contact will ACTUALLY rest.
            contacts_world = {}
            for n in names:
                p = np.asarray(ee_world[n], dtype=float).copy()
                off = float(self._contact_offsets[self._contact_names.index(n)])
                surf = self.surfaces.support_z(float(p[0]), float(p[1]),
                                               float(p[2]), default=0.0)
                p[2] = max(float(p[2]), surf + off)
                contacts_world[n] = p
            lo, hi, over, horiz = base_z_feasible_interval(
                root_pos, root_quat, contacts_world, geom, REACH_TARGET)
            if over > REACH_GATE:
                if np.isfinite(lo) and np.isfinite(hi):
                    base_z_new = (min(max(base_z, lo), hi) if lo <= hi
                                  else 0.5 * (lo + hi))              # incompatible -> compromise
                    delta_target = base_z_new - base_z
                    if self.frame % 30 == 0:
                        print(f"[reach-seat] base z {base_z:.3f}->{base_z + delta_target:.3f} "
                              f"(Δ{delta_target:+.3f}) so contacts {names} reach "
                              f"(over-reach {over:.2f}×)"
                              + ("" if lo <= hi else " [incompatible contacts -> midpoint]"))
                elif self.frame % 60 == 0:
                    print(f"[reach-seat] ⚠️ contact(s) {horiz} beyond HORIZONTAL reach "
                          f"(over-reach {over:.2f}×) — needs a base xy/orientation move; "
                          f"z-seat cannot fix this. The contact will float.")
        # rate-limit the correction (snap once after reset, then ease in/out) — in m/s
        # so it eases at the same real speed at any sim_freq (no base sawing).
        if self._reach_delta is None:
            delta = delta_target
        else:
            s = REACH_BASE_RATE * self.dt
            delta = min(max(delta_target, self._reach_delta - s), self._reach_delta + s)
        # HARD cap: never let the seat absorb more than REACH_SEAT_CAP of authored-base
        # error. Beyond it the base is grossly mis-authored — clamp and let the residual
        # float show up in the checks (do-no-harm: a genuine small seat is unaffected).
        if abs(delta) > REACH_SEAT_CAP:
            if self.frame % 60 == 0:
                print(f"[reach-seat] ⚠️ required correction Δ{delta:+.3f} exceeds cap "
                      f"±{REACH_SEAT_CAP:.2f} m — the authored base is grossly wrong; "
                      f"clamping and letting the contact float VISIBLY (check will fail).")
            delta = float(np.clip(delta, -REACH_SEAT_CAP, REACH_SEAT_CAP))
        self._reach_delta = delta
        if abs(delta) > 1e-5:
            root_pos = root_pos.copy(); root_pos[2] = float(root_pos[2]) + delta
            # Move BODY-frame free-limb targets WITH the base: they were converted to
            # world with the UN-seated root (in base.step), so without this they'd sit
            # `delta` off their authored body-relative pose — the distortion the seat
            # otherwise causes, worst exactly when the seat moves most. A DECLARED
            # CONTACT is the datum (stays fixed); a WORLD-frame target is absolute (a
            # grip / a ledge point — stays fixed). Only body-frame free limbs shift.
            contact_now = set(active) if active is not None else set(self.support_ees)
            ee_frame = getattr(self, "_ee_frame", None) or {}
            for n, fr in ee_frame.items():
                if fr == "body" and n not in contact_now and ee_world.get(n) is not None:
                    ee_world[n] = ee_world[n] + np.array([0.0, 0.0, delta])
        return root_pos

    # -------------------------------------------------
    # Main step
    # -------------------------------------------------
    def step(self):
        motion_state = self.motion_gen.step(self.dt)

        root_pos  = motion_state["root_pos"]
        root_quat = motion_state["root_quat"]
        ee_positions_world = motion_state["ee_positions_world"]
        # authoring frame per EE (world|body|None) — the reach-seat uses it to shift
        # only body-frame free targets when it moves the base.
        self._ee_frame = motion_state.get("ee_frame") or {}

        # Ground/register the base to the PER-PHASE contacts (falls back to the
        # static support set when the skill declares no per-phase schedule).
        active = motion_state.get("contacts")
        root_pos, ee_positions_world = self.grounding.apply(
            root_pos, root_quat, ee_positions_world, active=active
        )

        # BASE-REACH SEAT (BOTH grounding modes): move the base along z so the declared
        # world contacts are within limb reach. Runs BEFORE IK so IK solves against a
        # base from which the contacts are reachable (else it under-reaches and the
        # contact floats). Registration is reach-INVARIANT (grounding.apply shifts base
        # AND targets together), so a contact authored beyond reach survives contact-mode
        # grounding unchanged — a handstand's hands authored 0.4 m below an over-high
        # pelvis float exactly as they do under "off". The internal REACH_GATE (1.03×)
        # keeps a normal standing/walking pose untouched, so running it in contact mode
        # is a no-op for ordinary skills and only fires on a genuine over-extension.
        if self._reach_seat_on:
            root_pos = self._reach_seat_base(
                root_pos, root_quat, ee_positions_world, active
            )

        # Build IK target array in UNION order (EEs + contact bodies). A member with a
        # None target this phase (a joint-driven expressive limb, or a contact body the
        # skill isn't placing) is masked OUT so IK leaves it to FK / joint-overrides.
        # A contact body given a world target here is PLACED by IK exactly like a foot.
        ik_names = self._contact_names
        ee_targets = np.zeros((len(ik_names), 3))
        ik_mask = np.zeros(len(ik_names), dtype=bool)
        for i, ee_name in enumerate(ik_names):
            t = ee_positions_world.get(ee_name)
            if t is not None:
                ee_targets[i] = t
                ik_mask[i] = True

        # --- SEAM BRIDGE (EE IK targets) ------------------------------------
        # If an EE just became IK-driven (was joint/skip last frame) OR its target
        # jumped, crossfade the effective target from where the limb ACTUALLY IS
        # (its achieved position last frame) to the authored target over the bridge
        # window, so IK traces a continuous joint path instead of snapping. Runs
        # BEFORE the surface clamp below, so a bridged intermediate target is
        # re-projected out of any surface (it never cuts a limb through a slab).
        #
        # EXCEPTION — an EE that is a DECLARED CONTACT this frame SNAPS to its
        # target (no bridge): a landing must rest on its surface crisply, and
        # bridging it would slide/penetrate the contact during the window (a swing
        # that lands far from its foothold is a skill-authoring bug the repair loop
        # must fix, not one the framework should hide by dragging the contact).
        # RELEASES (contact -> free) and free-limb switches ARE bridged — that is
        # where the large upper-body pops live (a hand letting go of a grip).
        # A planted contact that stays planted does not trigger anyway (no jump).
        contact_now = set(active) if active is not None else set(self.support_ees)

        # FREE-EE REACH PROJECTION (dual of the reach-seat; runs AFTER the seat so it uses
        # the seated base). For every IK-active EE that is NOT a declared contact, project
        # its world target onto the limb's reachable ball — a free target beyond reach
        # otherwise makes IK flip configurations frame-to-frame (the ~1 m hand flicker).
        # Same gate as the seat (1.03x) so a gesture at full extension is untouched;
        # contacts are excluded (their target is the datum — the BASE moves for them).
        reach_proj = np.zeros(len(self._contact_names))
        if self._reach_proj_on:
            Rb = quat_to_rotation_matrix(root_quat)
            for i in range(len(ik_names)):
                name = ik_names[i]
                if not ik_mask[i] or name in contact_now:
                    continue
                g = self._limb_geom.get(name)
                if not g or g.get("reach", 0.0) <= 1e-6:
                    continue
                limb_root = np.asarray(root_pos, dtype=float) + Rb @ g["root_offset"]
                d = ee_targets[i] - limb_root
                dist = float(np.linalg.norm(d))
                if dist > REACH_GATE * g["reach"] and dist > 1e-9:
                    rr = REACH_TARGET * g["reach"]
                    ee_targets[i] = limb_root + d * (rr / dist)
                    reach_proj[i] = dist - rr

        authored_now = ee_targets.copy()   # pre-bridge targets — the jump reference for next frame
        if self._bridge_on:
            for i in range(len(ik_names)):
                name = ik_names[i]
                if not ik_mask[i] or name in contact_now:
                    self._ee_seam.pop(i, None)
                    continue
                seed_prev = self._prev_ee_achieved.get(name)   # crossfade SEED (where the limb ACTUALLY is)
                if seed_prev is None:
                    continue
                seam = self._ee_seam.get(i)
                if seam is None:
                    became_active = self._prev_ee_active.get(name) is False
                    # Arm on a TARGET jump (a real signal discontinuity), keyed on the
                    # previous authored TARGET — NOT on tracking error vs prev_achieved. With
                    # a chronically-unreachable free target, prev_achieved is always ~1 m off,
                    # so keying on it re-armed EVERY frame → a 12.5 Hz flicker (the sawtooth
                    # between the bridge and an out-of-reach hand). A steady tracking error is
                    # not a discontinuity; only a change in the authored signal is.
                    prev_t = self._prev_ee_target.get(name)
                    jumped = (prev_t is not None
                              and float(np.linalg.norm(ee_targets[i] - prev_t)) > EE_JUMP_M)
                    if became_active or jumped:
                        seam = [seed_prev.copy(), self._bridge_frames]
                        self._ee_seam[i] = seam
                if seam is not None:
                    seam[1] -= 1
                    w = _smoothstep(1.0 - seam[1] / self._bridge_frames)
                    ee_targets[i] = (1.0 - w) * seam[0] + w * ee_targets[i]
                    if seam[1] <= 0:
                        self._ee_seam.pop(i, None)

        # SURFACE-AWARE TARGET CLAMP: no IK target may be aimed INTO a surface —
        # not the floor, not a step, not a slab. For each active target, keep its
        # contact point (sole/palm = target_z - ground_offset) at or above the
        # floor AND, if it lies over a static box's footprint and below that box's
        # top, lift it ONTO the top (stand ON the surface, never through it). A
        # contact already resting on a surface is a boundary point -> unchanged
        # (no false correction). Interventions are COUNTED and surfaced in the
        # recorded trajectory, so a structurally-absorbed illegal target still
        # shows up to the repair loop instead of hiding the coder's bug.
        clamp_events = 0
        clamp_mag = 0.0                                       # deepest target lifted out of a surface (m)
        for i in np.where(ik_mask)[0]:
            off = float(self._contact_offsets[i])
            px, py, pz = float(ee_targets[i, 0]), float(ee_targets[i, 1]), float(ee_targets[i, 2])
            # Two separate corrections, because they answer different questions:
            #   (1) never below the surface this target sits OVER (floor / tread), and
            #   (2) if the contact point is INSIDE a solid, lift it onto that solid's
            #       top — the "stand ON the surface, never through it" rule. Gating (2)
            #       on inside() matters: a limb in free air BENEATH an overhang (a foot
            #       crawling under a table) also has a surface above it and must not be
            #       lifted onto it.
            sole = pz - off
            surf = self.surfaces.support_z(px, py, sole + 1e-4, default=0.0)
            new_z = max(pz, surf + off)
            depth, _gid = self.surfaces.inside((px, py, sole))
            if depth > 1e-6:
                top = self.surfaces.surface_above(px, py, sole)
                if top is not None:
                    new_z = max(new_z, top + off)
            if new_z > pz + 1e-4:
                clamp_events += 1
                clamp_mag = max(clamp_mag, new_z - pz)
            ee_targets[i, 2] = new_z

        # The skill's joint_overrides for THIS phase also become the IK null-space
        # REFERENCE, so the redundant DoFs are regularized toward the skill's
        # intended posture (legs straight UP in a handstand) instead of always
        # toward STANDING — the wrong prior for a far-from-standing pose.
        # --- EE ORIENTATION TARGETS ------------------------------------------
        # Position alone gives IK 3 constraints against ~7 DoF per limb, leaving the
        # TWIST axes (hip_z, shoulder_y, abdomen_z) essentially unobservable — which is
        # what let them chatter on inverted/rotated skills. A rotational task observes
        # them, and is also simply what a contact physically needs.
        #
        #   1. A skill's explicit orientation wins (full 3-DoF).
        #   2. Otherwise a DECLARED CONTACT defaults to SOLE-FLAT-ON-ITS-SURFACE:
        #      align the EE's home-'up' axis to the surface normal under it. This is an
        #      ALIGNMENT, not a full orientation, so the foot's YAW stays free — the
        #      skill still chooses which way the foot points.
        #   3. A FREE limb gets nothing. Forcing orientation on a gesture would
        #      dissolve exactly the shape the joint channel exists to preserve.
        ori_targets = {}
        if self._ee_orient_on:
            authored_ori = motion_state.get("ee_orientations") or {}
            for i, name in enumerate(ik_names):
                if not ik_mask[i]:
                    continue
                q = authored_ori.get(name)
                if q is not None:
                    ori_targets[i] = ("quat", np.asarray(q, dtype=float))
                elif name in contact_now and i in self._auto_orient_idx:
                    off = float(self._contact_offsets[i])
                    p = ee_targets[i]
                    hit = self.surfaces.support_under(float(p[0]), float(p[1]),
                                                      float(p[2]) - off + 0.05)
                    if hit is not None:
                        ori_targets[i] = ("align", hit[1])

        joint_overrides = motion_state.get("joint_overrides", {})
        self.ik.calculate(ee_targets, active_mask=ik_mask, debug=(self.frame % 60 == 0),
                          box_aabb=self._box_aabb, robot_geoms=self._robot_geom_list,
                          ref_overrides=joint_overrides, ori_targets=ori_targets)

        # Apply the skill's joint overrides — SET directly, but crossfade through
        # the seam bridge when a joint newly enters the override set far from where
        # it is, or its override value jumps between frames (the arm-gesture pops at
        # block boundaries). IK handles a joint LEAVING the override set via its
        # warm-start, so only the entering/jumping direction is bridged here.
        # ONE-CHANNEL-PER-LIMB GUARD: joints on the chain of ANY IK-ACTIVE EE (a limb with
        # a world/body target this frame — a planted contact OR a SWINGING foot) are IK-OWNED
        # — SKIP a compute_joint_overrides value on them. Applying an override on an IK-solved
        # chain OVERWRITES the solution post-solve and drives the limb open-loop (a swing foot
        # posed by a hip/knee sinusoid punched 35 cm THROUGH the climb slab — the guard used to
        # protect only DECLARED-CONTACT chains, so a swing was unprotected). The override still
        # biases the null-space via ref_overrides, so the intended posture shapes redundancy —
        # style an IK swing through that bias, never by fighting the solve. SLIDING contacts
        # (a moonwalk's feet) stay EXEMPT — their legs are override-driven by design. A
        # joint_posture limb returns None -> not IK-active -> its overrides apply, untouched.
        protected = set()
        if os.environ.get("CHANNEL_GUARD", "1") != "0":
            for i, name in enumerate(ik_names):
                if ik_mask[i] and name not in self._sliding_contacts:
                    protected |= self._ee_chain_joints.get(name, set())
        override_conflicts = 0

        applied_override = {}
        applied_eff = {}
        for joint_name, angle in joint_overrides.items():
            if joint_name not in self.spec.joint_names:
                continue
            if joint_name in protected:
                override_conflicts += 1        # IK owns this joint (its EE is a planted contact)
                continue
            idx = self.spec.joint_names.index(joint_name)
            eff = float(angle)
            if self._bridge_on and self._bridge_joints and self._prev_joint_q is not None:
                # Seed from the last APPLIED override (on the override manifold),
                # NOT the resolved achieved angle — resolve_penetration may have
                # pushed the joint off-manifold, and blending from there back to the
                # authored value can route a limb through the floor (a handstand
                # leg). For a joint NEWLY entering the override set, fall back to the
                # achieved angle (the correct IK->override handoff seed).
                prev_eff = self._prev_override_eff.get(joint_name)
                seed_prev = (prev_eff if prev_eff is not None
                             else float(self._prev_joint_q[idx]))
                seam = self._joint_seam.get(joint_name)
                if seam is None:
                    prev_auth = self._prev_override.get(joint_name)
                    ref = seed_prev if prev_auth is None else prev_auth
                    if abs(float(angle) - ref) > JOINT_JUMP_RAD:
                        seam = [seed_prev, self._bridge_frames]
                        self._joint_seam[joint_name] = seam
                if seam is not None:
                    seam[1] -= 1
                    w = _smoothstep(1.0 - seam[1] / self._bridge_frames)
                    eff = (1.0 - w) * seam[0] + w * float(angle)
                    if seam[1] <= 0:
                        self._joint_seam.pop(joint_name, None)
            self.data.qpos[7 + idx] = eff
            applied_override[joint_name] = float(angle)
            applied_eff[joint_name] = eff
        # drop seams for joints no longer overridden (IK owns them again)
        for j in list(self._joint_seam.keys()):
            if j not in joint_overrides:
                self._joint_seam.pop(j, None)

        self.data.qpos[0:3] = root_pos
        self.data.qpos[3:7] = root_quat

        # Position any scene mocap objects the skill animates (a carried box).
        for name, pose in motion_state.get("object_poses", {}).items():
            mid = self._mocap_ids.get(name)
            if mid is not None:
                pos, quat = pose
                self.data.mocap_pos[mid] = np.asarray(pos, dtype=float)
                self.data.mocap_quat[mid] = np.asarray(quat, dtype=float)

        # Forward-kinematic the final pose and record the robot's lowest point
        # (any geom, size-aware) so the unit test can catch head/knee/torso —
        # not just end-effectors — going through the floor.
        mujoco.mj_kinematics(self.model, self.data)

        # --- POST-FK CONTACT RE-SEAT ---------------------------------------
        # The realized pose (after IK + joint overrides) can leave the active
        # contacts off the floor for two reasons the pre-IK grounding cannot
        # fix: (1) IK UNDER-REACH in an extreme stance (a wide/low horse stance
        # the legs cannot quite reach), and (2) a JOINT-DRIVEN leg whose foot
        # position comes from FK, so grounding.apply had no commanded target to
        # seat. Drop/lift the WHOLE body so the lowest active contact rests on
        # the floor. Skipped in flight (contacts == []), preserving ballistics.
        active_names = active if active is not None else self.support_ees
        if self.grounding.mode != "off" and active_names:
            idxs = [i for i, n in enumerate(self._contact_names) if n in active_names]
            if idxs:
                xp = self.data.xpos[self._contact_body_ids]
                low = min(xp[i][2] - self._contact_offsets[i] for i in idxs)
                # Only LIFT to correct penetration; never DROP the base to chase a
                # contact that floats above the floor — an intentionally-reared
                # pose (a quadruped up on its hind legs) legitimately holds some
                # contacts high, and dropping to them would drive the body under.
                if low < -1e-3:
                    root_pos = root_pos.copy(); root_pos[2] -= low   # -low > 0 -> lift
                    self.data.qpos[2] -= low
                    mujoco.mj_kinematics(self.model, self.data)

        # --- GROUNDING-OFF FLOOR ASSIST (SMALL, capped) --------------------
        # Under "off" the skill owns the absolute base. This absorbs only a SMALL
        # IK under-reach (a declared contact a few cm shy of its surface) by lifting
        # the whole body so that contact rests. It is CAPPED (5 cm) on purpose: a
        # LARGE gap means the base and the legs disagree grossly — e.g. an intended
        # squat authored with straight (body-frame) legs, where lowering the pelvis
        # drives the feet far below the floor. Lifting the base to hide that would
        # silently CANCEL the squat; instead the residual penetration is left to
        # FAIL surface_penetration, so the repair loop must fix it properly (author
        # the feet as world-frame z=0 contacts and let IK bend the knees). Lift-only,
        # surface-aware (a foot on a raised tread is judged against the tread).
        MAX_FLOOR_ASSIST = 0.05
        floor_assist_lift = 0.0                              # how far the base was lifted (m)
        if self.grounding.mode == "off" and active_names:
            idxs = [i for i, n in enumerate(self._contact_names) if n in active_names]
            if idxs:
                deepest = 0.0
                for i in idxs:
                    p = self.data.xpos[self._contact_body_ids[i]]
                    sole_z = float(p[2] - self._contact_offsets[i])
                    surf = self.surfaces.support_z(float(p[0]), float(p[1]),
                                                   sole_z + 0.05, default=0.0)
                    deepest = min(deepest, sole_z - surf)
                if deepest < -1e-3:
                    lift = min(-deepest, MAX_FLOOR_ASSIST)   # cap the correction
                    floor_assist_lift = float(lift)
                    root_pos = root_pos.copy(); root_pos[2] += lift
                    self.data.qpos[2] += lift
                    mujoco.mj_kinematics(self.model, self.data)

        # --- POST-OVERRIDE PENETRATION RESOLVE ------------------------------
        # Bend the JOINTS to push any robot geom out of a surface, AFTER the base
        # is settled and joint_overrides applied. Closes the gap where a JOINT-
        # DRIVEN limb (whose angles are set post-IK, overwriting the in-IK
        # repulsion) — a handstand's tucked knee, a kicking shin — penetrates. Base
        # untouched; inert when nothing penetrates.
        self.ik.resolve_penetration(self._box_aabb, self._robot_geom_list)

        # --- JOINT-BRIDGE DO-NO-HARM GUARD ----------------------------------
        # If a joint-override crossfade left a limb penetrating a surface (a blended
        # intermediate leg config dipping a shin through the floor that
        # resolve_penetration cannot recover), fall back to the RAW authored
        # override for the seamed joints THIS frame — exactly the no-bridge pose,
        # which is penetration-safe. The bridge thus never INCREASES penetration; it
        # only smooths where smoothing is safe. Clearing the seam (with the applied
        # value now == authored) means it does not re-arm next frame -> no oscillation.
        # Threshold = the surface_penetration test's max_depth (5 cm): intervene only
        # when a bridge is about to cause a FAILURE, never on a pre-existing shallow
        # penetration on an unrelated limb (which would needlessly drop a valid
        # arm-gesture bridge over a foot's graze).
        if self._bridge_on and self._joint_seam and self._measure_surface_pen() > 0.05:
            for jn in list(self._joint_seam.keys()):
                if jn in applied_override and jn in self.spec.joint_names:
                    self.data.qpos[7 + self.spec.joint_names.index(jn)] = applied_override[jn]
                    applied_eff[jn] = applied_override[jn]
                    self._joint_seam.pop(jn, None)
            mujoco.mj_kinematics(self.model, self.data)
            self.ik.resolve_penetration(self._box_aabb, self._robot_geom_list)

        # --- PHYSICAL SLEW LIMIT (the last word on the joint vector) ----------
        # No actuated joint may change faster than its torque-derived ceiling
        # (robots/<name>/dynamics.json, rad/s). Applied HERE — after IK, after
        # joint_overrides, after the seam bridge and the penetration resolves — because
        # this is the only point that sees the FINAL joint vector, and every one of
        # those paths can step a joint. Limiting inside the IK alone would leave a
        # joint-driven limb (compute_joint_overrides) free to teleport.
        #
        # This makes `no_teleport` structurally impossible instead of checked after the
        # fact — it was the single largest consumer of the repair loop (55 archived
        # iterations, never solved in 3 skills). It is also surgically selective: at the
        # dynamics ceilings it clips ~0.07% of steps on a walk and ~2.7% on a cartwheel,
        # so ordinary motion is untouched and only the pathological spike is caught.
        #
        # The trade is deliberate: a skill demanding infeasible joint speed now UNDER-
        # REACHES (a lag) instead of snapping (a teleport). That is the better failure,
        # and it is RECORDED (`rate_limit_mag`) rather than silently absorbed, so the
        # repair loop still sees that the authored motion was too fast.
        rate_limit_mag = 0.0
        if self._rate_limit_on and self._joint_rate_cap is not None:
            q_now = self.data.qpos[7:]
            if self._prev_joint_q is not None:
                excess = np.abs(q_now - self._prev_joint_q) - self._joint_rate_cap
                if np.any(excess > 0):
                    rate_limit_mag = float(excess.max())
                    np.clip(q_now, self._prev_joint_q - self._joint_rate_cap,
                            self._prev_joint_q + self._joint_rate_cap, out=q_now)
                    mujoco.mj_kinematics(self.model, self.data)

        # lowest geom CENTER (a resting contact sits above the floor; only a real
        # penetration puts a geom's center below z=0). rbound is too conservative.
        gz = self.data.geom_xpos[self._robot_geoms, 2]
        min_body_z = float(gz.min()) if len(gz) else 0.0
        # ACHIEVED positions over the UNION contact set (EEs + extra contact bodies)
        ee_achieved = self.data.xpos[self._contact_body_ids].copy()   # (Nc, 3)

        # --- FUNDAMENTAL COLLISION MEASUREMENTS (skill-agnostic) --------------
        # SURFACE penetration (CENTER-based): deepest a robot geom's CENTER goes
        # below the floor (z<0) OR inside any scene BOX (tread, slab, carried box).
        # A contact resting ON a surface keeps its center outside -> ~0 (no false
        # positive); only a part driven THROUGH a surface registers. SELF
        # penetration: deepest overlap of non-adjacent robot geoms via
        # mj_geomDistance (the model filters self-collisions from mj_collision).
        xp = self.data.geom_xpos
        gsize = self.model.geom_size
        surface_pen, pen_body, pen_surface = self._measure_surface_pen(localize=True)

        # LOCALIZE the self-collision the same way surface_penetration is localized.
        # A bare scalar tells the repair loop THAT something interpenetrates but not
        # WHAT, and its feedback can then only say "separate the limbs" — which is
        # unactionable when the offending pair is not the one the author would guess.
        # Measured on a sit-chair regeneration: the skill authored the knee contact
        # bodies with SWAPPED signs (right_knee at y=+0.061, left_knee at y=-0.058),
        # so the shins crossed in an X and overlapped 10 cm. The loop burned all 10
        # iterations on it, correctly identified it as the sole remaining failure, and
        # tried three different levers (arms, foot-pelvis gap, torso pitch) — never the
        # knees, because nothing ever named them. Recording the pair makes the fix a
        # one-line sign change instead of an unsolvable search.
        self_pen = 0.0
        self_pair = None
        rb = self._rbound
        for g1, g2 in self._self_pairs:
            if np.linalg.norm(xp[g1] - xp[g2]) < rb[g1] + rb[g2] + 0.02:
                dd = mujoco.mj_geomDistance(self.model, self.data, g1, g2, 0.1, self._fromto)
                if dd < 0.0 and -float(dd) > self_pen:
                    self_pen = -float(dd)
                    self_pair = (
                        mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                          int(self.model.geom_bodyid[g1])),
                        mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                          int(self.model.geom_bodyid[g2])),
                    )

        # --- PER-PHASE CONTACT SCHEDULE + SURFACE CLEARANCE (for generic support/
        # no-slide tests). contact_mask[i] = is EE i DECLARED a contact this phase
        # (contacts(phase), else the static support set). clearance[i] = distance
        # from EE i to the nearest scene surface (floor or object).
        active_names = active if active is not None else self.support_ees
        active_names = active_names or []
        contact_mask = np.array(
            [n in active_names for n in self._contact_names], dtype=bool)
        # clearance = height of each EE's contact point (sole/palm) above the
        # surface DIRECTLY below it (floor z=0, or a box top the EE's xy is over).
        # A point-based gap, not geom-distance: it cleanly separates planted (~0)
        # from lifted (foot pitch doesn't confound it) and generalizes to objects.
        clearance = np.full(len(self._contact_names), 9.9)
        for i, bid in enumerate(self._contact_body_ids):
            p = self.data.xpos[bid]
            sole_z = float(p[2] - self._contact_offsets[i])
            # Surface DIRECTLY below this contact point (floor, tread, seat, ledge —
            # any primitive, any orientation). Probing from just above the sole means
            # a contact already resting registers ~0 rather than finding the floor
            # far beneath the step it is standing on.
            surf = self.surfaces.support_z(float(p[0]), float(p[1]), sole_z + 0.05,
                                           default=0.0)
            clearance[i] = sole_z - surf

        # --- AUTO-ATTACH: place any DECLARED-HELD object at its holder limbs' real
        # (ACHIEVED, post-IK/FK) positions, so a held object coincides with the
        # actual limbs by construction — no open-loop guess about where a joint- or
        # IK-driven hand ended up. Overrides compute_object_poses for the named
        # objects while declared; skills that declare nothing keep authoring poses.
        object_poses = dict(motion_state.get("object_poses", {}) or {})
        attachments = motion_state.get("attachments", {}) or {}
        attach_miss = 0.0
        if attachments:
            name_to_idx = {n: i for i, n in enumerate(self.spec.ee_names)}
            moved_any = False
            for obj, holders in attachments.items():
                idxs = [name_to_idx[h] for h in (holders or []) if h in name_to_idx]
                if not idxs:
                    continue
                pos = ee_achieved[idxs].mean(axis=0)
                prev = object_poses.get(obj)
                quat = (np.asarray(prev[1], dtype=float) if prev
                        else np.array([1.0, 0.0, 0.0, 0.0]))
                # PROXIMITY GATE — a declaration is not a grip. Snapping the object to
                # the holders UNCONDITIONALLY makes the grip true by construction even
                # when the limbs never arrived, which does not merely hide the error: it
                # MANUFACTURES a physically impossible object trajectory in the recorded
                # data. Measured on the carry-box: the hands bottomed out at z=0.448
                # while the crate's grip faces sat at 0.12, and the crate LEVITATED from
                # 0.12 to 0.64 to meet them — 33 cm of self-powered rise, recorded as
                # ground truth. Every other correction in this file is bounded, recorded
                # and surfaced (clamp 45 cm, floor assist 5 cm, reach seat 15 cm); this
                # one was unbounded and silent.
                # So: only attach when the holders are actually WITHIN GRIP RANGE of
                # where the object currently is. Out of range -> leave the object alone,
                # so the miss is VISIBLE (the robot reaches past/short of the crate) and
                # `object_grip` scores the real failure. Recorded as `attach_miss`.
                here = self._object_here(obj, prev)
                if here is not None:
                    gap = float(np.linalg.norm(np.asarray(pos, dtype=float) - here))
                    if gap > self._grip_range(obj):
                        attach_miss = max(attach_miss, gap)
                        continue                      # NOT held — do not move it
                object_poses[obj] = (pos, quat)
                mid = self._mocap_ids.get(obj)
                if mid is not None:
                    self.data.mocap_pos[mid] = np.asarray(pos, dtype=float)
                    self.data.mocap_quat[mid] = quat
                    moved_any = True
            if moved_any:
                mujoco.mj_kinematics(self.model, self.data)   # reflect the moved mocap

        # --- SEAM BRIDGE (object pose) --------------------------------------
        # Turning an attachment ON (pick) or OFF (place) switches the object's pose
        # AUTHORITY (scripted curve <-> achieved-hands mean); a mismatch teleports
        # the object at the seam. Crossfade the final object pose from where it
        # ACTUALLY WAS to the new authority over the bridge window, so a grip/release
        # never snaps the object — this is what previously forced skills to disable
        # attachments and hand-script a floaty box. General: keys only on the jump.
        if self._bridge_on and object_poses:
            moved = False
            for obj, pose in list(object_poses.items()):
                pos = np.asarray(pose[0], dtype=float)
                seed_prev = self._prev_object_pose.get(obj)
                seam = self._object_seam.get(obj)
                if seed_prev is None:
                    continue
                if seam is None and float(np.linalg.norm(pos - seed_prev)) > OBJ_JUMP_M:
                    seam = [seed_prev.copy(), self._bridge_frames]
                    self._object_seam[obj] = seam
                if seam is not None:
                    seam[1] -= 1
                    w = _smoothstep(1.0 - seam[1] / self._bridge_frames)
                    pos = (1.0 - w) * seam[0] + w * pos
                    object_poses[obj] = (pos, pose[1])
                    mid = self._mocap_ids.get(obj)
                    if mid is not None:
                        self.data.mocap_pos[mid] = pos
                        moved = True
                    if seam[1] <= 0:
                        self._object_seam.pop(obj, None)
            if moved:
                mujoco.mj_kinematics(self.model, self.data)

        # --- OBJECT-vs-ROBOT penetration (deepest a carried MOCAP object clips into
        # the robot's body). surface_penetration excludes mocap manipulands, so a box
        # carried INSIDE the torso would otherwise pass silently. The gripping hands
        # legitimately touch the object, so their bodies (and the wrist/forearm one
        # link up) are excluded; everything else (torso, legs, head) must stay clear.
        object_penetration = 0.0
        if self._mocap_object_geoms:
            mujoco.mj_kinematics(self.model, self.data)   # refresh geom_xpos (mocap moved)
            hold_idx = {n: i for i, n in enumerate(self.spec.ee_names)}
            holder_bodies = set()
            for obj, holders in (attachments or {}).items():
                for h in (holders or []):
                    if h in hold_idx:
                        b = self._ee_body_ids[hold_idx[h]]
                        holder_bodies.add(b)
                        holder_bodies.add(int(self.model.body_parentid[b]))
            for og in self._mocap_object_geoms:
                for rg in self._robot_geom_list:
                    if self.model.geom_bodyid[rg] in holder_bodies:
                        continue
                    dd = mujoco.mj_geomDistance(self.model, self.data, og, rg, 0.1, self._fromto)
                    if dd < 0.0:
                        object_penetration = max(object_penetration, -float(dd))

        self.frame += 1

        # The recorded world targets are already union-width (ee_targets spans EEs +
        # contact bodies). For any UNION member with NO IK target this phase (a
        # joint-driven EE, or a contact body driven by FK) the "commanded" target is
        # meaningless — record its ACHIEVED position so the reachability residual
        # (target vs achieved) is 0 and it is not falsely flagged as unreachable.
        ee_targets_rec = ee_targets.copy()
        for i in range(len(self._contact_names)):
            if not ik_mask[i]:
                ee_targets_rec[i] = ee_achieved[i]
        ee_world_rec = ee_targets_rec

        # Update seam-bridge memory for the next frame (seed = this frame's achieved
        # state). Also the per-EE IK-active mask is recorded so the readout/diagnosis
        # can see WHERE a limb switched channel (skip/joint <-> IK) and correlate it
        # with a pose snap.
        ee_active = ik_mask.copy()
        if self._bridge_on:
            for i, name in enumerate(self._contact_names):
                self._prev_ee_active[name] = bool(ik_mask[i])
                self._prev_ee_achieved[name] = ee_achieved[i].copy()
                # remember the AUTHORED (pre-bridge) target so next frame's jump test
                # measures a signal discontinuity, not tracking error (B1). Contacts
                # refresh too, so a contact->free release is judged against the last
                # contact target (usually a real jump -> bridges once, as intended).
                if ik_mask[i]:
                    self._prev_ee_target[name] = authored_now[i].copy()
                else:
                    self._prev_ee_target.pop(name, None)
            self._prev_override = dict(applied_override)
            self._prev_override_eff = dict(applied_eff)
            self._prev_joint_q = self.data.qpos[7:].copy()
            self._prev_object_pose = {o: np.asarray(p[0], dtype=float).copy()
                                      for o, p in object_poses.items()}

        return {
            "time": motion_state["time"],
            "root_pos": root_pos.copy(),
            "root_quat": root_quat.copy(),
            "joints": self.data.qpos[7:].copy(),
            "ee_positions_world": ee_world_rec,            # IK targets (commanded), union-padded
            # BODY-frame commanded targets, straight from the skill. Base-relative, so they
            # are immune to the base re-seating contact grounding applies afterwards — which
            # is what makes a cyclic-closure check meaningful in this pipeline.
            "ee_positions_body": motion_state.get("ee_positions_body"),
            "ee_positions_achieved": ee_achieved,          # post-IK (real), union
            "ee_active": ee_active,                        # (n_ee,) IK-active this phase (channel log)
            "min_body_z": min_body_z,
            "surface_penetration": surface_pen,            # robot vs any surface (m)
            "penetration_body": pen_body,                  # body of the deepest-penetrating geom (or None)
            "penetration_surface": pen_surface,            # surface it clips ("floor"/box name, or None)
            "self_collision": self_pen,                    # robot vs robot (m)
            "collision_pair": self_pair,                   # the two bodies overlapping deepest (or None)
            "attach_miss": attach_miss,                    # declared-held object the holders never reached (m)
            "contact_mask": contact_mask,                  # (N,) declared contacts this phase
            "surface_clearance": clearance,                # (N,) EE -> nearest surface (m)
            "object_poses": object_poses,                  # scene objects (attachments applied)
            "object_penetration": object_penetration,      # carried object vs robot body (m)
            "clamp_events": clamp_events,                  # # EE targets projected out of a surface
            # --- SILENT-INTERVENTION LOG (so the repair loop SEES what the framework
            # quietly corrected, instead of the coder chasing a phantom). Each is the
            # magnitude of a framework correction to the authored motion THIS frame. ---
            "clamp_mag": clamp_mag,                        # deepest EE target lifted out of a surface (m)
            "reach_seat_delta": float(self._reach_delta or 0.0),  # base z shift by the reach-seat (m)
            "floor_assist_lift": floor_assist_lift,        # base z lift by the floor-assist (m, capped 0.05)
            "override_conflicts": override_conflicts,      # # joint-overrides skipped (IK-owned contact chain)
            "reach_proj": float(np.max(reach_proj)),       # deepest free-EE target pulled into reach (m)
            # XY hold applied to planted contacts (m). Bounded and eased, so a small
            # value is the framework absorbing micro-slip and a SATURATED one means the
            # skill is really dragging a contact it declared planted. Recorded because
            # an unrecorded correction is one the repair loop cannot reason about — the
            # old hard pin's release counter was never read by anything.
            "pin_mag": float(self.grounding.pin_mag),
            # How far past its physical slew ceiling the authored motion tried to move a
            # joint this frame (rad). Non-zero means the skill asked for a speed the
            # robot cannot produce and the framework held it back — the repair loop
            # should fix the AUTHORING, not chase the resulting tracking lag.
            "rate_limit_mag": rate_limit_mag,
        }
