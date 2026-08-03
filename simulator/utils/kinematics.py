"""
utils/kinematics.py

Multi-link gradient-descent IK solver.

Morphology-agnostic: works with any number of end-effectors and joints —
both come explicitly from the robot spec.
"""

import numpy as np
import mujoco
import os


# ── IK line-search / convergence tuning ──────────────────────────────────────
# dq_task = J⁺·err is the FULL damped-least-squares (Levenberg–Marquardt) step; the
# damping λ already regularizes it near singularities. So the solve takes a LARGE
# fraction of that step and BACKTRACKS when the task error rises (the Jacobian is only
# valid locally) — a damped Gauss–Newton line search. This replaces the old fixed
# 1%/iteration crawl (step 0.01 × 250 iters left ~8% of a large move unreached: the
# ~50 mm residual that then masqueraded as an unreachable target to the repair loop).
LS_STEP0 = float(os.environ.get("IK_LS_STEP0", "0.9"))  # initial fraction of the DLS step
LS_BACKTRACK = 8      # max halvings when a step overshoots (task error goes up)
LS_MIN_STEP = 0.02    # accept a tiny step below this rather than backtrack forever

# Null-space reference blend (see calculate): the redundant/twist DoFs prefer where
# they ALREADY WERE, with only this fraction pulling back toward the standing home
# pose. Small enough that an inverted pose is not fought every frame, non-zero so a
# DoF the task never constrains cannot wander over a long clip.
HOME_PULL = float(os.environ.get("IK_HOME_PULL", "0.05"))

# Weight on the rotational task rows relative to position. Position error is in
# metres and orientation in radians, so they are not comparable raw; this keeps
# position dominant (a contact must land in the RIGHT PLACE first) while still
# observing the twist DoFs. IK_ORI_WEIGHT=0 disables the rotational task entirely.
ORI_WEIGHT = float(os.environ.get("IK_ORI_WEIGHT", "0.4"))


class MultiLinkGradientDescentIK:
    """
    Multi-link Gradient Descent IK solver using stacked Jacobians.

    Parameters
    ----------
    model       : mujoco.MjModel
    data        : mujoco.MjData
    body_names  : list[str] — IK target bodies (XML names), one per end-effector
    sites       : list[str] — end-effector tip site names, same order as
                  body_names. A missing site falls back to the body position.
    joint_names : list[str] — actuated joint names the IK may move
    step_size   : float
    tol         : float
    """

    def __init__(
        self,
        model,
        data,
        body_names,
        sites,
        joint_names,
        step_size=0.01,
        tol=5e-3,
        null_space_gain=0.1,
    ):
        self.model = model
        self.data = data
        self.body_names = body_names
        self.step_size = step_size
        self.tol = tol
        # Strength of the pull that keeps redundant DoFs near the standing pose.
        # Lower it for high-DoF robots so end-effectors can reach far from home
        # (e.g. a humanoid raising its arms overhead) instead of being tugged back.
        self.null_space_gain = null_space_gain
        # CONTINUITY_REF=0 reverts the null-space reference to home-only (A/B).
        self._continuity_ref = os.environ.get("CONTINUITY_REF", "1") != "0"

        self.qpos_ids = []
        joint_lo, joint_hi = [], []
        for jn in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                # jnt_qposadr is the correct index for qpos array
                self.qpos_ids.append(model.jnt_qposadr[jid])
                if model.jnt_limited[jid]:
                    joint_lo.append(model.jnt_range[jid][0])
                    joint_hi.append(model.jnt_range[jid][1])
                else:
                    joint_lo.append(-np.inf)
                    joint_hi.append(np.inf)
        self.qpos_ids = np.array(self.qpos_ids, dtype=int)
        self.joint_lo = np.array(joint_lo)
        self.joint_hi = np.array(joint_hi)

        # Store initial 'gold standard' standing pose for the null-space pull
        self.q0 = data.qpos.copy()

        # SINGULARITY-FREE REFERENCE: the null-space reference may never sit exactly
        # ON a joint wall. A rest pose with a LOCKED joint is an exact kinematic
        # singularity, not merely an awkward seed: the humanoid's `home` keyframe puts
        # both knees at exactly 0.0 with range [0.0, 2.79] and the leg perfectly
        # straight and vertical, so the knee's Jacobian column is purely HORIZONTAL and
        # has zero first-order authority over foot HEIGHT. A damped first-order solver
        # started there gets exactly zero knee correction, the leg stays locked at full
        # extension, and any downward base motion DRAGS THE FEET THROUGH THE FLOOR —
        # measured: lowering the base 0.88 -> 0.36 with the feet pinned at z=0.05 left
        # the knee at 0.000 for every frame and the foot 0.47 m underground (0.52 m
        # tracking error). That is the squat/crouch/landing primitive every scene and
        # parkour skill is built from, so it broke carrybox (feet 0.26 m under the
        # floor on both pick and place), the ledge-drop landing and the climb mantle.
        # Nudging the REFERENCE a small margin inside the range breaks the degeneracy:
        # the null-space term pushes the joint off the wall, the Jacobian immediately
        # becomes well-conditioned and ordinary IK takes over (same sweep: <= 0.0035 m
        # error at every height). Only the REFERENCE moves — the robot's actual rest
        # pose, init_ee_pos and base_height are untouched, and a pose that does not
        # need to fold is unaffected (standing still solves with the knee at 0.000).
        _span = self.joint_hi - self.joint_lo
        _margin = np.where(np.isfinite(_span), np.minimum(0.05, 0.02 * _span), 0.0)
        self.q0[self.qpos_ids] = np.clip(
            self.q0[self.qpos_ids], self.joint_lo + _margin, self.joint_hi - _margin
        )

        # ── Body IDs ────────────────────────────────────────────────────────
        self.body_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in body_names
        ]

        # ── Site IDs (end-effector tips) ─────────────────────────────────────
        self.site_ids = []
        for site_name in sites:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid < 0:
                print(f"⚠️  Site '{site_name}' not found; falling back to body position")
                sid = None
            self.site_ids.append(sid)

        # Per-EE "up" axis in its own BODY frame: the direction that points along world
        # +z at the HOME pose. Aligning it to a surface normal makes the end-effector
        # sit on that surface exactly the way it sits flat on the floor at home — which
        # avoids having to assume anything about which way a given robot's sole faces.
        # Used by the CONTACT-FLAT orientation task (see calculate). Must come after
        # site_ids/body_ids are resolved.
        self._axis_body = []
        for sid, bid in zip(self.site_ids, self.body_ids):
            R = (data.site_xmat[sid] if sid is not None else data.xmat[bid]).reshape(3, 3)
            self._axis_body.append(R.T @ np.array([0.0, 0.0, 1.0]))

        # ── DOF indices ──────────────────────────────────────────────────────
        self.dof_ids = self._dofs_from_names(joint_names)

        print(f"[IK] Robot bodies : {body_names}")
        print(f"[IK] DOFs found   : {len(self.dof_ids)}")
        print(f"[IK] Sites found  : {sum(s is not None for s in self.site_ids)}/{len(self.site_ids)}")

    # ── DOF helpers ─────────────────────────────────────────────────────────

    def _dofs_from_names(self, joint_names):
        """Build DOF index array from an explicit list of joint names."""
        dof_ids = []
        for jn in joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                dof_ids.append(self.model.jnt_dofadr[jid])
            else:
                print(f"⚠️  Joint '{jn}' not found in model — skipping")
        return np.array(dof_ids, dtype=int)

    # ── Forward kinematics ───────────────────────────────────────────────────

    def get_ee_positions(self):
        """Return current end-effector positions in world frame (N × 3)."""
        pos = []
        for sid, bid in zip(self.site_ids, self.body_ids):
            if sid is not None:
                pos.append(self.data.site_xpos[sid].copy())
            else:
                pos.append(self.data.xpos[bid].copy())
        return np.array(pos)

    def _ee_rotmat(self, i):
        """World rotation matrix of end-effector i (site frame if it has one)."""
        sid, bid = self.site_ids[i], self.body_ids[i]
        m = self.data.site_xmat[sid] if sid is not None else self.data.xmat[bid]
        return np.asarray(m, dtype=float).reshape(3, 3)

    def _ori_error(self, i, spec):
        """Rotational task error (3,) for end-effector i, or None.

        Two modes, because they answer different questions:

        ('align', axis_world) — put this EE's home-'up' axis along `axis_world`,
            leaving rotation ABOUT that axis free. The error is the cross product
            a_current x a_target, whose magnitude is sin(angle) and which vanishes
            for any twist once aligned. This is what a contact needs: a sole flat on
            a tread or a palm flat on a wall, WITHOUT pinning the foot's yaw — a full
            orientation target would over-constrain that and fight the skill.

        ('quat', q) — full orientation, for a skill that genuinely needs all three
            (gripping an object at a specific attitude).
        """
        mode, val = spec
        R = self._ee_rotmat(i)
        if mode == "align":
            a_tgt = np.asarray(val, dtype=float)
            n = np.linalg.norm(a_tgt)
            if n < 1e-9:
                return None
            a_tgt = a_tgt / n
            a_cur = R @ self._axis_body[i]
            return np.cross(a_cur, a_tgt)
        if mode == "quat":
            q_cur = np.zeros(4)
            mujoco.mju_mat2Quat(q_cur, R.flatten())
            err = np.zeros(3)
            mujoco.mju_subQuat(err, np.asarray(val, dtype=float), q_cur)
            return err
        return None

    # ── IK solver ────────────────────────────────────────────────────────────

    def _repel(self, box_aabb, robot_geoms):
        """Task-space push that keeps every ROBOT GEOM out of every surface (the
        floor and each static box). For each geom whose CENTER is inside a surface,
        add a transposed-Jacobian velocity pushing it out along the shallowest
        (min-depth) axis. This is what makes ACHIEVED penetration impossible — the
        target clamp only fixes the AIMED point, but an under-reaching IK chord (a
        foot on the pelvis->target line) or a mid-limb geom (a crawl knee) can still
        pass through a surface even with a legal target. Transposed-Jacobian (not
        pseudo-inverse) keeps it a soft, stable gradient push. INERT when nothing
        penetrates (returns 0), so it never perturbs a clean pose."""
        dq = np.zeros(len(self.dof_ids))
        jacp = np.zeros((3, self.model.nv))
        for gid in robot_geoms:
            p = self.data.geom_xpos[gid]
            depth, d = 0.0, None
            if -float(p[2]) > depth:                      # geom center below the floor
                depth, d = -float(p[2]), np.array([0.0, 0.0, 1.0])
            for c, h in box_aabb:
                dx, dy, dz = h[0]-abs(p[0]-c[0]), h[1]-abs(p[1]-c[1]), h[2]-abs(p[2]-c[2])
                if dx > 0 and dy > 0 and dz > 0:          # center inside this box
                    if dz <= dx and dz <= dy:
                        m, dr = dz, np.array([0.0, 0.0, (1.0 if p[2] >= c[2] else -1.0)])
                    elif dx <= dy:
                        m, dr = dx, np.array([(1.0 if p[0] >= c[0] else -1.0), 0.0, 0.0])
                    else:
                        m, dr = dy, np.array([0.0, (1.0 if p[1] >= c[1] else -1.0), 0.0])
                    if m > depth:
                        depth, d = m, dr
            if d is not None and depth > 1e-4:
                mujoco.mj_jacGeom(self.model, self.data, jacp, None, gid)
                dq += jacp[:, self.dof_ids].T @ (d * depth)
        return dq

    def resolve_penetration(self, box_aabb, robot_geoms, max_iter=12, repel_gain=3.0):
        """Push any ROBOT GEOM out of a surface (floor/box) by adjusting the JOINTS,
        run AFTER joint_overrides have been applied. The in-`calculate` repulsion
        acts during IK, but a JOINT-DRIVEN limb's angles are set by
        compute_joint_overrides AFTER IK — overwriting that fix — so a joint-driven
        knee/shin can still penetrate. This closes that gap generally: it bends
        whatever joints move a penetrating geom out (overridden joints included),
        never touching the base. INERT when nothing penetrates (early-exit)."""
        if box_aabb is None or robot_geoms is None:
            return
        for _ in range(max_iter):
            mujoco.mj_fwdPosition(self.model, self.data)
            dq = self._repel(box_aabb, robot_geoms)
            if not dq.any():
                break
            dq_full = np.zeros(self.model.nv)
            dq_full[self.dof_ids] = dq * repel_gain * self.step_size
            mujoco.mj_integratePos(self.model, self.data.qpos, dq_full, 1)
            self.data.qpos[self.qpos_ids] = np.clip(
                self.data.qpos[self.qpos_ids], self.joint_lo, self.joint_hi)
        mujoco.mj_fwdPosition(self.model, self.data)

    def calculate(self, target_ee_pos, active_mask=None, max_iter=60, damping=1e-2, debug=False,
                  box_aabb=None, robot_geoms=None, repel_gain=3.0, ref_overrides=None,
                  ori_targets=None):
        """Solve IK for the end-effectors flagged in active_mask (default: all).
        End-effectors not flagged are ignored — the IK does not chase them, so a
        limb driven by joint overrides (an expressive arm) is not fought over.
        If box_aabb/robot_geoms are given, a geom-repulsion push keeps the whole
        body out of every surface (see _repel). `ref_overrides` ({joint: angle})
        overlays the skill's declared posture onto the null-space REFERENCE, so the
        redundant DoFs prefer the skill's intended pose rather than always the
        standing home pose (right for inverted/extreme skills)."""
        mujoco.mj_fwdPosition(self.model, self.data)

        # NULL-SPACE REFERENCE — what the redundant DoFs prefer when the task does not
        # determine them. Position-only IK gives 3 constraints against ~7 DoF per limb,
        # so the TWIST axes (hip_z, shoulder_y, abdomen_z) are nearly unobservable in
        # the residual and follow this reference almost entirely.
        #
        # Pulling them toward the STANDING home pose is right for a walk and actively
        # harmful for an inverted or rotated one, where home is maximally wrong: the
        # reference fights the pose every frame and the twist DoFs oscillate (a
        # cartwheel's right_hip_z spiked on 837 frames and REVERSED DIRECTION on 21% of
        # them). So the base reference is CONTINUITY — "stay near where you already
        # were", i.e. the previous frame's solution, which the solver is warm-started
        # from — with only a weak pull toward home so a redundant DoF cannot drift away
        # over a long clip. (mink's posture task is already built this way.)
        #
        # Only the ACTUATED addresses are blended: qpos also holds the base freejoint,
        # whose quaternion must never be linearly interpolated. CONTINUITY_REF=0 → the
        # old home-only behaviour, for A/B.
        q_ref = self.q0.copy()
        if self._continuity_ref:
            prev = self.data.qpos[self.qpos_ids]
            home = self.q0[self.qpos_ids]
            q_ref[self.qpos_ids] = (1.0 - HOME_PULL) * prev + HOME_PULL * home
        # The skill's DECLARED posture still wins outright wherever it is given — an
        # explicit intent beats both continuity and home.
        if ref_overrides:
            for jn, ang in ref_overrides.items():
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid >= 0:
                    q_ref[self.model.jnt_qposadr[jid]] = float(ang)

        n_ee = len(self.body_ids)
        if active_mask is None:
            active = list(range(n_ee))
        else:
            active = [i for i in range(n_ee) if active_mask[i]]
        if not active:
            return

        def residual():
            """Stacked task error: position for every active EE, plus a weighted
            rotational block for those with an orientation target."""
            cur = self.get_ee_positions()
            parts = []
            for i in active:
                parts.append(target_ee_pos[i] - cur[i])
                spec = ori_targets.get(i) if ori_targets else None
                if spec is not None:
                    e = self._ori_error(i, spec)
                    if e is not None:
                        parts.append(ORI_WEIGHT * e)
            return np.concatenate(parts) if parts else np.zeros(0)

        # SINGULARITY SEED — unlock a joint resting exactly ON its limit before the
        # first Jacobian is ever evaluated.
        #
        # A joint pinned at a limit is not merely a poor seed, it can be an EXACT
        # kinematic singularity, and a first-order solver cannot leave one: the
        # humanoid's `home` keyframe puts both knees at exactly 0.0 (range
        # [0.0, 2.79]) with the leg straight and vertical, so the knee axis moves the
        # foot purely HORIZONTALLY and its Jacobian column has ZERO authority over
        # foot HEIGHT. Damped least squares therefore returns zero knee correction,
        # the leg stays locked at full extension, and lowering the base DRAGS THE FEET
        # THROUGH THE FLOOR. That is the squat/crouch/landing primitive underneath
        # every scene-interaction and parkour skill: measured on the carry-box, both
        # knees held exactly 0.0000 for 100 frames while the feet sank 0.26 m under
        # the floor, then snapped to 1.31 rad once the hip had rotated far enough to
        # restore the knee's leverage — a penetration AND a teleport from one cause.
        #
        # Biasing the null-space REFERENCE off the wall is not enough on a real skill:
        # with hands and contact bodies also active the null-space projector squeezes
        # that term to nothing (measured: 0.2844 -> 0.2805 m, no practical change).
        # The seed has to move the CONFIGURATION, which no projector can cancel.
        #
        # Gated on there being a task error worth solving, so a converged pose is
        # never jittered; bounded by the same small margin; and self-correcting —
        # if the nudge was unnecessary the 60 following iterations undo it, and the
        # home pull holds the joint at its limit whenever that is genuinely right.
        if np.linalg.norm(residual()) >= self.tol:
            q_cur = self.data.qpos[self.qpos_ids]
            span = self.joint_hi - self.joint_lo
            margin = np.where(np.isfinite(span), np.minimum(0.05, 0.02 * span), 0.0)
            seeded = np.clip(q_cur, self.joint_lo + margin, self.joint_hi - margin)
            if not np.allclose(seeded, q_cur):
                self.data.qpos[self.qpos_ids] = seeded
                mujoco.mj_fwdPosition(self.model, self.data)

        for it in range(max_iter):
            err = residual()

            if np.linalg.norm(err) < self.tol:
                break

            # 1. Build Jacobian for the ACTIVE end-effectors (dof_ids map to nv).
            # An EE with an orientation target contributes 6 rows (position +
            # rotation) instead of 3 — that is the whole point: position alone gives
            # 3 constraints against ~7 leg DoF, leaving the TWIST axes unobservable and
            # free to chatter. The rotational block observes them.
            J_list = []
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            for i in active:
                sid, bid = self.site_ids[i], self.body_ids[i]
                want_rot = bool(ori_targets and ori_targets.get(i) is not None)
                if sid is not None:
                    mujoco.mj_jacSite(self.model, self.data, jacp,
                                      jacr if want_rot else None, sid)
                else:
                    mujoco.mj_jacBody(self.model, self.data, jacp,
                                      jacr if want_rot else None, bid)
                J_list.append(jacp[:, self.dof_ids].copy())
                if want_rot and self._ori_error(i, ori_targets[i]) is not None:
                    J_list.append(ORI_WEIGHT * jacr[:, self.dof_ids].copy())

            J = np.vstack(J_list)

            # 2. Damped Pseudo-inverse
            lambda_sq = damping ** 2
            J_inv = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(J.shape[0]))

            # 3. Primary Task: Move end-effectors to targets
            dq_task = J_inv @ err

            # 4. Secondary Task: The "Knee Fix" (Null-space)
            # Identity - (Pseudo-inverse * Jacobian)
            I = np.eye(len(self.dof_ids))
            null_space_proj = I - (J_inv @ J)

            # Use the qpos_ids we created in __init__ to avoid IndexError
            q_current = self.data.qpos[self.qpos_ids]
            q_ref_vals = q_ref[self.qpos_ids]

            # Pull redundant DoFs towards the REFERENCE pose (home, overlaid with
            # the skill's declared posture) — configurable gain
            dq_null = null_space_proj @ (q_ref_vals - q_current)
            dq_total = dq_task + self.null_space_gain * dq_null

            # Surface repulsion: push any robot geom out of the floor/boxes, in the
            # task NULL-SPACE so it de-penetrates a mid-limb geom (a knee/shin) without
            # dragging a planted end-effector off its target (the EE targets are already
            # clamped out of surfaces). Inert unless something actually penetrates.
            if box_aabb is not None and robot_geoms is not None:
                dq_repel = self._repel(box_aabb, robot_geoms)
                if dq_repel.any():
                    dq_total = dq_total + repel_gain * (null_space_proj @ dq_repel)

            # Clamping-mode limits: a DoF already pinned at its range and still being
            # pushed further into the wall is frozen — zero its component so the step
            # isn't wasted fighting it (the old clamp-then-continue stalled deep squats
            # and inversions where several leg joints sit at their limits).
            dq_total[(q_current >= self.joint_hi - 1e-6) & (dq_total > 0)] = 0.0
            dq_total[(q_current <= self.joint_lo + 1e-6) & (dq_total < 0)] = 0.0

            # 5. Line-searched integrate: take a LARGE fraction of the damped-Newton
            # step and BACKTRACK if the task error rises (the Jacobian is only valid
            # locally, so a full step can overshoot a contorted pose). This converges a
            # reachable target to sub-mm in a handful of iterations instead of the old
            # 1%/iter crawl, and leaves a genuinely-unreachable target as a clean LARGE
            # residual (which is what the reachability readout wants).
            err0 = np.linalg.norm(err)
            qsave = self.data.qpos.copy()
            step = LS_STEP0
            for _bt in range(LS_BACKTRACK):
                self.data.qpos[:] = qsave
                dq_full = np.zeros(self.model.nv)
                dq_full[self.dof_ids] = dq_total * step
                mujoco.mj_integratePos(self.model, self.data.qpos, dq_full, 1)
                self.data.qpos[self.qpos_ids] = np.clip(
                    self.data.qpos[self.qpos_ids], self.joint_lo, self.joint_hi
                )
                mujoco.mj_fwdPosition(self.model, self.data)
                # Backtrack on the FULL residual (position + orientation), not position
                # alone — otherwise a step that fixes position while wrecking the sole
                # angle would be accepted as progress.
                new_err = np.linalg.norm(residual())
                if new_err <= err0 or step <= LS_MIN_STEP:
                    break
                step *= 0.5

        if debug:
            print(f"[IK] it={it:3d} | err={np.linalg.norm(err) * 1000:.2f}mm")


class MinkIK(MultiLinkGradientDescentIK):
    """
    mink-backed IK — MuJoCo differential-IK as a per-frame QP. SAME public interface
    as the DLS solver (swap via IK_BACKEND), so the controller is unchanged. It is the
    STRUCTURAL cure for the DLS drive-through-floor under-reach on extreme poses:
      * joint ranges are HARD QP constraints (ConfigurationLimit), not a post-hoc clip
        that freezes DoFs at their walls and stalls a near-singular leg;
      * the null-space POSTURE task biases redundant DoFs toward the AUTHORED pose
        (ref_overrides / the crouch), NOT toward standing (the DLS's q0 pull) — so a
        deep crouch / handstand / mantle is regularized toward itself.
    FAST: warm-started from the previous frame + early-exit on a tiny solve velocity,
    so normal motion converges in 1-3 QP iterations (~0.1-0.5 ms each, daqp backend).
    """

    def __init__(self, model, data, body_names, sites, joint_names,
                 step_size=0.01, tol=5e-3, null_space_gain=0.1,
                 velocity_ceilings=None, self_collision=True,
                 collision_margin=0.004):
        super().__init__(model, data, body_names, sites, joint_names,
                         step_size=step_size, tol=tol, null_space_gain=null_space_gain)
        import mink
        self._mink = mink
        self.configuration = mink.Configuration(model)
        # one POSITION-only FrameTask per union member (EE or contact body). A site is
        # a precise tip; without one we drive the body frame (matches the DLS fallback).
        self._tasks = []
        for i, bn in enumerate(body_names):
            if self.site_ids[i] is not None:
                name, ftype = sites[i], "site"
            else:
                name, ftype = bn, "body"
            self._tasks.append(mink.FrameTask(
                frame_name=name, frame_type=ftype,
                position_cost=1.0, orientation_cost=0.0, lm_damping=1e-2))
        # posture (null-space) regularizer — target is set per frame to the authored pose
        self._posture = mink.PostureTask(model, cost=max(float(null_space_gain), 1e-3))

        # ── HARD CONSTRAINTS in the QP (this is mink's real advantage over the DLS:
        #    the solution is FEASIBLE by construction, not fixed up after the fact) ──
        self._limits = [mink.ConfigurationLimit(model)]     # (1) joint POSITION ranges

        # (2) joint VELOCITY ceilings — no actuated joint may step faster than its
        #     torque-derived ceiling (rad/s), so an infeasible slew is impossible.
        vc = dict(velocity_ceilings or {})
        if vc:
            self._limits.append(mink.VelocityLimit(
                model, {jn: float(v) for jn, v in vc.items() if jn in self._qadr_names(joint_names)}))

        # (3) COLLISION AVOIDANCE — object AND body penetration as a hard inequality:
        #     the foot CANNOT be solved through the floor/a box, and (self_collision)
        #     the legs CANNOT be solved into each other. Robot geoms = every geom on a
        #     non-worldbody body; surface geoms = the worldbody collidables (floor + boxes).
        robot_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] > 0
                       and model.geom_contype[g]]
        surface_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == 0
                         and model.geom_contype[g]]
        # CONTACT geoms — those on the union IK roster's bodies (soles, palms, and any
        # declared contact body) — are EXCLUDED from surface avoidance. This simulator
        # is KINEMATIC: a planted contact is *meant* to rest on its surface at ~0
        # clearance, so a hard "stay 4 mm off every surface" constraint over those geoms
        # is a contradiction — it holds the feet off the floor and the robot reads as
        # unsupported for most of the clip (support_or_ballistic 0.875 on a plain walk).
        # Everything ELSE still avoids surfaces, so a shin cannot cut through a slab.
        contact_geoms = {g for g in range(model.ngeom)
                         if int(model.geom_bodyid[g]) in set(int(b) for b in self.body_ids)}
        avoid_geoms = [g for g in robot_geoms if g not in contact_geoms]
        pairs = []
        if avoid_geoms and surface_geoms:
            pairs.append((avoid_geoms, surface_geoms))       # non-contact links vs floor/boxes
        if self_collision and len(robot_geoms) > 1:
            pairs.append((robot_geoms, robot_geoms))         # robot vs itself (adjacency
                                                             # auto-excluded via the model)
        self._collision = None
        if pairs:
            try:
                self._collision = mink.CollisionAvoidanceLimit(
                    model, geom_pairs=pairs, gain=0.85,
                    minimum_distance_from_collisions=collision_margin,
                    collision_detection_distance=max(0.02, collision_margin * 3))
                self._limits.append(self._collision)
            except Exception as e:
                print(f"[IK] mink collision-avoidance disabled ({e})")

        # the base freejoint's tangent DoFs (and any passive DoFs, e.g. go2w wheels) are
        # FROZEN — the skill authors the base absolutely; IK only bends the actuated joints.
        self.base_dof = np.setdiff1d(np.arange(model.nv), self.dof_ids)
        # joint name -> qpos address, to overlay ref_overrides into the posture target.
        self._qadr = {}
        for jn in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                self._qadr[jn] = model.jnt_qposadr[jid]
        print(f"[IK] backend: MINK (QP) | limits: joint-pos"
              f"{' + vel' if vc else ''}"
              f"{' + collision(obj' + (', self' if (self_collision and self._collision) else '') + ')' if self._collision else ''}"
              f" | posture->authored-pose")

    @staticmethod
    def _qadr_names(joint_names):
        return set(joint_names)

    def calculate(self, target_ee_pos, active_mask=None, max_iter=20, damping=1e-2,
                  debug=False, box_aabb=None, robot_geoms=None, repel_gain=3.0,
                  ref_overrides=None):
        mink = self._mink
        # 1. warm-start the QP config from the CURRENT pose (includes the authored base)
        self.configuration.update(self.data.qpos.copy())
        # 2. posture target = the AUTHORED pose (home qpos overlaid with ref_overrides) —
        #    this is the fix: redundant DoFs pull toward the crouch, not toward standing.
        q_ref = self.data.qpos.copy()
        for jn, ang in (ref_overrides or {}).items():
            a = self._qadr.get(jn)
            if a is not None:
                q_ref[a] = float(ang)
        self._posture.set_target(q_ref)
        # 3. only ACTIVE EEs get a task (inactive ones contribute nothing to the QP)
        active = (list(range(len(self._tasks))) if active_mask is None
                  else [i for i in range(len(self._tasks)) if active_mask[i]])
        for i in active:
            self._tasks[i].set_target(
                mink.SE3.from_translation(np.asarray(target_ee_pos[i], dtype=float)))
        tasks = [self._tasks[i] for i in active] + [self._posture]
        # 4. iterate differential solve -> integrate, FREEZING the base; early-exit fast.
        dt, it = 1.0, 0
        for it in range(max_iter):
            vel = mink.solve_ik(self.configuration, tasks, dt, "daqp",
                                damping=damping, limits=self._limits)
            vel[self.base_dof] = 0.0
            if np.abs(vel[self.dof_ids]).max() < 1e-4:      # converged (target met or stuck)
                break
            self.configuration.integrate_inplace(vel, dt)
        # 5. write back; belt-and-suspenders clamp (mink #104 runaway on unreachable
        #    targets) keeps a genuinely-unreachable target at the joint-limited best pose
        #    (a clean LARGE residual for the reach nets) instead of a wild config; FK.
        self.data.qpos[:] = self.configuration.q
        self.data.qpos[self.qpos_ids] = np.clip(
            self.data.qpos[self.qpos_ids], self.joint_lo, self.joint_hi)
        mujoco.mj_fwdPosition(self.model, self.data)
        if debug and active:
            new = self.get_ee_positions()
            e = np.linalg.norm((np.asarray(target_ee_pos)[active] - new[active]).flatten())
            print(f"[IK-mink] it={it} err={e*1000:.2f}mm")


def make_ik(model, data, body_names, sites, joint_names, null_space_gain=0.1,
            velocity_ceilings=None):
    """Construct the IK backend selected by IK_BACKEND.

    'dls'  (DEFAULT) — damped least squares. Tracks the authored targets tightly,
           which is what the contact checks measure; joint limits are enforced by
           clamping each iteration, everything else by the surrounding safety nets.
    'mink' (opt-in)  — the same problem as a per-frame QP, with joint POSITION and
           VELOCITY limits and self/object collision avoidance as HARD constraints,
           so an infeasible solution cannot be produced in the first place.

    Which is better is not obvious and was measured, not assumed. mink's constraints
    are genuinely stronger, but its first-order velocity solve TRACKS more loosely,
    and the contact laws are scored on ACHIEVED positions — so on the bundled skills
    mink currently trades a class of violation it makes impossible for a tracking lag
    that shows up as contact drift. DLS therefore stays the default until the tracking
    gap is closed; use IK_BACKEND=mink to A/B.

    velocity_ceilings: {joint: rad/s} for mink's VelocityLimit (ignored by DLS).
    """
    backend = os.environ.get("IK_BACKEND", "dls").lower()
    if backend == "mink":
        try:
            return MinkIK(model, data, body_names, sites, joint_names,
                          null_space_gain=null_space_gain,
                          velocity_ceilings=velocity_ceilings)
        except ImportError as e:
            # An OPTIONAL dependency: degrade loudly rather than crash, and say what
            # is lost, so a missing package is never mistaken for a working backend.
            print(f"[IK] ⚠️  IK_BACKEND=mink requested but unavailable ({e}).\n"
                  f"[IK]     Falling back to DLS — hard joint-velocity limits and QP "
                  f"collision avoidance are OFF.\n"
                  f"[IK]     Install with:  pip install mink 'qpsolvers[daqp]'")
    return MultiLinkGradientDescentIK(model, data, body_names, sites, joint_names,
                                      null_space_gain=null_space_gain)
