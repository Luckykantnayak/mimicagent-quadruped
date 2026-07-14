"""
utils/kinematics.py

Multi-link gradient-descent IK solver.

Morphology-agnostic: works with any number of end-effectors and joints —
both come explicitly from the robot spec.
"""

import numpy as np
import mujoco


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
        tol=1e-3,
    ):
        self.model = model
        self.data = data
        self.body_names = body_names
        self.step_size = step_size
        self.tol = tol

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

    # ── IK solver ────────────────────────────────────────────────────────────

    def damped_least_squares(self, J, dx, damping=1e-3):
        H = J.T @ J + (damping ** 2) * np.eye(J.shape[1])
        return np.linalg.solve(H, J.T @ dx)

    def calculate(self, target_ee_pos, max_iter=100, damping=1e-2, debug=False):
        mujoco.mj_fwdPosition(self.model, self.data)

        for it in range(max_iter):
            current = self.get_ee_positions()
            err = (target_ee_pos - current).flatten()

            if np.linalg.norm(err) < self.tol:
                break

            # 1. Build Jacobian (using dof_ids which map to qvel/nv)
            J_list = []
            jacp = np.zeros((3, self.model.nv))
            for sid, bid in zip(self.site_ids, self.body_ids):
                if sid is not None:
                    mujoco.mj_jacSite(self.model, self.data, jacp, None, sid)
                else:
                    mujoco.mj_jacBody(self.model, self.data, jacp, None, bid)
                J_list.append(jacp[:, self.dof_ids].copy())

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
            q_ref = self.q0[self.qpos_ids]

            # Pull towards standing pose (0.1 is the gain/strength of the pull)
            dq_null = null_space_proj @ (q_ref - q_current)
            dq_total = dq_task + 0.1 * dq_null

            # 5. Integrate back into simulation, clamped to joint limits
            dq_full = np.zeros(self.model.nv)
            dq_full[self.dof_ids] = dq_total * self.step_size
            mujoco.mj_integratePos(self.model, self.data.qpos, dq_full, 1)
            self.data.qpos[self.qpos_ids] = np.clip(
                self.data.qpos[self.qpos_ids], self.joint_lo, self.joint_hi
            )
            mujoco.mj_fwdPosition(self.model, self.data)

        if debug:
            print(f"[IK] it={it:3d} | err={np.linalg.norm(err) * 1000:.2f}mm")
