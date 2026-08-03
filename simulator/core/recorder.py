import numpy as np
import pickle
from utils.math_utils import quat_to_exp_map
import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class MimicKitRecorder:
    """
    Records the controller's per-step state and saves two artifacts:

    - <out>.pkl          : evaluation trajectory consumed by the agent's
                           unit tests (root_pos, root_quat, joints,
                           ee_positions arrays + fps/robot/ee_names metadata)
    - <out>_mimickit.pkl : MimicKit motion dict {fps, loop_mode, frames}
                           where frames = [root_pos(3), exp_map(3), joints(J)]
    """

    def __init__(self, fps, robot="", ee_names=None, support_ees=None,
                 limb_radius=None, object_extents=None, passive_joint_indices=None):
        self.fps = fps
        self.robot = robot
        self.ee_names = list(ee_names) if ee_names else []
        self.support_ees = list(support_ees) if support_ees else []
        # Scale references (for spec-relative unit-test thresholds): the robot's
        # representative limb radius (self-collision) and each object's half-extent
        # (object-grip). None -> the check falls back to its absolute default.
        self.limb_radius = limb_radius
        self.object_extents = dict(object_extents) if object_extents else {}
        # Passive (unactuated) joint columns — excluded from the cyclic-seam test so a
        # continuously-rolling wheel doesn't dominate the loop_mode decision.
        self.passive_joint_indices = list(passive_joint_indices or [])
        self.frames = []
        self.root_pos = []
        self.root_quat = []
        self.joints = []
        self.ee_positions = []
        self.ee_positions_body = []   # BODY-frame targets (base-relative, immune to base re-seating)
        self.ee_positions_achieved = []
        self.min_body_z = []
        self.surface_penetration = []
        self.self_collision = []
        self.contact_mask = []
        self.surface_clearance = []
        self.clamp_events = []     # per-frame count of targets clamped out of a surface
        self.clamp_mag = []        # per-frame deepest target lifted out of a surface (m)
        self.reach_seat_delta = [] # per-frame base-z shift by the reach-seat (m)
        self.floor_assist_lift = []# per-frame base-z lift by the floor-assist (m)
        self.override_conflicts = []# per-frame # joint-overrides skipped (IK-owned contact chain)
        self.reach_proj = []       # per-frame deepest free-EE target pulled into reach (m)
        self.pin_mag = []          # per-frame XY hold applied to a planted contact (m)
        self.rate_limit_mag = []   # per-frame joint slew held back by the physical ceiling (rad)
        self.penetration_body = [] # per-frame body of the deepest-penetrating geom (or None)
        self.penetration_surface = []  # per-frame surface it clips ("floor"/box, or None)
        self.collision_pair = []   # per-frame (bodyA, bodyB) overlapping deepest (or None)
        self.attach_miss = []      # per-frame: declared-held object the holders never reached (m)
        self.object_penetration = []  # per-frame carried-object-vs-robot-body overlap (m)
        self.ee_active = []        # per-frame per-EE IK-active mask (control-channel log)
        self.object_pos = {}       # name -> list of (3,) positions

    def record(self, state):
        """
        state: dict returned by MotionControllerRunner.step()
        """

        pos = state["root_pos"]

        # MuJoCo quat: [w,x,y,z] → MimicKit: [x,y,z,w]
        q = state["root_quat"]
        quat = np.array([q[1], q[2], q[3], q[0]])

        exp = quat_to_exp_map(quat)
        joints = state["joints"]

        frame = np.concatenate([pos, exp, joints])
        self.frames.append(frame)

        self.root_pos.append(np.asarray(pos).copy())
        self.root_quat.append(np.asarray(q).copy())
        self.joints.append(np.asarray(joints).copy())
        self.ee_positions.append(np.asarray(state["ee_positions_world"]).copy())
        _eb = state.get("ee_positions_body")
        if _eb is not None:
            self.ee_positions_body.append(
                np.asarray([(np.zeros(3) if v is None else v) for v in
                            (_eb.values() if isinstance(_eb, dict) else _eb)], dtype=float))
        if "ee_positions_achieved" in state:
            self.ee_positions_achieved.append(np.asarray(state["ee_positions_achieved"]).copy())
        self.min_body_z.append(float(state.get("min_body_z", 0.0)))
        self.surface_penetration.append(float(state.get("surface_penetration", 0.0)))
        self.self_collision.append(float(state.get("self_collision", 0.0)))
        self.clamp_events.append(float(state.get("clamp_events", 0.0)))
        self.clamp_mag.append(float(state.get("clamp_mag", 0.0)))
        self.reach_seat_delta.append(float(state.get("reach_seat_delta", 0.0)))
        self.floor_assist_lift.append(float(state.get("floor_assist_lift", 0.0)))
        self.override_conflicts.append(float(state.get("override_conflicts", 0.0)))
        self.reach_proj.append(float(state.get("reach_proj", 0.0)))
        self.pin_mag.append(float(state.get("pin_mag", 0.0)))
        self.rate_limit_mag.append(float(state.get("rate_limit_mag", 0.0)))
        self.penetration_body.append(state.get("penetration_body"))
        self.penetration_surface.append(state.get("penetration_surface"))
        self.collision_pair.append(state.get("collision_pair"))
        self.attach_miss.append(float(state.get("attach_miss", 0.0)))
        self.object_penetration.append(float(state.get("object_penetration", 0.0)))
        if "contact_mask" in state:
            self.contact_mask.append(np.asarray(state["contact_mask"], dtype=bool).copy())
        if "surface_clearance" in state:
            self.surface_clearance.append(np.asarray(state["surface_clearance"], dtype=float).copy())
        if "ee_active" in state:
            self.ee_active.append(np.asarray(state["ee_active"], dtype=bool).copy())
        for name, pose in state.get("object_poses", {}).items():
            self.object_pos.setdefault(name, []).append(np.asarray(pose[0], dtype=float).copy())

    def save(self, path):
        path = Path(path)

        # ---- Evaluation trajectory (consumed by the agent's unit tests) ----
        traj = dict(
            root_pos=np.vstack(self.root_pos),          # (T, 3)
            root_quat=np.vstack(self.root_quat),        # (T, 4) [w,x,y,z]
            joints=np.vstack(self.joints),              # (T, J)
            ee_positions=np.stack(self.ee_positions),   # (T, N, 3) IK targets
            # BODY-frame targets: base-relative, so they are unaffected by the base
            # re-seating that contact grounding applies after the skill sets root_pos.
            ee_positions_body=(np.stack(self.ee_positions_body)
                               if len(self.ee_positions_body) == len(self.ee_positions) else None),
            ee_positions_achieved=(np.stack(self.ee_positions_achieved)
                                   if self.ee_positions_achieved else None),  # (T, N, 3) real
            fps=self.fps,
            robot=self.robot,
            ee_names=self.ee_names,
            support_ees=self.support_ees,      # per-skill ground-contact EEs
            min_body_z=np.asarray(self.min_body_z),   # (T,) lowest robot point per frame
            surface_penetration=np.asarray(self.surface_penetration),  # (T,) robot vs any surface (m)
            clamp_events=np.asarray(self.clamp_events),                 # (T,) # targets clamped out of a surface
            clamp_mag=np.asarray(self.clamp_mag),                       # (T,) deepest target lifted out of a surface (m)
            reach_seat_delta=np.asarray(self.reach_seat_delta),         # (T,) base-z shift by the reach-seat (m)
            floor_assist_lift=np.asarray(self.floor_assist_lift),       # (T,) base-z lift by the floor-assist (m)
            override_conflicts=np.asarray(self.override_conflicts),     # (T,) # overrides skipped (IK-owned contact)
            reach_proj=np.asarray(self.reach_proj),                     # (T,) deepest free-EE target pulled into reach (m)
            pin_mag=np.asarray(self.pin_mag),                           # (T,) XY hold applied to a planted contact (m)
            rate_limit_mag=np.asarray(self.rate_limit_mag),              # (T,) joint slew held back by the ceiling (rad)
            penetration_body=np.array(self.penetration_body, dtype=object),      # (T,) body of deepest penetration
            penetration_surface=np.array(self.penetration_surface, dtype=object),# (T,) surface it clips
            object_penetration=np.asarray(self.object_penetration),     # (T,) carried object vs robot body (m)
            self_collision=np.asarray(self.self_collision),            # (T,) robot vs robot (m)
            collision_pair=np.array(self.collision_pair, dtype=object),# (T,) the two bodies overlapping deepest
            attach_miss=np.asarray(self.attach_miss),                  # (T,) missed grip distance (m)
            contact_mask=(np.stack(self.contact_mask)                  # (T, N) declared contacts
                          if self.contact_mask else None),
            surface_clearance=(np.stack(self.surface_clearance)        # (T, N) EE -> nearest surface
                               if self.surface_clearance else None),
            ee_active=(np.stack(self.ee_active)                        # (T, n_ee) IK-active mask
                       if self.ee_active else None),
            object_pos={k: np.stack(v) for k, v in self.object_pos.items()},  # {name: (T,3)}
            robot_limb_radius=self.limb_radius,        # scalar: self-collision scale
            object_extents=self.object_extents,        # {name: half-extent}: object-grip scale
        )
        with open(path, "wb") as f:
            pickle.dump(traj, f)
        print(f"✅ Saved eval trajectory: {path}")

        # ---- MimicKit motion ----
        frames = np.vstack(self.frames)
        # Cyclic vs one-shot, measured from the recorded frames (NO skill label): a
        # skill whose end pose returns to its start pose loops cleanly (loop_mode 1);
        # one whose end pose is distinct is one-shot (loop_mode 0). This flag is for
        # the MimicKit TRAINING format (whether the RL trainer loops the clip or plays
        # it once) — NOT a preview-direction hint. GIF/video previews always play
        # FORWARD (looping), never ping-pong/backward, regardless of loop_mode.
        # UNIT-SEPARATED seam (NOT a mixed metres+radians L2, which grows with
        # sqrt(DoF) and is dominated by passive wheel columns — so a wheeled or
        # high-DoF gait was misclassified). Base = pelvis HEIGHT + orientation return
        # (exclude x,y ground advance — a gait shifts along the ground each cycle);
        # joints = ACTUATED joints only, per-joint MAX radians (passive wheels roll
        # freely). Mirrors _inv_end_state / trajectory_readout so all three agree.
        if len(frames) > 1:
            j0 = np.asarray(self.joints[0]); j1 = np.asarray(self.joints[-1])
            passive = set(self.passive_joint_indices)
            act = [k for k in range(len(j0)) if k not in passive]
            base_ret = float(np.linalg.norm(
                np.concatenate([np.asarray(self.root_pos[-1])[2:], frames[-1][3:6]])
                - np.concatenate([np.asarray(self.root_pos[0])[2:], frames[0][3:6]])))
            joint_ret = max((abs(float(j1[k] - j0[k])) for k in act), default=0.0)
            loop_mode = 1 if (base_ret < 0.20 and joint_ret < 0.20) else 0
        else:
            loop_mode = 1
        motion = dict(
            fps=self.fps,
            loop_mode=loop_mode,
            frames=frames,
        )
        mimickit_path = path.with_name(path.stem + "_mimickit" + path.suffix)
        with open(mimickit_path, "wb") as f:
            pickle.dump(motion, f)
        print(f"✅ Saved MimicKit motion: {mimickit_path}")
        print(f"   frames: {frames.shape}")
