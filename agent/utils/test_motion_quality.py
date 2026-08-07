import os
import re

import numpy as np
import json
from typing import Dict, Any


# =====================================================
#                MATH UTILS
# =====================================================

def quat_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_relative(q1, q2):
    return quat_multiply(q2, quat_conjugate(q1))


def quat_angle(q):
    w = np.clip(q[0], -1.0, 1.0)
    return 2.0 * np.arccos(abs(w))


# =====================================================
#          TRAJECTORY LOADING
# =====================================================

def load_traj(path: str) -> Dict[str, Any]:
    return np.load(path, allow_pickle=True).item()


# =====================================================
#          MORPHOLOGY HELPERS (from robot spec)
# =====================================================

def _actuated(q_full, spec):
    """Drop passive (unactuated) joint columns, preserving order."""
    passive = set(spec.passive_joint_indices)
    idx = [i for i in range(q_full.shape[1]) if i not in passive]
    return q_full[:, idx]


def _ee_ground_z(ee_positions, spec):
    """
    Contact-point heights (T, N): EE z minus per-EE ground offset
    (e.g. wheel center -> rim distance), so 0 means touching the ground.
    """
    offsets = np.asarray(spec.ee_ground_offsets)
    return ee_positions[:, :, 2] - offsets


def _ee_name(traj, spec, i):
    names = traj.get("ee_names") or spec.ee_names
    return names[i] if i < len(names) else str(i)


def _ee_pos(traj):
    """ACHIEVED (post-IK) end-effector positions when the sim recorded them, else
    the IK targets. Contact checks read this so an unreachable/floating pose fails
    honestly instead of passing on grounded-but-unreachable targets."""
    ach = traj.get("ee_positions_achieved")
    return np.asarray(ach) if ach is not None else traj["ee_positions"]


def _support_indices(traj, spec):
    """EE indices of this skill's ground-contact SUPPORT set — the per-skill
    `support_ees` recorded in the trajectory if present, else the spec's contact
    feet. This is what the contact-based checks read, so a cartwheel's hands or a
rear-only support sets are handled without touching the spec."""
    ee_names = traj.get("ee_names") or spec.ee_names
    declared = traj.get("support_ees")
    if declared:
        idx = [ee_names.index(n) for n in declared if n in ee_names]
        if idx:
            return idx
    return spec.contact_ee_indices


# =====================================================
#          RESULT TEMPLATE
# =====================================================

def base_result(name: str, category: str):
    return dict(
        name=name,
        category=category,  # "discrete" | "fidelity"
        passed=True,
        skipped=False,
        reward=0.0,
        metrics={},
        thresholds={},
        violations=[],
        feedback=""
    )


# =====================================================
#          DISCRETE / HARD CONSTRAINTS
# =====================================================








def _deep_or_sustained(values, tol, max_depth, max_frac):
    """Shared verdict for a per-frame penetration signal: fail only on a DEEP
    (> max_depth) or SUSTAINED (> max_frac of frames beyond tol) violation; a
    shallow, brief dip is tracking noise. Returns (passed, deepest, n, budget)."""
    v = np.asarray(values, dtype=float)
    T = len(v)
    deepest = float(v.max()) if T else 0.0
    n = int((v > tol).sum())
    budget = max(3, int(max_frac * T))
    passed = not (deepest > max_depth or n > budget)
    return passed, deepest, n, budget


def eval_surface_penetration(traj, spec, tol=0.02, max_depth=0.05, max_frac=0.03):
    """
    FUNDAMENTAL, skill-agnostic. No robot geom may penetrate ANY surface — the
    floor OR any scene object (stair tread, slab, wall, carried box). Reads the
    per-frame deepest robot-vs-surface overlap the simulator records via
    mj_collision (every surface handled at once). A resting contact or a grip is
    ~0; only a part driven INTO a surface is large. Deep OR sustained -> fail.
    """
    r = base_result("surface_penetration", "discrete")
    sp = traj.get("surface_penetration")
    if sp is None:
        r["passed"] = True
        r["feedback"] = "No surface-collision data recorded (older trajectory)."
        return r
    passed, deepest, n, budget = _deep_or_sustained(sp, tol, max_depth, max_frac)
    r["metrics"] = dict(max_penetration=deepest, num_frames=n, frame_budget=budget)
    r["thresholds"] = dict(tol=tol, max_depth=max_depth, max_frames=budget)
    r["passed"] = passed
    r["reward"] = 0.0 if passed else -1.0
    if passed:
        r["feedback"] = (f"No significant surface penetration (deepest {deepest*100:.1f} cm "
                         f"on {n} frame(s) — within tracking tolerance).")
    else:
        why = "too DEEP" if deepest > max_depth else "too SUSTAINED"
        # LOCALIZE: name the exact body + surface at the deepest frame (recorded by the
        # controller) — "shin_link clips the ledge" is actionable; a bare depth is not.
        where = ""
        pb = traj.get("penetration_body"); ps = traj.get("penetration_surface")
        if pb is not None:
            spv = np.asarray(sp, dtype=float)
            k = int(np.argmax(spv))
            if k < len(pb) and pb[k]:
                surf = (ps[k] if ps is not None and k < len(ps) and ps[k] else "a surface")
                where = f" The offender is body '{pb[k]}' clipping the {surf}."
        r["feedback"] = (f"A body part penetrates a SURFACE ({why}: up to {deepest*100:.0f} cm on "
                         f"{n} frames).{where} Nothing on the robot may go INTO the floor or any object — "
                         f"contacts rest ON surfaces, never through them. Raise the base or adjust the "
                         f"joint angles so THAT link clears (a foot-only fix won't help a shin/torso clip).")
    return r


def eval_self_collision(traj, spec, gross_depth=None, graze_floor=None,
                        min_frames=3, max_frac=0.05, k_gross=0.6, k_graze=0.2):
    """
    SOFT self-collision check (deliberately lenient — a robot grazing itself is
    NOT a deal-breaker; only gross interpenetration is). The overlap is measured
    through FAT collision proxies (capsules) that legitimately overlap in tight
    folds (thigh-calf, arm-across-chest), so a shallow overlap is a measurement
    artifact, not a fault — that noise is what drove the crawl repair loop to
    oscillate. Two gates:
      * DISCARD: if the deepest overlap over the whole trajectory never exceeds
        `graze_floor`, self-penetration is not meaningfully happening -> SKIP
        (report in the geometric readout instead, not as a fail lamp).
      * FAIL only on GROSS interpenetration: overlap > `gross_depth` (a limb
        centerline crossing well inside another body), persisting >= min_frames.
    Shallow/brief overlaps pass silently. (Follow-up: a geometry-normalized,
    per-pair-radius threshold + offline pair-reachability skip is the principled
    version; this depth-floor form is the pragmatic one.)
    """
    r = base_result("self_collision", "discrete")
    sc = traj.get("self_collision")
    if sc is None:
        r["passed"] = True
        r["feedback"] = "No self-collision data recorded (older trajectory)."
        return r
    # SPEC-RELATIVE thresholds: scale to the robot's limb THICKNESS (recorded
    # radius), so a genuine cross fires on a thin-limbed robot too — an absolute
    # 0.08 m "gross" depth is unreachable when limb capsules are ~2 cm across.
    # Falls back to the absolute defaults if no radius was recorded.
    r_limb = traj.get("robot_limb_radius")
    if gross_depth is None:
        gross_depth = max(0.03, k_gross * 2.0 * float(r_limb)) if r_limb else 0.08
    if graze_floor is None:
        graze_floor = max(0.01, k_graze * 2.0 * float(r_limb)) if r_limb else 0.03
    sc = np.asarray(sc, dtype=float)
    deepest = float(sc.max()) if len(sc) else 0.0
    # DISCARD: never more than a graze -> not meaningfully possible/occurring
    if deepest <= graze_floor:
        r["skipped"] = True
        r["metrics"] = dict(max_overlap=deepest)
        r["feedback"] = (f"No meaningful self-collision (deepest overlap only "
                         f"{deepest*100:.1f} cm — grazing/measurement noise).")
        return r
    # FAIL only on gross + sustained interpenetration
    n_gross = int((sc > gross_depth).sum())
    budget = max(min_frames, int(max_frac * len(sc)))
    passed = n_gross <= budget
    r["metrics"] = dict(max_overlap=deepest, num_gross_frames=n_gross, frame_budget=budget)
    r["thresholds"] = dict(gross_depth=gross_depth, graze_floor=graze_floor, max_frames=budget)
    r["passed"] = passed
    r["reward"] = 0.0 if passed else -1.0
    if passed:
        r["feedback"] = (f"No gross self-collision (deepest overlap {deepest*100:.1f} cm; "
                         f"soft check tolerates grazing).")
    else:
        # NAME THE PAIR. "Separate the limbs" is unactionable when the offending pair
        # is not the one the author would guess: a sit-chair regeneration burned all
        # 10 iterations on a 10 cm shin-shin cross caused by SWAPPED knee-target signs,
        # trying arms, foot-pelvis gap and torso pitch — never the knees, because
        # nothing named them. Same lesson as `surface_penetration`'s localization.
        where = ""
        cp = traj.get("collision_pair")
        if cp is not None:
            k = int(np.argmax(sc))
            pair = cp[k] if k < len(cp) else None
            if pair is not None and len(pair) == 2 and pair[0] and pair[1]:
                where = (f" The deepest overlap is '{pair[0]}' against '{pair[1]}'. "
                         f"Fix THOSE two — check the sign/side of each one's target "
                         f"(a left/right swap crosses the limbs) before retuning anything else.")
        r["feedback"] = (f"GROSS self-interpenetration: up to {deepest*100:.0f} cm overlap on "
                         f"{n_gross} frames. A limb is passing well through another body/the "
                         f"torso — not a graze.{where or ' Separate the limbs.'}")
    return r


def eval_support_or_ballistic(traj, spec, dt, eps=0.06, min_flight_s=0.12,
                              fall_thresh=3.0, g=9.81):
    """
    GENERIC support law (replaces the flat-floor `floating`). Reads the SKILL'S OWN
    per-phase contact declaration (`contact_mask`) and the SENSED distance to the
    nearest surface (`surface_clearance`) — no z=0 assumption, works on the floor,
    stairs, a slab, a handstand, one-leg balance, and a real jump.

      resting(t)        := some DECLARED contact is within eps of a surface
      declared_flight(t):= the skill declares NO contact this phase

    Segment the maximal non-resting windows. A window longer than min_flight_s must
    be a REAL jump: if the skill declared flight throughout, the base must be
    ballistic (mean vertical accel < -fall_thresh, i.e. actually falling). If the
    skill declared support but nothing rests, that is a BROKEN contact = floating.
    The skill's own declaration is the discriminator, so no gravity is needed.
    """
    r = base_result("support_or_ballistic", "discrete")
    cm = traj.get("contact_mask")
    cl = traj.get("surface_clearance")
    if cm is None or cl is None:
        r["passed"] = True
        r["feedback"] = "No contact schedule recorded (older trajectory)."
        return r
    cm = np.asarray(cm, dtype=bool)
    cl = np.asarray(cl, dtype=float)
    z = np.asarray(traj["root_pos"])[:, 2]
    T = len(z)
    resting = (cm & (cl <= eps)).any(axis=1)
    declared_flight = ~cm.any(axis=1)
    min_flight = max(3, int(min_flight_s / dt))

    bad = []
    t = 0
    while t < T:
        if resting[t]:
            t += 1
            continue
        t0 = t
        while t < T and not resting[t]:
            t += 1
        if (t - t0) < min_flight:
            continue                                   # brief blip -> ignore
        if declared_flight[t0:t].all():                # declared flight: must fall
            zz = z[t0:t]
            acc = (np.diff(zz, 2) / dt ** 2).mean() if len(zz) >= 3 else 0.0
            if acc > -fall_thresh:
                bad.append((t0, f"declared-flight window is not ballistic (accel {acc:+.1f} m/s^2, not falling)"))
        else:                                          # declared support, none rests
            bad.append((t0, "declared support but no contact is on a surface"))

    r["metrics"] = dict(unsupported_windows=len(bad), rest_frac=float(resting.mean()))
    r["thresholds"] = dict(eps=eps, min_flight_s=min_flight_s, fall_thresh=fall_thresh)
    r["passed"] = len(bad) == 0
    r["reward"] = 0.0 if r["passed"] else -1.0
    if r["passed"]:
        r["feedback"] = f"Supported or ballistic throughout ({resting.mean()*100:.0f}% of frames resting)."
    else:
        r["feedback"] = (f"FLOATING: {len(bad)} unsupported window(s) that neither rest nor fall. "
                         f"First at frame {bad[0][0]}: {bad[0][1]}. Keep a declared contact on a "
                         f"surface, except during a real ballistic jump (base falls, then lands).")
    return r


def eval_contact_grounded(traj, spec, dt, float_thresh=0.08, max_float_frac=0.30,
                          min_declared_s=0.10, exempt_names=None):
    """GENERIC per-contact grounding law: EVERY end-effector/body DECLARED a contact
    must actually REST on its surface — not merely SOME of them (which
    support_or_ballistic already checks: it passes as long as ANY one declared contact
    rests). A hands-and-knees crawl that declares the KNEES as floor contacts but leaves
    them floating in the air passes the support law (the hands rest) yet is NOT a real
    crawl — the knees never bear weight. For each contact, over the frames it is DECLARED,
    the fraction whose SENSED surface clearance exceeds float_thresh must stay small: a
    brief arrival/handoff overlap is allowed (max_float_frac), but a contact that floats
    for most of its declared window fails. Surface-aware (clearance is to the nearest
    surface below), so a contact on a tread/ledge counts as grounded. Skill/morphology-
    agnostic: it only reads the skill's OWN contact declaration + the sensed clearance."""
    r = base_result("contact_grounded", "discrete")
    cm = traj.get("contact_mask"); cl = traj.get("surface_clearance")
    if cm is None or cl is None:
        return _skip("contact_grounded", "No contact schedule recorded.")
    names = traj.get("ee_names") or spec.ee_names
    exempt = set(exempt_names or [])
    cm = np.asarray(cm, dtype=bool); cl = np.asarray(cl, dtype=float)
    min_declared = max(3, int(min_declared_s / dt))
    worst = None
    n_floating = 0
    for i, n in enumerate(names):
        if i >= cm.shape[1] or n in exempt:
            continue
        dcl = cm[:, i]
        if int(dcl.sum()) < min_declared:
            continue                                  # not meaningfully declared -> skip
        c = cl[dcl, i]
        frac = float((c > float_thresh).mean())
        if frac > max_float_frac:
            n_floating += 1
            meanc = float(c.mean())
            if worst is None or meanc > worst[2]:
                worst = (n, frac, meanc)
    r["passed"] = worst is None
    r["metrics"] = dict(num_floating=n_floating,
                        worst_ee=(worst[0] if worst else None),
                        float_frac=(round(worst[1], 3) if worst else 0.0),
                        float_clearance=(round(worst[2], 3) if worst else 0.0))
    r["thresholds"] = dict(float_thresh=float_thresh, max_float_frac=max_float_frac)
    r["reward"] = 0.0 if r["passed"] else -1.0
    if r["passed"]:
        r["feedback"] = "Every declared contact rests on its surface."
    else:
        n, frac, meanc = worst
        r["feedback"] = (f"DECLARED CONTACT FLOATS: '{n}' is declared a contact but hovers "
                         f"{meanc*100:.0f} cm above its surface for {frac*100:.0f}% of its declared "
                         f"frames — it never bears weight. A declared contact MUST reach its surface: "
                         f"lower the BASE (pelvis height + forward pitch) and/or bend the limb so this "
                         f"contact actually rests. A hands-and-knees crawl needs the base low enough that "
                         f"the KNEES rest on the floor, not only the hands. If a limb is NOT meant to bear "
                         f"weight, do NOT declare it a contact.")
    return r


def eval_contact_no_slide(traj, spec, dt, eps=0.035, min_stance_s=0.06, max_drift=0.10,
                          exempt_names=None, pivot_names=None):
    """
    GENERIC no-slide law (replaces the flat-floor `foot_skating`) — a DECLARATION-
    CONSISTENCY check, not a skill prior: a contact declared PLANTED must not
    translate, because that is what "planted" MEANS. It stays always-on and
    fail-closed (planted is the default), so an agent that forgets to declare a
    slide gets the strict check. A limb that legitimately slides (a moonwalk, a
    dragging step) is EXEMPTED only by the planner naming it in the plan's
    `sliding_contacts` (passed here as `exempt_names`) — the opt-out is visible in
    the plan and greppable, never silent. (support_or_ballistic still requires an
    exempt sliding contact to rest on its surface, so "sliding" can't smuggle in
    floating.)

    Over each maximal STANCE RUN (a limb DECLARED a contact AND FIRMLY resting,
    clearance <= eps, for >= min_stance_s), net horizontal displacement must stay
    below max_drift. NET drift over a run (not per-frame speed) is robust: a brief
    swing lift/land is skipped (short run), a wheel is exempt, a pivot foot that
    spins on the spot returns near its start (net ~0), while a drag accumulates and
    fails. `eps` (3.5 cm) is a FIRMLY-planted threshold: a foot lifted past it is
    mid-swing, so its lift-off ramp is excluded (else swing travel reads as slide).
    """
    r = base_result("contact_no_slide", "discrete")
    cm = traj.get("contact_mask")
    cl = traj.get("surface_clearance")
    if cm is None or cl is None:
        r["passed"] = True
        r["feedback"] = "No contact schedule recorded (older trajectory)."
        return r
    cm = np.asarray(cm, dtype=bool)
    cl = np.asarray(cl, dtype=float)
    ee = _ee_pos(traj)                                  # (T, N, 3) achieved
    T = ee.shape[0]
    wheels = set(getattr(spec, "wheel_ee_indices", None) or [])
    # planner-declared sliding contacts are exempt (fail-closed: empty by default)
    exempt = set()
    ee_names = traj.get("ee_names") or spec.ee_names
    for nm in (exempt_names or []):
        if nm in ee_names:
            exempt.add(ee_names.index(nm))
    wheels = wheels | exempt
    # A declared PIVOT contact spins on the spot: its EE ORIGIN traces a small circle
    # (an ankle a few cm off the spin axis), so it is exempted with a BOUNDED budget —
    # NOT fully like a wheel/slider — so a real pirouette passes but a "pivot" that
    # translates half a metre still fails.
    pivots = set()
    for nm in (pivot_names or []):
        if nm in ee_names:
            pivots.add(ee_names.index(nm))
    pivot_drift = 3.0 * max_drift                        # ~0.30 m: covers the spin circle
    resting = cm & (cl <= eps)
    min_stance = max(2, int(min_stance_s / dt))
    worst_any = 0.0                                       # worst drift of ANY stance (info)
    viol, viol_ee = 0.0, None                             # worst OVER-limit drift (drives fail)
    for i in range(cm.shape[1]):
        if i in wheels:
            continue
        limit = pivot_drift if i in pivots else max_drift
        t = 0
        while t < T:
            if not resting[t, i]:
                t += 1
                continue
            t0 = t
            while t < T and resting[t, i]:
                t += 1
            if (t - t0) < min_stance:                   # brief lift/land -> not a stance
                continue
            net = float(np.linalg.norm(ee[t - 1, i, :2] - ee[t0, i, :2]))
            worst_any = max(worst_any, net)
            if net > limit and net > viol:              # per-EE limit (pivots get more)
                viol, viol_ee = net, _ee_name(traj, spec, i)
    fail = viol_ee is not None
    r["metrics"] = dict(max_stance_drift=(viol if fail else worst_any), worst_ee=viol_ee)
    r["thresholds"] = dict(eps=eps, max_drift=max_drift, pivot_drift=pivot_drift,
                           min_stance_s=min_stance_s)
    r["passed"] = not fail
    r["reward"] = 0.0 if r["passed"] else -1.0
    if r["passed"]:
        r["feedback"] = f"No contact sliding (worst planted-contact drift {worst_any*100:.1f} cm)."
    else:
        r["feedback"] = (f"Contact SLIDING — {viol_ee} drifts {viol*100:.0f} cm while planted, over "
                         f"its allowed budget. A planted contact must not translate; lift it to "
                         f"reposition (a wheel rolls; a pivot may spin but not travel).")
    return r








def eval_joint_limits(traj, spec):
    r = base_result("joint_limits", "discrete")

    q = _actuated(traj["joints"], spec)

    violated = []

    for i, (name, (jmin, jmax)) in enumerate(spec.joint_limits.items()):
        if q[:, i].min() < jmin or q[:, i].max() > jmax:
            violated.append(name)

    r["metrics"]["violated_joints"] = violated
    r["passed"] = len(violated) == 0
    r["reward"] = 0.0 if r["passed"] else -1.0
    r["feedback"] = (
        "Joint limits respected."
        if r["passed"]
        else f"Joint limits violated for: {violated}."
    )
    return r


def eval_no_teleport(traj, spec, dt, rate_ceiling=None, abs_floor=0.05,
                     iso_factor=4.0, max_frac=0.005):
    """
    GENERIC, skill-agnostic PHYSICS law (a general law, NOT the z-scored smoothness
    aesthetic): no actuated joint may STEP discontinuously between two adjacent
    frames. A control-channel handoff (joint-override <-> IK) or an unbridged target
    jump teleports the joints 1-2 rad in one 5 ms frame — hundreds of rad/s, far
    beyond any actuator — which reads as an "abrupt pose change" even when the motion
    is semantically right. This catches the ISOLATED single-frame pop specifically:
    a frame is a teleport where a joint's step BOTH exceeds a physical slew ceiling
    (rad/s) AND stands out as an isolated spike vs that joint's own median motion, so
    genuinely fast but smooth motion is never flagged. It reads only `joints`, so it
    works for every robot and every skill with no per-skill parameters.
    """
    r = base_result("no_teleport", "discrete")
    q = _actuated(traj["joints"], spec)
    if q.shape[0] < 3:
        r["feedback"] = "Trajectory too short to assess continuity."
        return r
    dq = np.abs(np.diff(q, axis=0))                       # (T-1, J) rad/frame
    # LOCAL isolation: a teleport is a lone spike whose IMMEDIATE neighbours are
    # small (the joint jumps, then stays), NOT a sustained fast RAMP (many
    # consecutive large steps — a real swing, or the seam bridge's own crossfade,
    # which is smooth and must NOT be flagged). Compare each step to the larger of
    # its two temporal neighbours per joint.
    up = np.vstack([dq[1:], dq[-1:]])                     # dq[k+1] (last clamped)
    dn = np.vstack([dq[:1], dq[:-1]])                     # dq[k-1] (first clamped)
    neigh = np.maximum(up, dn)
    # Physical slew ceiling (rad/s), PER JOINT: a joint cannot step faster than this
    # in one frame. Use the spec's per-joint velocity ceilings when it exposes them
    # (personalizes the check per robot), else a generous morphology-agnostic default
    # well above any real legged-robot actuator. (Previously read a typo'd singular
    # `joint_speed_ceiling` that no spec exposes, so EVERY robot silently got the
    # 30 rad/s default — the per-robot dynamics were never applied.)
    if rate_ceiling is not None:
        ceil = np.full(q.shape[1], float(rate_ceiling))
    else:
        per = getattr(spec, "joint_speed_ceilings", None) or {}
        names = spec.actuated_joint_names
        ceil = np.array([float(per.get(names[j], 30.0)) if j < len(names) else 30.0
                         for j in range(q.shape[1])])
    step_ceil = ceil * dt                                 # (J,) rad/frame, per joint
    # abs_floor is a PERCEPTUAL noise floor (a step below it is invisible at any fps);
    # it is NOT the physical gate. The binding gate is `dq > step_ceil` (the per-joint
    # rad/s ceiling * dt) below — fps- and robot-aware. (The old 0.15 rad/FRAME floor
    # was fps-BLIND: at 200 fps it demanded 30 rad/s before the ~8 rad/s physical ceil
    # was even consulted, so infeasible sub-0.15 pops — the moonwalk foot pop — slipped
    # through. Lowered to 0.05; sustained-fast-but-smooth is still excluded by the
    # isolation term below, which a ramp/crossfade never satisfies.)
    isolated = (dq > abs_floor) & (dq > iso_factor * (neigh + 1e-4))
    pops = isolated & (dq > step_ceil)                    # (T-1,J) > (J,) broadcasts per joint
    gross = isolated & (dq > 4.0 * step_ceil)             # isolated, >> its own ceil (~4x)
    n = int(pops.any(axis=1).sum())                       # frames with >= 1 borderline pop
    n_gross = int(gross.any(axis=1).sum())                # frames with a gross teleport
    worst = float(dq.max())
    # A small budget tolerates a few borderline (just-over-ceiling) frames; a single
    # GROSS teleport (>= 4x the slew ceiling, isolated) fails regardless of count —
    # a lone 1.7 rad snap is a handoff pop, not a borderline detection.
    budget = max(1, int(max_frac * q.shape[0]))
    r["metrics"] = dict(num_teleport_frames=n, num_gross=n_gross, worst_step=worst,
                        frame_budget=budget, rate_ceiling=round(float(ceil.min()), 1))
    r["thresholds"] = dict(step_ceiling_rad=round(float(step_ceil.min()), 3),
                           gross_step_rad=round(float(4.0 * step_ceil.min()), 3),
                           abs_floor=abs_floor)
    r["passed"] = (n_gross == 0) and (n <= budget)
    r["reward"] = 0.0 if r["passed"] else -1.0
    if r["passed"]:
        r["feedback"] = f"No joint teleports (worst single-frame step {worst:.2f} rad)."
    else:
        masked = np.where(pops, dq, 0.0)
        fi, ji = np.unravel_index(int(masked.argmax()), dq.shape)
        phase = (fi + 1) / (q.shape[0] - 1)
        names = list(spec.joint_limits.keys())
        jname = names[ji] if ji < len(names) else str(ji)
        # Condition the CAUSE hint on whether the framework was pulling a free limb's
        # target into reach near the worst frame — an out-of-reach free target makes IK
        # flip configurations (the joints jump), which is a DIFFERENT fix (bring the
        # target into reach) than a channel handoff (retime the seam). Don't guess one.
        rp = traj.get("reach_proj")
        reach_cause = False
        if rp is not None:
            rp = np.asarray(rp, dtype=float)
            w = slice(max(0, fi - 2), min(len(rp), fi + 3))
            reach_cause = rp.size > 0 and float(rp[w].max() if rp[w].size else 0.0) > 0.05
        cause = (
            "A free limb's IK TARGET IS BEYOND REACH near this frame (the framework was "
            "pulling it back onto the reachable ball) — an unreachable target makes IK flip "
            "configurations, jerking the joints. Bring that limb's target into reach (author "
            "it closer to the torso, move the base nearer, or drive the limb by JOINT angles "
            "if its world position doesn't matter)."
            if reach_cause else
            "This is a control-channel handoff snapping (a limb switched between joint-override "
            "and IK, or its IK target jumped, in a single instant). Make the switch happen over "
            "a phase WINDOW where the outgoing and incoming targets AGREE at the seam (ease in "
            "and out); never toggle a limb's channel or jump its target in one frame.")
        r["feedback"] = (
            f"JOINT TELEPORT: {n} frame(s) where a joint STEPS discontinuously "
            f"(worst {worst:.2f} rad in one {dt*1000:.0f} ms frame ~ {worst/dt:.0f} rad/s), "
            f"the worst at phase {phase:.2f} on '{jname}'. {cause}")
    return r


def eval_ee_no_teleport(traj, spec, dt, v_ceil=10.0, abs_floor=0.03,
                        iso_factor=4.0, max_frac=0.005):
    """GENERIC continuity law in END-EFFECTOR space (the EE-space companion to
    no_teleport, which only sees JOINT steps). A FREE limb whose IK target is beyond
    reach makes the seam bridge sawtooth — the hand teleports ~1 m per frame while the
    joints move less dramatically — invisible to the joint check. Measured
    BASE-COMPENSATED: subtract the whole-body root step, so a legitimate whole-body
    shift (grounding snap-up, post-FK re-seat, floor assist) is cancelled and only the
    per-LIMB flicker (a hand jumping relative to a stationary torso) remains. Same
    isolated-spike grammar as no_teleport; morphology-agnostic (reads achieved EE +
    root only)."""
    r = base_result("ee_no_teleport", "discrete")
    ach = traj.get("ee_positions_achieved")
    if ach is None:
        return _skip("ee_no_teleport", "No achieved EE positions recorded (older trajectory).")
    ach = np.asarray(ach, dtype=float)                    # (T, Nc, 3)
    root = np.asarray(traj["root_pos"], dtype=float)      # (T, 3)
    if ach.shape[0] < 3:
        r["passed"] = True; r["feedback"] = "Trajectory too short to assess."; return r
    d = np.linalg.norm(np.diff(ach, axis=0) - np.diff(root, axis=0)[:, None, :], axis=2)  # (T-1, Nc)
    up = np.vstack([d[1:], d[-1:]]); dn = np.vstack([d[:1], d[:-1]]); neigh = np.maximum(up, dn)
    step_ceil = v_ceil * dt
    isolated = (d > abs_floor) & (d > iso_factor * (neigh + 1e-4))
    pops = isolated & (d > step_ceil)
    gross = isolated & (d > 4.0 * step_ceil)
    n = int(pops.any(axis=1).sum()); n_gross = int(gross.any(axis=1).sum())
    worst = float(d.max()); budget = max(1, int(max_frac * ach.shape[0]))
    r["metrics"] = dict(num_teleport_frames=n, num_gross=n_gross, worst_step=round(worst, 3),
                        frame_budget=budget)
    r["thresholds"] = dict(v_ceil=v_ceil, abs_floor=abs_floor)
    r["passed"] = (n_gross == 0) and (n <= budget)
    r["reward"] = 0.0 if r["passed"] else -1.0
    names = traj.get("ee_names") or spec.ee_names
    if r["passed"]:
        r["feedback"] = f"No end-effector teleports (worst base-relative step {worst*100:.0f} cm/frame)."
    else:
        masked = np.where(pops, d, 0.0)
        fi, ei = np.unravel_index(int(masked.argmax()), d.shape)
        phase = (fi + 1) / (ach.shape[0] - 1)
        en = names[ei] if ei < len(names) else str(ei)
        r["feedback"] = (
            f"END-EFFECTOR TELEPORT: '{en}' jumps {worst*100:.0f} cm in one {dt*1000:.0f} ms frame "
            f"(~{worst/dt:.0f} m/s) relative to the body, at phase {phase:.2f}. Usually a FREE limb "
            f"whose IK target is BEYOND REACH (the seam bridge sawtooths against an unreachable point), "
            f"or an unbridged channel handoff. Pull the target into reach (reach_limited_target, or "
            f"lower/advance the base) — do not retime the handoff.")
    return r


def eval_framework_interventions(traj, spec, dt, clamp_cap=0.15, seat_frac=0.20,
                                 assist_frac=0.30, proj_cap=0.20, override_frac=0.20):
    """GENERIC observability law: the framework SILENTLY corrects impossible authored
    intent (it lifts an EE target OUT of a solid it was authored inside; it seats the base
    to keep a declared contact reachable, up to a hard cap; it lifts the base off a
    too-deep contact). Small corrections are fine, but a SYSTEMATIC / SATURATED one means
    the AUTHORING is wrong in a way no other check names — so FAIL, and name the exact
    root cause the recorded magnitude implies. (clamp_mag=0.45 => 'your targets are inside
    a solid'; reach_seat pinned at its cap => 'a contact is held beyond reach'.) This is
    what turns a 20-iteration circle into a one-iteration fix."""
    r = base_result("framework_interventions", "discrete")
    T = len(traj["root_pos"])
    cmag = np.asarray(traj.get("clamp_mag", []), float)
    rs = np.asarray(traj.get("reach_seat_delta", []), float)
    fa = np.asarray(traj.get("floor_assist_lift", []), float)
    rpj = np.asarray(traj.get("reach_proj", []), float)
    ovc = np.asarray(traj.get("override_conflicts", []), float)
    msgs = []
    clamp_max = float(cmag.max()) if cmag.size else 0.0
    seat_pin = float((np.abs(rs) >= 0.149).mean()) if rs.size else 0.0
    assist_pin = float((fa >= 0.049).mean()) if fa.size else 0.0
    proj_max = float(rpj.max()) if rpj.size else 0.0
    # CHANNEL CONFLICT: joint_overrides SILENTLY DISCARDED because that limb also carried an
    # IK target this frame (ONE CHANNEL PER LIMB). Sustained drops mean the authored JOINT
    # POSE never happened at all — the limb is wherever IK put it — and NO other check names
    # this: the motion just looks wrong. (Measured on a stance pose authored with both
    # channels: 12 joints dropped on 100% of frames, stance thigh 1.3 rad off its target.)
    ovc_frac = float((ovc > 0).mean()) if ovc.size else 0.0
    ovc_max = float(ovc.max()) if ovc.size else 0.0
    if clamp_max > clamp_cap:
        msgs.append(
            f"TARGET-CLAMP lifted an EE target {clamp_max*100:.0f} cm OUT of a solid surface — "
            f"an EE's (x,y) is INSIDE a static object's footprint at a z BELOW that object's top "
            f"(e.g. a foot authored 'on the floor' at an x,y that is actually inside a chair/box, "
            f"which is SOLID to the floor). Move that contact OUTSIDE the object's footprint (onto "
            f"the real floor in front of the object), or onto the object's TOP at z=top+offset.")
    if seat_pin > seat_frac:
        msgs.append(
            f"REACH-SEAT pinned at its {clamp_cap*100:.0f} cm cap for {seat_pin*100:.0f}% of frames — "
            f"a DECLARED contact is held while the base moved BEYOND that limb's reach of it (the base "
            f"rises/advances past surface_top + limb_reach). RELEASE that contact and re-plant it before "
            f"the base leaves its reach ball — do not hold it 'for stability'.")
    if assist_pin > assist_frac:
        msgs.append(
            f"FLOOR-ASSIST pinned at its 5 cm cap for {assist_pin*100:.0f}% of frames — declared contacts "
            f"are authored well BELOW their surface (z too low); author each contact z = surface + offset.")
    am = np.asarray(traj.get("attach_miss", []), float)
    attach_max = float(am.max()) if am.size else 0.0
    if attach_max > 1e-6:
        msgs.append(
            f"MISSED GRIP — a limb declared in `attachments()` as HOLDING an object was up to "
            f"{attach_max*100:.0f} cm away from it, so the framework REFUSED to attach (it will not "
            f"teleport an object onto a limb that never arrived). The object stayed where it was and "
            f"the 'carry' never happened. Get the limb TO the object first: crouch lower and/or step "
            f"nearer so the hand actually reaches the grip point, THEN declare the attachment.")
    if proj_max > proj_cap:
        msgs.append(
            f"FREE-LIMB REACH-PROJECTION pulled a limb's target {proj_max*100:.0f} cm back onto the reachable "
            f"ball — a free limb (often a HAND reaching for / GRIPPING an object) was authored BEYOND its "
            f"reach, so it cannot get there and (if it grips an object) the object TELEPORTS to the achieved "
            f"hand. Get the limb-root CLOSER to the target: crouch LOWER and/or step NEARER so the hand "
            f"actually reaches the grip point — a grip authored beyond reach is a missed grip, not a hold.")
    # ADVISORY, NOT A FAILURE. Dropping an override on an IK-owned limb is sometimes the
    # DESIGN (the value still biases the IK null-space, which is how a swing is styled), and
    # sometimes a silent bug (the limb's joint POSE was the point and never happened). The
    # recorded count cannot tell those apart, so this REPORTS instead of failing — failing it
    # would flunk legitimate null-space styling (measured: 12 of 35 existing skills carry
    # sustained conflicts). The diagnosis agent decides, using the motion readout.
    notes = []
    if ovc_frac > override_frac:
        notes.append(
            f"CONTROL-CHANNEL CONFLICT — up to {ovc_max:.0f} compute_joint_overrides value(s) were "
            f"DISCARDED on {ovc_frac*100:.0f}% of frames because those joints belong to a limb that ALSO "
            f"returned an IK target (world- or body-frame) that phase. ONE CHANNEL PER LIMB PER PHASE: "
            f"a limb with any IK target is IK-owned and its joint angles are NOT applied (they only bias "
            f"the IK null-space). If those angles were meant as STYLING, this is fine — ignore it. If the "
            f"limb's joint POSE was the point, IT NEVER HAPPENED — the limb is wherever IK put it, and no "
            f"other check names this; check the motion readout against the intended pose. Fix (only if the "
            f"pose was the point): for every limb whose JOINT "
            f"CONFIGURATION is the point (a rearing/handstand/side stance, an expressive gesture), return "
            f"None from BOTH compute_ee_position_body_frame and compute_ee_position_world_frame for that "
            f"limb while the pose is held — then its overrides apply. (Declaring it in contacts() is fine "
            f"and does NOT give it an IK target.) Conversely, if the limb should be IK-driven, DELETE its "
            f"overrides and shape it via the null-space bias instead.")
    r["metrics"] = dict(clamp_max=round(clamp_max, 3), reach_seat_pinned_frac=round(seat_pin, 2),
                        floor_assist_pinned_frac=round(assist_pin, 2), reach_proj_max=round(proj_max, 3),
                        override_conflict_frac=round(ovc_frac, 2), override_conflict_max=round(ovc_max, 1),
                        attach_miss_max=round(attach_max, 3))
    r["thresholds"] = dict(clamp_cap=clamp_cap, seat_frac=seat_frac, assist_frac=assist_frac, proj_cap=proj_cap)
    r["passed"] = not msgs
    r["reward"] = 0.0 if r["passed"] else -1.0
    r["feedback"] = ("No saturated framework interventions (authored intent is physically placeable)."
                     if r["passed"] else
                     "SATURATED FRAMEWORK CORRECTION — the authored motion is impossible and was silently "
                     "corrected; fix the SOURCE:\n  - " + "\n  - ".join(msgs))
    # ADVISORY notes ride along in the feedback but never change pass/fail or reward.
    if notes:
        r["feedback"] += ("\n  [ADVISORY — not a failure, judge against the motion readout]\n  - "
                          + "\n  - ".join(notes))
    return r


def eval_joint_velocity_envelope(traj, spec, dt, tol=1.5, pct=90):
    """
    GENERIC physical check: no actuated joint may SUSTAIN a velocity above its
    physical ceiling — the peak speed a full-range oscillation reaches at the
    actuator's torque limit (τ = I·q̈), pre-computed per joint into
    robots/<name>/dynamics.json. Complements `no_teleport` (which catches an
    ISOLATED single-frame teleport); this catches a SUSTAINED physically-infeasible
    speed — a joint moving faster than any actuator could drive it for MOST of its
    motion — which is fiction an imitation-learning policy could never track.
    "Sustained" = the joint's `pct`-th-percentile speed exceeds `tol`× its ceiling,
    so a BRIEF fast burst (a ballet whip, a kick — where p90 stays low) is NOT
    flagged, only a joint that is fast throughout. `tol` (~1.5×) is lenient because
    the ceilings are order-of-magnitude estimates. Self-SKIPS when the robot has no
    ceilings (a torque-less MJCF), like the summary's dynamics tier.
    """
    r = base_result("joint_velocity_envelope", "discrete")
    ceils = spec.joint_speed_ceilings
    if not ceils:
        r["skipped"] = True
        r["feedback"] = "No joint velocity ceilings (model carries no torque source)."
        return r
    q = _actuated(traj["joints"], spec)
    if q.shape[0] < 3:
        r["feedback"] = "Trajectory too short to assess velocities."
        return r
    vel = np.abs(np.diff(q, axis=0)) / dt                 # (T-1, J) rad/s
    names = list(spec.joint_limits.keys())
    n_bad, worst_ratio, worst_joint = 0, 0.0, None
    for j, name in enumerate(names):
        c = ceils.get(name)
        if not c or j >= vel.shape[1]:
            continue
        sustained = float(np.percentile(vel[:, j], pct))  # fast for >(100-pct)% of frames
        ratio = sustained / (float(c) + 1e-9)
        if ratio > tol:
            n_bad += 1
            if ratio > worst_ratio:
                worst_ratio, worst_joint = ratio, name
    r["metrics"] = dict(num_joints_over=n_bad, worst_ratio=round(worst_ratio, 2),
                        worst_joint=worst_joint)
    r["thresholds"] = dict(tol=tol, percentile=pct)
    r["passed"] = n_bad == 0
    r["reward"] = 0.0 if r["passed"] else -1.0
    if r["passed"]:
        r["feedback"] = "All joint velocities within physical ceilings."
    else:
        r["feedback"] = (
            f"JOINT OVER-SPEED: '{worst_joint}' sustains ~{worst_ratio:.1f}x its "
            f"physical velocity ceiling (its p{pct} speed) — faster than its actuator "
            f"could drive it for most of the motion, so an imitation policy cannot "
            f"track it. Slow that joint (smaller amplitude or a longer phase window).")
    return r






# =====================================================
#          AGGREGATION LOGIC
# =====================================================

def aggregate_discrete(results):
    discrete = [
        r for r in results
        if r["category"] == "discrete" and not r["skipped"]
    ]

    return dict(
        passed=all(r["passed"] for r in discrete),
        failed_tests=[r["name"] for r in discrete if not r["passed"]]
    )


# =====================================================
#          GATED / SKILL-SPECIFIC EVALUATORS
# =====================================================

def _skip(name, why):
    r = base_result(name, "discrete")
    r["skipped"] = True
    r["feedback"] = why
    return r


def eval_flight_window(traj, spec, dt, max_flight_s=None, max_apex=None):
    """GATED on a declared flight window ([] contact segment). The intended aerial
    phase must be plausible for this morphology (a real jump, not a launch).

    The apex is measured as the ballistic RISE above the TAKEOFF SURFACE (base z just
    before flight), NOT absolute z — so a legal hop onto raised terrain (a slab/step)
    is judged by its arc height, terrain-independent. The rise ceiling is the
    dynamics-tier leg-push jump estimate (the physical scale), falling back to a
    height-SPAN multiple; the static envelope MAX is the wrong scale (it false-fails a
    hop onto terrain and is vacuous on a tall robot). support_or_ballistic already
    enforces the arc is a real ballistic rise."""
    cm = traj.get("contact_mask")
    if cm is None:
        return _skip("flight_window", "No contact schedule.")
    flight = ~np.asarray(cm, dtype=bool).any(axis=1)
    if not flight.any():
        return _skip("flight_window", "No declared flight window (not aerial).")
    z = np.asarray(traj["root_pos"])[:, 2]
    runs, t, T = [], 0, len(flight)
    while t < T:
        if not flight[t]:
            t += 1; continue
        t0 = t
        while t < T and flight[t]:
            t += 1
        runs.append((t0, t))
    dur = max((b - a) for a, b in runs) * dt
    # apex = RISE above the takeoff surface (base z at the frame just before each
    # flight run), so a hop onto a raised surface is judged by its arc, not terrain z.
    rise = max((float(z[a:b].max()) - float(z[max(0, a - 1)])) for a, b in runs)
    apex = float(rise)
    # Ceiling from the dynamics-tier jump estimate (leg push) — 2.0x, generous over
    # the conservative statics estimate so an athletic jump/flip isn't false-failed
    # while an absurd "launch" still is; else a height-SPAN fallback (the envelope max
    # would false-fail a terrain hop / be vacuous). Airtime ceiling is derived from
    # the apex ceiling via gravity (t = 2*sqrt(2h/g)), so the two can't disagree.
    jh = getattr(spec, "jump_height", None)
    span = float(spec.base_height_range[1] - spec.base_height_range[0])
    apex_ceil = float(max_apex) if max_apex is not None else (2.0 * jh if jh else 1.2 * span)
    flight_ceil = (float(max_flight_s) if max_flight_s is not None
                   else 1.3 * 2.0 * (2.0 * apex_ceil / 9.81) ** 0.5)
    r = base_result("flight_window", "discrete")
    r["metrics"] = dict(flight_s=dur, apex_rise=apex)
    r["thresholds"] = dict(apex_rise_ceil=round(apex_ceil, 2), flight_s_ceil=round(flight_ceil, 2))
    r["passed"] = dur <= flight_ceil and apex <= apex_ceil
    r["reward"] = 0.0 if r["passed"] else -1.0
    r["feedback"] = ("Aerial phase plausible." if r["passed"]
                     else f"Implausible aerial: {dur:.1f} s (ceil {flight_ceil:.1f}), "
                          f"apex RISE {apex:.2f} m (ceil {apex_ceil:.2f}).")
    return r


def eval_object_continuity(traj, spec, dt, v_max=20.0):
    """GATED on scene objects. A carried/animated object's pose must be C0 (no
    teleport between frames). fps-INDEPENDENT: it thresholds a VELOCITY (v_max, m/s)
    via dt, not a fixed per-frame step, so the same motion no longer passes at
    100 fps and fails at 25 fps. v_max is generous — a real throw is a few m/s while
    a grip/release SNAP is tens of m/s, so this catches teleports without capping
    legitimate speed."""
    objs = traj.get("object_pos") or {}
    if not objs:
        return _skip("object_continuity", "No scene objects.")
    max_step = float(v_max) * float(dt)
    worst = max(float(np.linalg.norm(np.diff(np.asarray(p), axis=0), axis=1).max())
                for p in objs.values())
    r = base_result("object_continuity", "discrete")
    r["metrics"] = dict(max_step=worst, max_step_allowed=round(max_step, 3))
    r["passed"] = worst <= max_step
    r["reward"] = 0.0 if r["passed"] else -1.0
    r["feedback"] = ("Object motion continuous." if r["passed"]
                     else f"Object TELEPORTS ({worst*100:.0f} cm in one frame ~ "
                          f"{worst/dt:.0f} m/s, over the {v_max:.0f} m/s ceiling).")
    return r


def eval_object_unheld_no_rise(traj, spec, dt=None, grip_offset=None, rise_tol=0.05,
                               max_frac=0.03, g=9.81, accel_margin=4.0):
    """GATED on scene objects. GENERAL PHYSICS LAW, the object-side companion to
    `support_or_ballistic`: an object that NOTHING is holding may fall, rest or be
    carried — it may NOT gain height on its own.

    This was the one levitation nothing forbade. `object_at_rest` only rules out
    SPAWNING mid-air (its docstring explicitly declines to forbid floating-unheld);
    `object_continuity` only rules out a velocity discontinuity, so a SMOOTH rise
    passes it; `object_grip` measures the hold gap but is satisfied once the object
    and limb coincide, however they came to. So a skill could animate a crate
    upward through empty space and every check passed. Measured on the carry-box:
    the hands bottomed out 33 cm above the crate, and the crate rose from z=0.12 to
    0.64 to meet them — self-powered flight, recorded as ground truth training data.

    HELD is derived from the trajectory, not from names or declarations: the object
    is held on a frame if SOME end-effector is within (its own half-extent + a
    grip offset) of it. So a genuine carry is unaffected, and only an object rising
    with no limb anywhere near it fails.

    BALLISTIC RISES ARE LEGAL. An unheld object may fall, rest, OR FLY — a thrown
    ball, a kicked can and a tossed crate all climb with nothing touching them, and
    forbidding that would make the FRAMEWORK the limit on what a user can ask for
    rather than the physics. So the law is not "must not rise" but "must not climb
    UNDER POWER": while unheld and rising, the object's vertical acceleration must
    be gravity (≈ −g). A ballistic arc passes (a ≈ −9.81); a crate carried upward at
    a steady speed (a ≈ 0) or accelerating upward does not. Exactly the discriminator
    `support_or_ballistic` applies to the robot's own base."""
    objs = traj.get("object_pos") or {}
    if not objs:
        return _skip("object_unheld_no_rise", "No scene objects.")
    ee = _ee_pos(traj)
    if ee is None or not len(ee):
        return _skip("object_unheld_no_rise", "No end-effector data.")
    # HOLDERS ARE END-EFFECTORS ONLY. The recorded roster is the UNION (EEs + contact
    # bodies: knees, thighs, torso, head), and a knee cannot grip anything — counting
    # one as a holder silently excuses the exact levitation this law exists to catch
    # (measured: the crate sat at z=0.12 with the nearest body a KNEE 0.226 m away,
    # inside tolerance, so the frame read as "held" and the crate's 0.12 -> 0.50 m
    # jump to meet a hand went unflagged).
    names = list(traj.get("ee_names") or spec.ee_names)
    keep = [i for i, n in enumerate(names) if n in set(spec.ee_names)]
    if keep and len(keep) < ee.shape[1]:
        ee = ee[:, keep, :]
    extents = traj.get("object_extents") or {}
    if grip_offset is None:
        r_limb = traj.get("robot_limb_radius")
        grip_offset = 2.5 * float(r_limb) if r_limb else 0.12
    _dt = float(dt) if dt else (1.0 / float(traj.get("fps") or 200.0))
    r = base_result("object_unheld_no_rise", "discrete")
    worst, worst_name, n_bad, total = 0.0, None, 0, 0
    for name, p in objs.items():
        p = np.asarray(p, dtype=float)
        if p.ndim != 2 or len(p) < 3:
            continue
        tol = float(extents.get(name, 0.0)) + grip_offset
        d = np.linalg.norm(ee - p[:, None, :3], axis=2)      # (T, n_ee)
        held = d.min(axis=1) <= tol
        dz = np.diff(p[:, 2])
        # rising on a frame where NOTHING was holding it at either end of the step.
        # The 1e-4 floor keeps per-frame numerical jitter from accumulating into a
        # phantom climb over a long clip.
        unheld = ~(held[:-1] | held[1:])
        rising = unheld & (dz > 1e-4)
        # ...and EXEMPT a ballistic arc: a thrown/launched object legitimately climbs
        # with nothing on it, so only a rise that is NOT free flight counts.
        az = np.gradient(np.gradient(p[:, 2], _dt), _dt)          # d2z/dt2, per frame
        powered = az[:-1] > (-g + accel_margin)                   # a >> -g  =>  supported
        bad = rising & powered
        total += len(dz)
        if bad.any():
            # cumulative unheld rise, not a single step: a slow levitation is the case
            climb = float(dz[bad].sum())
            n_bad += int(bad.sum())
            if climb > worst:
                worst, worst_name = climb, name
    if total == 0:
        return _skip("object_unheld_no_rise", "No usable object tracks.")
    # CUMULATIVE RISE is the whole criterion; a frame budget would defeat the law,
    # because a teleport is by definition a large jump over FEW frames (measured:
    # 11 cm of unheld climb on 6 frames passed a 65-frame budget). `rise_tol`
    # already absorbs noise, so no second escape hatch.
    budget = max(3, int(max_frac * total))
    passed = worst <= rise_tol
    r["metrics"] = dict(max_unheld_rise=round(worst, 3), object=worst_name,
                        num_frames=n_bad, frame_budget=budget)
    r["thresholds"] = dict(rise_tol=rise_tol, max_frames=budget)
    r["passed"] = bool(passed)
    r["reward"] = 0.0 if passed else -1.0
    r["feedback"] = (
        f"No unheld object gained height (worst {worst*100:.1f} cm)." if passed else
        f"OBJECT LEVITATES — '{worst_name}' rises {worst*100:.0f} cm over {n_bad} frames while "
        f"NO end-effector is close enough to be holding it, and it is NOT in free flight (its "
        f"vertical acceleration is not gravity — something is lifting it). An unheld object may "
        f"fall, rest or fly ballistically, never climb under power. The limbs never reached it and "
        f"the object was animated "
        f"up to meet them: bring the LIMB to the object (crouch lower / step nearer so the hand "
        f"reaches the grip point), then move them together — do not author the object's rise.")
    return r


def eval_object_grip(traj, spec, grip_offset=None, lift_thresh=0.08):
    """GATED on scene objects. A lifted object must be genuinely PICKED UP, not
    teleported onto a raised limb. Two general, morphology-agnostic conditions
    (the HOLDER is derived from the trajectory — the end-effector the object rides
    with while lifted — so no name-matching, and it is NOT fooled by an attachment
    that trivially pins the object to a hand):
      (1) HELD: while lifted, the object stays coincident with its holder.
      (2) PICKED UP: the holder actually REACHED the object at its resting spot
          before it left the ground — so the limb came to the object, rather than
          the object jumping up to a limb that never bent/reached down to it.

    Distances are OBJECT-RELATIVE: the hold tolerance is the object's own half-extent
    (recorded) plus a grip offset (wrist-origin -> grip point + grip radius). An
    absolute 0.40 m wrongly FAILED a legal one-hand carry of a large box (whose hand
    sits half_size + wrist off the centre) and was vacuous on a small robot."""
    objs = traj.get("object_pos") or {}
    if not objs:
        return _skip("object_grip", "No scene objects.")
    ee = _ee_pos(traj)
    extents = traj.get("object_extents") or {}
    if grip_offset is None:
        # wrist-origin -> grip point scales with hand/limb size; the recorded
        # limb_radius proxies it (2.5x median limb radius ~ 0.12 m on the humanoid,
        # shrinks on a small robot) — no absolute human-hand constant. Combined with
        # the per-object half-extent below, the hold tolerance is fully robot+object
        # relative. Fallback 0.12 m when no radius was recorded (older trajectory).
        r_limb = traj.get("robot_limb_radius")
        grip_offset = 2.5 * float(r_limb) if r_limb else 0.12
    worst_hold, worst_pickup = 0.0, 0.0
    hold_fail = pickup_fail = any_lift = False
    for name, p in objs.items():
        p = np.asarray(p)
        lifted = p[:, 2] > p[:, 2].min() + lift_thresh
        if not lifted.any():
            continue
        any_lift = True
        # hold tolerance = this object's half-extent + the grip offset (per object).
        hold_dist = float(extents.get(name, 0.12)) + grip_offset
        li = np.where(lifted)[0]
        # holder SET = the end-effectors that ride with the object while lifted
        # (within hold_dist — captures BOTH hands of a two-hand grip, each a grip
        # radius off the object's centre). Feet, far away, are excluded.
        dmean = np.linalg.norm(ee[li] - p[li][:, None, :], axis=2).mean(axis=0)  # (N,)
        holders = np.where(dmean < hold_dist)[0]
        if len(holders) == 0:
            holders = np.array([int(dmean.argmin())])
        cen = ee[:, holders, :].mean(axis=1)   # (T,3) centroid of the holders — the grip point
        # HELD: the object stays at the holders' grip point while lifted
        h = float(np.linalg.norm(p[li] - cen[li], axis=1).max())
        worst_hold = max(worst_hold, h)
        if h > hold_dist:
            hold_fail = True
        # PICKED UP: the grip point REACHED the object's resting spot (its pose just
        # before it first left the ground), i.e. the limbs came DOWN to the object
        t0 = int(li[0])
        rest_pos = p[max(0, t0 - 1)]
        pu = float(np.linalg.norm(cen[:t0 + 1] - rest_pos, axis=1).min())
        worst_pickup = max(worst_pickup, pu)
        if pu > hold_dist:
            pickup_fail = True
    if not any_lift:
        return _skip("object_grip", "No object is lifted.")
    r = base_result("object_grip", "discrete")
    r["metrics"] = dict(max_hold_gap=worst_hold, pickup_reach=worst_pickup)
    held_ok, picked_ok = (not hold_fail), (not pickup_fail)
    r["passed"] = held_ok and picked_ok
    r["reward"] = 0.0 if r["passed"] else -1.0
    if r["passed"]:
        r["feedback"] = "Object reached at its resting spot and held while lifted."
    elif not picked_ok:
        r["feedback"] = (f"Object NOT picked up from its resting spot — the holding limb "
                         f"never got closer than {worst_pickup*100:.0f} cm to it. The object "
                         f"teleported up to a raised limb; the robot must BEND/REACH the limb "
                         f"DOWN TO the object and grasp it, not have it jump to the hand.")
    else:
        r["feedback"] = f"Object FLOATS off its holder ({worst_hold*100:.0f} cm) while lifted."
    return r


def eval_object_at_rest(traj, spec, tol=0.06):
    """GATED on scene objects. General law: an object does not SPAWN in mid-air — it
    begins at rest on its surface (lowest z). The END state is skill-specific (a
    place ends at rest, a hold ends lifted, a throw ends in flight), so it is NOT
    asserted here — object_continuity already forbids teleporting, and object_grip
    forbids floating-unheld."""
    objs = traj.get("object_pos") or {}
    if not objs:
        return _skip("object_at_rest", "No scene objects.")
    bad = [f"{name} starts lifted" for name, p in objs.items()
           if np.asarray(p)[0, 2] > np.asarray(p)[:, 2].min() + tol]
    r = base_result("object_at_rest", "discrete")
    r["passed"] = not bad
    r["reward"] = 0.0 if r["passed"] else -1.0
    r["feedback"] = ("Object starts at rest on its surface." if r["passed"]
                     else "Object spawns mid-air: " + "; ".join(bad))
    return r


def eval_object_penetration(traj, spec, tol=0.02, max_depth=0.10, max_frac=0.05):
    """GATED on scene objects. A CARRIED object must not clip THROUGH the robot's
    body. `surface_penetration` EXCLUDES mocap manipulands (the hands are meant to
    grip them), so a box held against/inside the torso would otherwise pass every
    check — this is the only test that catches it. Reads the recorded
    `object_penetration` (deepest carried-object-vs-robot-body overlap; the gripping
    hands + forearms are excluded in the sim). Deep OR sustained overlap -> fail."""
    op = traj.get("object_penetration")
    objs = traj.get("object_pos") or {}
    if op is None or not objs:
        return _skip("object_penetration", "No carried object / no penetration data.")
    # Only a CARRIED (lifted) object clipping the body is a fault. A box resting on
    # the floor while the robot crouches OVER it to pick/place legitimately has limbs
    # (shins/knees) right beside it — that is not a carry-through-torso. Count the
    # penetration budget ONLY on frames where SOME object is lifted clear of its rest
    # height; zero it on rest frames. (General: uses only the recorded object path.)
    op = np.asarray(op, dtype=float).copy()
    lifted = np.zeros(len(op), dtype=bool)
    for _name, p in objs.items():
        p = np.asarray(p, dtype=float)
        if p.ndim == 2 and p.shape[0] == len(op) and p.shape[1] >= 3:
            lifted |= (p[:, 2] - float(p[:, 2].min())) > 0.08   # >8 cm above its lowest (rest)
    if lifted.any():
        op = np.where(lifted, op, 0.0)
    passed, deepest, n, budget = _deep_or_sustained(op, tol, max_depth, max_frac)
    r = base_result("object_penetration", "discrete")
    r["metrics"] = dict(max_penetration=deepest, num_frames=n, frame_budget=budget)
    r["thresholds"] = dict(tol=tol, max_depth=max_depth, max_frames=budget)
    r["passed"] = passed
    r["reward"] = 0.0 if passed else -1.0
    if passed:
        r["feedback"] = (f"Carried object stays clear of the body (deepest overlap "
                         f"{deepest*100:.1f} cm — within tolerance).")
    else:
        why = "too DEEP" if deepest > max_depth else "too SUSTAINED"
        r["feedback"] = (f"The carried OBJECT clips THROUGH the robot's body ({why}: up to "
                         f"{deepest*100:.0f} cm on {n} frames). Hold it FURTHER out in front so its "
                         f"near face clears the chest/pelvis — carry it at arm's reach ahead of the "
                         f"torso, not pinned against the body.")
    return r


def eval_clip_duration(traj, spec, dt, min_s=2.0):
    """GENERAL law: the RECORDED clip must be long enough to be usable motion data.
    A sub-2-second recording is a snippet, not a demonstration — however correct each
    pose in it may be. The lever is duration_s (and `repetitions` for a cyclic skill)."""
    r = base_result("clip_duration", "discrete")
    T = len(traj.get("root_pos", []))
    secs = T * float(dt) if dt else 0.0
    r["metrics"] = dict(clip_seconds=round(secs, 2), frames=T)
    r["thresholds"] = dict(min_seconds=min_s)
    r["passed"] = secs >= min_s
    r["reward"] = 0.0 if r["passed"] else -1.0
    r["feedback"] = (f"Clip is {secs:.2f} s of motion."
                     if r["passed"] else
                     f"CLIP TOO SHORT: only {secs:.2f} s recorded (minimum {min_s:.0f} s). This is a "
                     f"snippet, not a demonstration. Raise duration_s so the whole action plays at a "
                     f"natural pace (and `repetitions` for a cyclic skill). Do NOT fix it by slowing "
                     f"one cycle down — that drags each stance past its foothold.")
    return r


def build_gated_evaluators(traj, spec, dt):
    """
    Tests whose DOMAIN is derived from a PHYSICAL declaration in the recorded
    trajectory (a []-contact flight window, scene objects present) — never from an
    editorial skill label. Each self-skips when out-of-domain, so an out-of-domain
    test can never fail the skill and never drives the repair loop.
    (foot_clearance was removed: "a support foot must lift" is a stepping-gait
    assumption, not a general law — a static balance/squat legitimately keeps its
    feet planted, and a dragging foot is already caught by contact_no_slide.)
    """
    return [
        lambda t: eval_flight_window(t, spec, dt),     # domain: a [] contact segment
        lambda t: eval_object_continuity(t, spec, dt),  # domain: scene objects present
        lambda t: eval_object_grip(t, spec),           # domain: scene objects present
        lambda t: eval_object_at_rest(t, spec),        # domain: scene objects present
        lambda t: eval_object_unheld_no_rise(t, spec, dt), # domain: unheld object climbing under power
        lambda t: eval_object_penetration(t, spec),    # domain: object clips through the body
    ]


# =====================================================
#     AGENT-DECLARED FIDELITY INVARIANTS (planner-owned)
# =====================================================
#
# The planner declares, per skill, a handful of measurable checks composed from
# GENERAL geometric primitives (below) — NOT skill heuristics. The primitives are
# universal geometric relations; the SKILL-SPECIFICITY is entirely in WHICH ones
# the planner picks and with what parameters (exactly like the control-channel
# decision). These are "does it look like the intended motion?" checks — the
# fidelity layer the pass/fail physics floor cannot express. They rank versions
# among floor-passing ones; they never override a floor (physics) failure.
#
# GOVERNANCE: invariants are PLANNER-owned. The critic-coder must never add,
# remove, or edit them (that is Goodhart — the repairer deleting the test it
# fails). If diagnosis concludes an invariant is itself wrong, the loop routes to
# the planner (replan edge), which re-declares with the readout in hand.

def _contact_onset_phase(cm, i, T):
    edges = np.where((~cm[:-1, i]) & (cm[1:, i]))[0]
    return float((edges[0] + 1) / (T - 1)) if edges.size else None


def _lift_onset_phase(cl, i, T, lift_eps=0.03):
    """First phase at which EE i LIFTS off its surface (clearance rises past lift_eps).
    Used as the alternation-event fallback for a CONTINUOUS-contact gait (a moonwalk's
    feet never break contact, so contact onsets don't exist — but the heel-lift/role
    change does)."""
    lifted = np.asarray(cl[:, i], float) > lift_eps
    edges = np.where((~lifted[:-1]) & (lifted[1:]))[0]
    return float((edges[0] + 1) / (T - 1)) if edges.size else None


def _fidelity_result(name, passed, measured, target, feedback):
    r = base_result(name, "fidelity")
    r["passed"] = bool(passed)
    r["reward"] = 0.0 if passed else -0.5     # softer than a physics-law failure
    r["metrics"] = dict(measured=measured, target=target)
    r["feedback"] = feedback
    return r


def _inv_net_progress(traj, spec, dt, p):
    """Net base travel along an axis (end minus start). axis 'z' is the right
    primitive for a CLIMB — a stair/ledge ascent starts LOW and ends high, so its
    net vertical rise is well-defined, whereas a fixed base_height_band can't
    express it (the base legitimately begins below any 'on top' band)."""
    root = np.asarray(traj["root_pos"], dtype=float)
    axis = p.get("axis", "horizontal")
    d = root[-1] - root[0]
    if axis == "x":
        measured = float(d[0])
    elif axis == "y":
        measured = float(d[1])
    elif axis == "z":
        measured = float(d[2])                 # net vertical rise (climb height)
    else:
        measured = float(np.linalg.norm(d[:2]))
    lo, hi = p.get("min", 0.0), p.get("max", 1e9)
    ok = lo - 0.05 <= measured <= hi + 0.05
    return ok, round(measured, 3), [lo, hi], (
        f"base net {axis} displacement {measured*100:.0f} cm "
        f"(want {lo*100:.0f}..{hi*100:.0f}).")


def _inv_swing_lift(traj, spec, dt, p):
    ee_names = traj.get("ee_names") or spec.ee_names
    cm = traj.get("contact_mask"); cl = traj.get("surface_clearance")
    if cm is None or cl is None or p.get("ee") not in ee_names:
        return None
    i = ee_names.index(p["ee"])
    cm = np.asarray(cm, bool); cl = np.asarray(cl, float)
    swing = ~cm[:, i]
    measured = float(cl[swing, i].max()) if swing.any() else 0.0
    lo, hi = p.get("min", 0.0), p.get("max", 1e9)
    ok = lo - 0.01 <= measured <= hi + 0.02
    return ok, round(measured, 3), [lo, hi], (
        f"{p['ee']} peak swing lift {measured*100:.1f} cm (want {lo*100:.0f}..{hi*100:.0f}).")


def _inv_phase_offset(traj, spec, dt, p):
    ee_names = traj.get("ee_names") or spec.ee_names
    cm = traj.get("contact_mask")
    if cm is None or p.get("ee_a") not in ee_names or p.get("ee_b") not in ee_names:
        return None
    cm = np.asarray(cm, bool); T = len(cm)
    ia, ib = ee_names.index(p["ee_a"]), ee_names.index(p["ee_b"])
    a = _contact_onset_phase(cm, ia, T)
    b = _contact_onset_phase(cm, ib, T)
    src = "contact-onset"
    if a is None or b is None:
        # Continuous-contact gait (a moonwalk's feet never break contact -> no onset
        # edges): fall back to HEEL-LIFT onsets (both limbs, same event type). Without
        # this the invariant is UNMEASURABLE and (post-A2) fails as "no alternation" —
        # this recovers a real alternation measurement so a genuinely-alternating gait
        # can pass and a both-feet-skate FAILS on the offset.
        cl = traj.get("surface_clearance")
        if cl is not None:
            cl = np.asarray(cl, float)
            a = _lift_onset_phase(cl, ia, T); b = _lift_onset_phase(cl, ib, T)
            src = "heel-lift"
    if a is None or b is None:
        return None
    d = abs(a - b); d = min(d, 1 - d)
    # The measured offset d is UNSIGNED cyclic distance in [0, 0.5]. A planner that
    # declares a signed expected (e.g. -0.25 for "antiphase, B leads") is then NEVER
    # satisfiable (|d-(-0.25)| >= 0.25 > tol always) and misdirects the repair loop.
    # Normalize expected the same way the metric is normalized.
    exp, tol = p.get("expected", 0.5), p.get("tol", 0.15)
    exp = min(abs(exp), 1 - abs(exp))
    ok = abs(d - exp) <= tol
    return ok, round(d, 3), exp, (
        f"{p['ee_a']}/{p['ee_b']} {src} offset {d:.2f} (want {exp:.2f}±{tol:.2f}).")


def _inv_base_height_band(traj, spec, dt, p):
    z = np.asarray(traj["root_pos"], float)[:, 2]
    lo, hi = p.get("min"), p.get("max")
    if lo is None or hi is None:
        # A paramless band used to inherit base_height_range as an ABSOLUTE z cap —
        # which false-fails a legitimate TALL climb (a bigger slab lifts the pelvis
        # above the range top). A band with no bounds is a BAD DECLARATION, not a
        # physics failure: raise so eval_declared_invariants SKIPS it (never fail a
        # skill for the planner's malformed invariant). The band stays agent-declared
        # and floor-absolute ONLY when the planner gives real bounds.
        raise ValueError("base_height_band declared without min/max — cannot evaluate")
    zlo, zhi = float(z.min()), float(z.max())
    ok = zlo >= lo - 0.05 and zhi <= hi + 0.05
    return ok, [round(zlo, 3), round(zhi, 3)], [lo, hi], (
        f"base height {zlo:.2f}..{zhi:.2f} m (want {lo:.2f}..{hi:.2f}).")


def _inv_object_coincident(traj, spec, dt, p):
    objs = traj.get("object_pos") or {}
    if not objs:
        return None
    ee = _ee_pos(traj)
    ee_names = traj.get("ee_names") or spec.ee_names
    extents = traj.get("object_extents") or {}
    name = p.get("object")
    items = [(name, objs[name])] if name in objs else list(objs.items())
    max_dist = p.get("max_dist", 0.25)
    worst = 0.0
    any_lifted = False
    for nm, pos in items:
        pos = np.asarray(pos, float)
        lifted = pos[:, 2] > pos[:, 2].min() + 0.08
        if not lifted.any():
            continue
        any_lifted = True
        if p.get("ee") in ee_names:
            d = np.linalg.norm(ee[lifted, ee_names.index(p["ee"]), :] - pos[lifted], axis=1)
        else:                                   # nearest EE at each lifted frame
            d = np.linalg.norm(ee[lifted] - pos[lifted][:, None, :], axis=2).min(axis=1)
        # Measure to the object's nearest SURFACE, not its CENTER: a legitimate
        # side-face grip holds the hand one HALF-EXTENT from the center by construction
        # (a 0.24 m box gripped on its face reads 0.12 m to center forever), so
        # subtracting the half-extent is what makes "box rides with the hand" satisfiable.
        half = float(extents.get(nm, 0.0))
        d = np.maximum(0.0, d - half)
        worst = max(worst, float(d.max()))
    if not any_lifted:
        return None                                   # object never lifted -> "held" phenomenon absent (fails via A2)
    ok = worst <= max_dist
    return ok, round(worst, 3), max_dist, (
        f"held object surface strays {worst*100:.0f} cm from its holder (want <= {max_dist*100:.0f}).")


def _inv_end_state(traj, spec, dt, p):
    root = np.asarray(traj["root_pos"], float)
    # ACTUATED joints only — a passive continuous-rotation wheel (go2w) rolls freely
    # over a cyclic gait, so including it makes the joint-return delta ~its total roll
    # (12+ rad) and the cyclic invariant could NEVER pass on a wheeled robot.
    joints = _actuated(np.asarray(traj["joints"], float), spec)
    adv = float(np.linalg.norm(root[-1, :2] - root[0, :2]))
    mode = p.get("mode", "advances")
    if mode == "returns":                       # cyclic: end pose ~= start pose
        # Split the return into BASE (metres) and JOINTS (radians, per-joint MAX)
        # instead of one mixed-unit L2 norm — the old seam added metres and radians
        # and its magnitude GREW with sqrt(DoF), so a perfect return failed on a
        # high-DoF robot (28 joints: an avg 0.057 rad/joint mismatch alone hit 0.30).
        base_ret = float(np.linalg.norm(root[-1] - root[0]))
        joint_ret = float(np.max(np.abs(joints[-1] - joints[0]))) if joints.shape[1] else 0.0
        base_tol = float(p.get("base_tol", 0.10))     # m
        joint_tol = float(p.get("joint_tol", 0.15))   # rad (per-joint max)
        ok = base_ret < base_tol and joint_ret < joint_tol
        return ok, round(max(base_ret, joint_ret), 3), f"base<{base_tol}m, joint<{joint_tol}rad", (
            f"cyclic return: base moved {base_ret*100:.0f} cm (want <{base_tol*100:.0f}), "
            f"worst joint {joint_ret:.2f} rad (want <{joint_tol:.2f}).")
    ok = adv >= p.get("min_advance", 0.10)       # one-shot: distinct end / travel
    return ok, round(adv, 3), p.get("min_advance", 0.10), (
        f"base advanced {adv*100:.0f} cm to a distinct end (want >= {p.get('min_advance',0.10)*100:.0f}).")


def _inv_stance_no_drift(traj, spec, dt, p):
    ee_names = traj.get("ee_names") or spec.ee_names
    if p.get("ee") not in ee_names:
        return None
    sub = eval_contact_no_slide(traj, spec, dt, max_drift=p.get("max_drift", 0.10))
    return None if sub.get("skipped") else (
        sub["passed"], sub["metrics"].get("max_stance_drift", 0.0),
        p.get("max_drift", 0.10), sub["feedback"])


def _inv_peak_rise(traj, spec, dt, p):
    """Peak base EXCURSION along an axis relative to its START — the right primitive
    for a motion that rises/dips then RETURNS (a jump's apex, a handstand's inversion,
    a squat's dip), where net_progress (end-start) is ~0 because it comes back, and a
    whole-trajectory base_height_band can't express it (the band is violated by the
    very excursion + the standing bookends). Direction from the band sign: a POSITIVE
    band [min,max] measures max(axis)-start (an upward rise); a band with max<=0
    measures min(axis)-start (a downward dip, reported negative). Optional
    `phase_window` [a,b] restricts the extremum to that phase span."""
    root = np.asarray(traj["root_pos"], float)
    T = len(root)
    axis = {"x": 0, "y": 1, "z": 2}.get(p.get("axis", "z"), 2)
    a = root[:, axis]
    w = p.get("phase_window")
    if isinstance(w, (list, tuple)) and len(w) == 2:
        i0 = max(0, int(float(w[0]) * (T - 1)))
        i1 = min(T, int(float(w[1]) * (T - 1)) + 1)
        seg = a[i0:max(i0 + 1, i1)]
    else:
        seg = a
    start = float(a[0])
    lo, hi = float(p.get("min", 0.0)), float(p.get("max", 1e9))
    measured = (float(seg.min()) - start) if hi <= 0 else (float(seg.max()) - start)
    ok = lo - 0.05 <= measured <= hi + 0.05
    return ok, round(measured, 3), [lo, hi], (
        f"base peak {p.get('axis', 'z')} excursion {measured*100:+.0f} cm from start "
        f"(want {lo*100:.0f}..{hi*100:.0f}).")


def _inv_contact_over_surface(traj, spec, dt, p):
    """A named contact body must end up HORIZONTALLY OVER a surface's footprint.

    The missing half of every "put X on Y" goal. A HEIGHT band alone is satisfiable
    without ever going near the thing: measured on a sit-chair, the pelvis reached
    exactly seat-height (z=0.56 for a 0.45 m seat + 0.09 m offset) while standing
    0.98 m along x — 19 cm IN FRONT of a seat spanning [1.17, 1.63] — and declared
    ZERO pelvis contacts. It scored a clean physics floor for squatting in mid-air
    next to the chair. Height says how low, this says where.

    params: {ee, x_range: [lo,hi], y_range: [lo,hi] (optional), phase_window: [a,b]}
    """
    ee_names = traj.get("ee_names") or spec.ee_names
    name = p.get("ee")
    if name not in ee_names:
        return None
    pos = traj.get("ee_positions_achieved")
    if pos is None:
        return None
    pos = np.asarray(pos, float)
    i = list(ee_names).index(name)
    T = len(pos)
    w = p.get("phase_window") or [0.0, 1.0]
    a, b = int(max(0, w[0] * T)), int(min(T, max(1, w[1] * T)))
    if b <= a:
        return None
    seg = pos[a:b, i, :]
    xr = p.get("x_range"); yr = p.get("y_range")
    # signed distance OUTSIDE the footprint (0 = inside), worst over the window
    def _out(v, rng):
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
            return np.zeros(len(v))
        return np.maximum(np.maximum(rng[0] - v, v - rng[1]), 0.0)
    out = np.maximum(_out(seg[:, 0], xr), _out(seg[:, 1], yr))
    worst = float(out.min())        # BEST frame in the window: did it EVER get over it?
    tol = float(p.get("tol", 0.05))
    # NOTE: build the verdict OUTSIDE the f-string. A multi-line implicit string
    # concatenation inside an f-string expression only parses on Python 3.12+
    # (PEP 701); this project runs on 3.11, where it is a SyntaxError.
    verdict = ("over it" if worst <= tol else
               "NEVER over it, so it never landed ON the surface; move the BASE so the "
               "body is above the footprint, not merely at its height")
    ybit = "" if yr is None else " y" + str(yr)
    return (worst <= tol, round(worst, 3), tol,
            (f"'{name}' comes within {worst*100:.0f} cm of the target footprint "
             f"(x{xr}{ybit}) during phase {w} — {verdict}."))


_INVARIANT_KINDS = {
    "contact_over_surface": _inv_contact_over_surface,
    "net_progress": _inv_net_progress,
    "peak_rise": _inv_peak_rise,
    "swing_lift": _inv_swing_lift,
    "phase_offset": _inv_phase_offset,
    "base_height_band": _inv_base_height_band,
    "object_coincident": _inv_object_coincident,
    "end_state": _inv_end_state,
    "stance_no_drift": _inv_stance_no_drift,
}


def eval_declared_invariants(traj, spec, dt, invariants):
    """Run the planner-declared fidelity invariants. Robust: an unknown kind, a
    bad param, or missing data self-skips (never crashes the pipeline, never
    fails a skill for a malformed declaration)."""
    results = []
    for inv in (invariants or []):
        if not isinstance(inv, dict):
            continue
        kind = inv.get("kind")
        fn = _INVARIANT_KINDS.get(kind)
        name = f"fidelity:{inv.get('name', kind or 'unknown')}"
        if fn is None:
            results.append(_skip(name, f"Unknown invariant kind '{kind}'."))
            continue
        try:
            out = fn(traj, spec, dt, inv.get("params", {}) or {})
        except Exception as e:
            results.append(_skip(name, f"Invariant errored ({e})."))
            continue
        if out is None:
            # UNMEASURABLE ≠ passed. A well-formed invariant whose PHENOMENON is absent
            # (a phase_offset on feet that never break contact; a held-object invariant
            # whose object never lifts) used to skip-as-green — so a motion missing its
            # DEFINING feature scored full fidelity (the moonwalk that never alternated).
            # Fail it (the defining phenomenon is absent → the motion is wrong, or the
            # invariant is mis-declared → replan). Malformed declarations (unknown kind,
            # exceptions above) still skip — we never fail a skill for a bad declaration.
            results.append(_fidelity_result(
                name, False, None, None,
                "UNMEASURABLE: the phenomenon this invariant checks never occurred in the "
                "recorded motion (no contact break / heel-lift / object-lift, or the named "
                "EE/object never engaged). The motion does not exhibit the invariant's subject "
                "— either the motion is wrong, or the invariant names the wrong target (replan)."))
            continue
        ok, measured, target, fb = out
        why = inv.get("why", "")
        results.append(_fidelity_result(name, ok, measured, target,
                                        (fb + (f"  [{why}]" if why else ""))))
    return results


# =====================================================
#          MASTER EVALUATOR
# =====================================================

def evaluate_trajectory(
    traj,
    spec,
    dt,
    invariants=None,
    sliding_contacts=None,
    pivot_contacts=None,
    cyclic=None,
) -> Dict[str, Any]:
    """
    Args:
        traj : dict with root_pos (T,3), root_quat (T,4), joints (T,J),
               ee_positions (T,N,3), and optionally ee_names
        spec : RobotSpec (morphology: passive joints, joint limits,
               ground offsets, base height range)
        dt   : timestep (s)
        invariants : planner-declared fidelity invariants (list of dicts) — the
               "does it look like the intended motion?" layer (ranks versions,
               never overrides a physics-floor failure)
        sliding_contacts : EE names the planner declared as legitimately sliding —
               exempted from the contact_no_slide law (fail-closed: empty default)
    """

    _cg_kw = {}

    # ---- PHYSICS FLOOR (universal laws; always on, drive the repair loop) ----
    # base_height_envelope is DEMOTED to an agent-declared `base_height_band`
    # invariant (a crouch/crawl/jump legitimately leaves any fixed band; its one
    # real job — hover/burial — is already covered by support + penetration).
    # self_collision is now SOFT (gross-only). contact_no_slide stays a hard law
    # (planted-default, declared-sliding opt-out).
    evaluators = [
        lambda t: eval_surface_penetration(t, spec),        # matter through matter
        lambda t: eval_self_collision(t, spec),             # SOFT: gross-only, discardable
        lambda t: eval_support_or_ballistic(t, spec, dt),   # declaration honesty gate (SOME contact rests)
        lambda t: eval_contact_no_slide(t, spec, dt, exempt_names=sliding_contacts,
                                        pivot_names=pivot_contacts),
        lambda t: eval_joint_limits(t, spec),               # model ranges
        lambda t: eval_no_teleport(t, spec, dt),            # no single-frame joint step
        lambda t: eval_ee_no_teleport(t, spec, dt),         # no single-frame EE-space jump (free-limb flicker)
        lambda t: eval_joint_velocity_envelope(t, spec, dt),  # no sustained over-speed
        lambda t: eval_framework_interventions(t, spec, dt),  # a SATURATED silent correction => authoring is impossible
        lambda t: eval_clip_duration(t, spec, dt),           # a sub-2s recording is a snippet, not data
    ]
    evaluators.insert(3, lambda t: eval_contact_grounded(
        t, spec, dt, exempt_names=sliding_contacts, **_cg_kw))
    # ---- SKILL-SPECIFIC / GATED (intent; enabled by declaration, may be skipped) ----
    evaluators += build_gated_evaluators(traj, spec, dt)

    results = []
    total_reward = 0.0

    for fn in evaluators:
        r = fn(traj)
        total_reward += r["reward"]
        results.append(r)

    # ---- AGENT-DECLARED FIDELITY (ranks versions; not part of the physics floor) ----
    fidelity = eval_declared_invariants(traj, spec, dt, invariants)
    for r in fidelity:
        if not r.get("skipped"):
            total_reward += r["reward"]
        results.append(r)

    discrete_summary = aggregate_discrete(results)

    fid = [r for r in results if r["category"] == "fidelity" and not r["skipped"]]
    fid_passed = sum(1 for r in fid if r["passed"])

    return dict(
        summary=dict(
            passed_all=discrete_summary["passed"],
            passed_all_discrete=discrete_summary["passed"],
            total_reward=float(total_reward),
            num_tests=len(results),
            num_failed=len(discrete_summary["failed_tests"]),
            num_fidelity_passed=fid_passed,
            num_fidelity_total=len(fid),
            fidelity_failed=[r["name"] for r in fid if not r["passed"]],
        ),
        tests=results
    )


# =====================================================
#          SAVE REPORT
# =====================================================

def save_report(report: Dict[str, Any], out_dir: str, skill_name: str) -> str:
    return save_versioned_json(
        data=report,
        out_dir=out_dir,
        base_name=skill_name,
    )


def save_versioned_json(
    data: dict,
    out_dir: str,
    base_name: str,
    ext: str = ".json",
    pad: int = 4,
) -> str:
    """
    Saves data as base_name_XXXX.json in out_dir, where XXXX is the next version.
    Handles numpy types (bool, int, float) automatically.
    """
    os.makedirs(out_dir, exist_ok=True)

    pattern = re.compile(rf"{re.escape(base_name)}_(\d+){re.escape(ext)}$")
    versions = []

    for fname in os.listdir(out_dir):
        match = pattern.match(fname)
        if match:
            versions.append(int(match.group(1)))

    next_version = max(versions) + 1 if versions else 1
    fname = f"{base_name}_{next_version:0{pad}d}{ext}"
    file_path = os.path.join(out_dir, fname)

    # Convert numpy types to native Python
    def convert(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Cannot serialize object of type {type(o)}")

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, default=convert)

    return file_path
