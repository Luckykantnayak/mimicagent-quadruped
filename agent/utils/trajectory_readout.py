"""
Geometric trajectory readout — a compact, human/agent-readable summary of what a
recorded motion ACTUALLY DID (not whether it broke a law).

The repair loop's deepest flaw was that the critic never saw the motion — only a
vector of pass/fail legality lamps — so it local-patched and the checks fought
each other. This module turns the recorded eval pkl into a few paragraphs of
geometry (base path, per-limb contact timeline + lift + drift, gait phase offsets,
object path) that are fed into the diagnosis and critic prompts, so the agent can
reason about FIDELITY ("does this look like a crawl?") the way a human watching the
render does.

Pure geometry, no thresholds, no verdicts — it DESCRIBES, it does not judge.
"""

import numpy as np

REACH_SEAT_CAP_CM = 15.0   # mirrors controller.REACH_SEAT_CAP (m) — the cap the seat hits


def _quat_to_rpy(q):
    """(w,x,y,z) -> (roll, pitch, yaw) in radians, per-frame if q is (T,4)."""
    q = np.asarray(q, dtype=float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def _ee_pos(traj):
    ach = traj.get("ee_positions_achieved")
    return np.asarray(ach) if ach is not None else np.asarray(traj["ee_positions"])


def _runs(mask):
    """Yield (start, end) index pairs of maximal True runs in a bool array."""
    mask = np.asarray(mask, dtype=bool)
    T = len(mask)
    t = 0
    while t < T:
        if not mask[t]:
            t += 1
            continue
        t0 = t
        while t < T and mask[t]:
            t += 1
        yield t0, t


def _timeline(mask, buckets=24):
    """Coarse 'C/.' string of when a limb is in contact across the cycle."""
    mask = np.asarray(mask, dtype=bool)
    T = len(mask)
    if T == 0:
        return ""
    out = []
    for b in range(buckets):
        a = int(b * T / buckets)
        z = max(a + 1, int((b + 1) * T / buckets))
        out.append("C" if mask[a:z].mean() >= 0.5 else ".")
    return "".join(out)


def summarize_trajectory(traj, spec, dt) -> str:
    """Return a compact text readout of the recorded motion `traj`."""
    root = np.asarray(traj["root_pos"], dtype=float)
    T = len(root)
    if T < 2:
        return "Trajectory too short to summarize."
    ee_names = traj.get("ee_names") or spec.ee_names
    dur = (T - 1) * dt
    lines = []

    # ---------------- BASE ----------------
    xy = root[:, :2]
    net = xy[-1] - xy[0]
    net_dist = float(np.linalg.norm(net))
    path_len = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    z = root[:, 2]
    roll, pitch, yaw = _quat_to_rpy(traj["root_quat"])
    # net yaw turned (unwrapped, handles a full spin)
    yaw_turned = float(np.rad2deg(np.unwrap(yaw)[-1] - np.unwrap(yaw)[0]))
    lines.append("BASE:")
    lines.append(f"  duration {dur:.2f}s ({T} frames)")
    lines.append(f"  net horizontal displacement {net_dist*100:.0f} cm "
                 f"(dx={net[0]*100:+.0f}, dy={net[1]*100:+.0f}); path length {path_len*100:.0f} cm")
    lines.append(f"  height: min {z.min():.2f} / mean {z.mean():.2f} / max {z.max():.2f} m")
    lines.append(f"  pitch {np.rad2deg(pitch.min()):+.0f}..{np.rad2deg(pitch.max()):+.0f} deg, "
                 f"roll {np.rad2deg(roll.min()):+.0f}..{np.rad2deg(roll.max()):+.0f} deg, "
                 f"net yaw turned {yaw_turned:+.0f} deg")
    # cyclic-ness via a UNIT-SEPARATED seam (base metres vs ACTUATED-joint radians),
    # matching _inv_end_state + recorder.loop_mode — a mixed metres+radians L2 grows
    # with sqrt(DoF) and is dominated by passive wheels, mislabelling a wheeled/high-
    # DoF gait. Exclude x,y ground advance and passive joints.
    j = np.asarray(traj["joints"], float)
    passive = set(getattr(spec, "passive_joint_indices", []) or [])
    act = [k for k in range(j.shape[1]) if k not in passive]
    base_ret = float(abs(root[-1, 2] - root[0, 2]))
    joint_ret = float(np.max(np.abs(j[-1, act] - j[0, act]))) if act else 0.0
    cyclic = base_ret < 0.20 and joint_ret < 0.20
    lines.append(f"  start->end seam: base Δz {base_ret*100:.0f} cm, worst joint "
                 f"{joint_ret:.2f} rad ({'returns to start (cyclic)' if cyclic else 'distinct end (one-shot)'})")

    # ---------------- END-EFFECTORS ----------------
    ee = _ee_pos(traj)                                    # (T,N,3)
    cm = traj.get("contact_mask")
    cl = traj.get("surface_clearance")
    lines.append("")
    lines.append("END-EFFECTORS (contact timeline over one cycle; C=planted):")
    onsets = {}
    if cm is not None:
        cm = np.asarray(cm, dtype=bool)
        cl = np.asarray(cl, dtype=float) if cl is not None else None
        for i, name in enumerate(ee_names):
            m = cm[:, i]
            frac = float(m.mean())
            tl = _timeline(m)
            # swing lift: peak clearance while NOT in contact
            lift = float(cl[~m, i].max()) if (cl is not None and (~m).any()) else 0.0
            # in-contact drift: worst net horizontal move over a firm contact run
            drift = 0.0
            for a, b in _runs(m):
                if b - a < 2:
                    continue
                drift = max(drift, float(np.linalg.norm(ee[b - 1, i, :2] - ee[a, i, :2])))
            # contact-onset phase (first rising edge) for gait coordination
            edges = np.where((~m[:-1]) & (m[1:]))[0]
            if edges.size:
                onsets[name] = float((edges[0] + 1) / (T - 1))
            lines.append(f"  {name:12s} [{tl}] contact {frac*100:3.0f}%  "
                         f"swing-lift {lift*100:4.1f}cm  in-contact drift {drift*100:4.1f}cm")
    else:
        lines.append("  (no contact schedule recorded)")

    # pairwise contact-onset phase offsets (gait coordination signature)
    if len(onsets) >= 2:
        names = list(onsets)
        pairs = []
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                d = abs(onsets[names[a]] - onsets[names[b]])
                d = min(d, 1 - d)                          # phase is circular
                pairs.append((d, names[a], names[b]))
        pairs.sort()
        lines.append("  contact-onset phase offsets (0=together, 0.5=antiphase):")
        for d, a, b in pairs[:6]:
            lines.append(f"    {a} vs {b}: {d:.2f}")

    # ---------------- REACHABILITY (IK target vs achieved) ----------------
    # A gap between the COMMANDED IK target and the ACHIEVED position means the limb
    # could not reach its target. The correct FIX depends on WHAT the target is —
    # and prescribing the wrong one wastes the repair loop:
    #   - a DECLARED CONTACT (a foot on a tread) is the DATUM: the world/surface
    #     fixes where it lands, so the miss means the BASE is wrong. Move the BASE
    #     toward it (base height RELATIVE to the surface, or advance the base xy to
    #     the foothold) — never move the contact. (The pre-sim reach-lint prints the
    #     exact base interval; this is the post-sim confirmation, attributed right.)
    #   - a FREE LIMB reaching a world point (a gesture) is the GUESS: clamp it into
    #     reach with reach_limited_target.
    com = traj.get("ee_positions")
    if com is not None:
        com = np.asarray(com, dtype=float)
        if com.shape == ee.shape:
            resid = np.linalg.norm(com - ee, axis=2)                 # (T, N)
            # SUSTAINED miss, not the single worst frame: a lone one-frame spike at a
            # swing->plant seam is a CONTINUITY artifact (caught below), not a reach
            # failure — flagging it as an "80 cm float" misdirects the repair to the
            # base when the base is fine. Score each EE by the p90 of its residual
            # over the frames that MATTER (while it is a declared contact, since a
            # contact that floats is the real fault; else all frames for a free limb).
            big = []
            for i in range(len(ee_names)):
                ci = (cm[:, i] if (cm is not None and i < cm.shape[1]) else None)
                if ci is not None and ci.any():
                    rr = resid[ci, i]; frames = np.where(ci)[0]
                else:
                    rr = resid[:, i]; frames = np.arange(T)
                sustained = float(np.percentile(rr, 90))
                if sustained > 0.06:
                    pk = int(frames[int(np.argmax(rr))])             # worst relevant frame
                    big.append((ee_names[i], sustained, i, pk))
            if big:
                lines.append("")
                lines.append("REACHABILITY (limb SUSTAINS a miss of its IK target — a DECLARED "
                             "CONTACT that misses FLOATS; fix the BASE, not the contact):")
                for n, w, i, pk in sorted(big, key=lambda x: -x[1])[:4]:
                    miss = com[pk, i] - ee[pk, i]                    # target - achieved
                    vert, horiz = abs(float(miss[2])), float(np.linalg.norm(miss[:2]))
                    is_contact = (bool(cm[pk, i])
                                  if (cm is not None and i < cm.shape[1]) else None)
                    where = f"at phase {pk/(T-1):.2f}"
                    if is_contact:
                        fix = (("its base is too HIGH/LOW for this surface — set the base "
                                "height RELATIVE to the contact surface "
                                "(base_z = surface_z + a fixed crouch), don't hand-pick it")
                               if vert >= horiz else
                               ("the base has not advanced to this foothold — move the base "
                                "xy so it stays within a leg-length of the contact"))
                        lines.append(f"  {n}: DECLARED CONTACT missed by {w*100:.0f} cm "
                                     f"{where} (it FLOATS) — {fix}.")
                    elif is_contact is False:
                        lines.append(f"  {n}: free-limb target missed by {w*100:.0f} cm "
                                     f"{where} — clamp the target into reach with "
                                     f"reach_limited_target.")
                    else:
                        lines.append(f"  {n}: target missed by {w*100:.0f} cm {where} — if "
                                     f"it is a declared contact, fix the BASE; if a free "
                                     f"limb, clamp the target with reach_limited_target.")

    # ---------------- CONTINUITY (channel switches + joint teleports) -----
    # A control-channel switch (a limb going joint-override <-> IK) or an unbridged
    # target jump snaps the joints in ONE frame. The framework now BRIDGES releases
    # and free-limb switches, so a residual isolated pop almost always sits on a
    # limb LANDING into a contact (a swing that ends far from its foothold) — an
    # authoring fix, not a retime. Surfacing WHERE the pop is + which limb switched
    # channel there lets the critic target the exact seam.
    disc = []
    act = traj.get("ee_active")
    if act is not None:
        act = np.asarray(act, dtype=bool)
        for i, name in enumerate(ee_names):
            if i >= act.shape[1]:
                break
            sw = np.where(act[:-1, i] != act[1:, i])[0]
            if sw.size:
                ph = ", ".join(f"{(s+1)/(T-1):.2f}{'->IK' if act[s+1, i] else '->joint/free'}"
                               for s in sw[:6])
                disc.append(f"  {name}: control-channel switch at phase {ph}")
    q = np.asarray(traj["joints"], dtype=float)
    if q.shape[0] > 2:
        dq = np.abs(np.diff(q, axis=0))
        worst = float(dq.max())
        if worst > 0.30:
            fi = int(dq.max(axis=1).argmax())
            disc.append(f"  worst single-frame joint step {worst:.2f} rad at phase "
                        f"{(fi+1)/(T-1):.2f} — if isolated, a channel-handoff snap; make the "
                        f"switch happen over a phase window (ease the swing INTO its foothold).")
    if disc:
        lines.append("")
        lines.append("CONTINUITY (channel switches often align with pose snaps):")
        lines.extend(disc)

    # ---------------- FREE-LIMB IK RESIDUAL (commanded vs achieved) ----------------
    # For each IK-active, NON-contact ("free") limb, the gap between the COMMANDED
    # world target and where the limb ACTUALLY ended up. A large residual means the
    # target is unreachable (the IK under-reaches) — the direct cause of a free-hand
    # flicker BEFORE the reach projection clamps it, and it stays visible even when a
    # body-frame target is the culprit (which the joint no_teleport can't attribute).
    com = traj.get("ee_positions")
    if com is not None and act is not None:
        com = np.asarray(com, dtype=float)
        res = []
        free = act.copy()
        if cm is not None:
            cmask = np.asarray(cm, dtype=bool)
            k = min(free.shape[1], cmask.shape[1])
            free[:, :k] &= ~cmask[:, :k]                  # IK-active AND not a contact
        for i, name in enumerate(ee_names):
            if i >= free.shape[1] or i >= com.shape[1] or i >= ee.shape[1]:
                break
            fr = free[:, i]
            if fr.sum() < 3:
                continue
            gap = np.linalg.norm(com[fr, i] - ee[fr, i], axis=1)
            if float(gap.mean()) > 0.05:
                res.append(f"  {name}: free-limb IK under-reaches by {gap.mean()*100:.0f} cm mean "
                           f"({gap.max()*100:.0f} cm worst) — its commanded target is beyond reach.")
        if res:
            lines.append("")
            lines.append("FREE-LIMB IK RESIDUAL (commanded target vs achieved — a big gap = unreachable):")
            lines.extend(res)

    # ---------------- COLLISIONS / PENETRATION (localized) ----------------
    sp = traj.get("surface_penetration")
    sc = traj.get("self_collision")
    cev = traj.get("clamp_events")
    pen_lines = []
    if sp is not None:
        sp = np.asarray(sp, dtype=float)
        bad = np.where(sp > 0.02)[0]
        if bad.size:
            pk = int(bad[np.argmax(sp[bad])])
            # LOCALIZED: the recorder names the exact BODY + SURFACE of the deepest
            # geom (penetration_body / penetration_surface) — this catches a link
            # that ISN'T an end-effector (the CLIMB shin clipping the slab), which
            # the old lowest-achieved-EE-z guess could never name. Fall back to the
            # EE guess only if the localized fields are absent (older pkls).
            pb = traj.get("penetration_body"); ps = traj.get("penetration_surface")
            if pb is not None and pk < len(pb) and pb[pk]:
                where = f"body '{pb[pk]}'"
                surf = (ps[pk] if ps is not None and pk < len(ps) and ps[pk] else None)
                if surf:
                    where += f" clipping the {surf}"
            else:
                where = f"limb {ee_names[int(np.argmin(ee[pk, :, 2]))]}"
            pen_lines.append(
                f"  SURFACE penetration: peaks {sp.max()*100:.0f} cm at phase "
                f"{pk/(T-1):.2f} ({bad.size} frames > 2 cm); the offender is "
                f"{where}. Route it ABOVE/AROUND the surface (raise the base or the "
                f"joint angles so THAT link clears — a foot-only fix won't help a shin/torso clip).")
    if sc is not None:
        sc = np.asarray(sc, dtype=float)
        if sc.max() > 0.04:
            pk = int(np.argmax(sc))
            pen_lines.append(f"  SELF-collision: peaks {sc.max()*100:.0f} cm at phase {pk/(T-1):.2f}.")
    if cev is not None:
        cev = np.asarray(cev, dtype=float)
        n = int((cev > 0).sum())
        if n:
            pen_lines.append(f"  the framework had to CLAMP a limb target out of a surface on "
                             f"{n} frames — your authored path drove a limb into an obstacle there.")
    if pen_lines:
        lines.append("")
        lines.append("PENETRATION / CLAMPS:")
        lines.extend(pen_lines)

    # ---------------- SILENT FRAMEWORK INTERVENTIONS ----------------
    # The framework quietly corrects an infeasible authored motion — it lifts the
    # base off a too-deep contact (floor-assist), seats it so a floating contact
    # stays reachable (reach-seat), or clamps a target out of a surface. If the
    # critic can't SEE these, it chases a phantom (re-tuning a crouch height that a
    # floor-assist lift keeps silently cancelling — the CARRYBOX 6-version plateau).
    # Attribute the authored-vs-achieved gap to the exact intervention that caused it.
    rs = traj.get("reach_seat_delta"); fa = traj.get("floor_assist_lift")
    cmag = traj.get("clamp_mag")
    interv = []
    if fa is not None:
        fa = np.asarray(fa, dtype=float)
        if fa.size and fa.max() > 0.005:
            n = int((fa > 0.005).sum())
            interv.append(
                f"  FLOOR-ASSIST lifted the base by up to {fa.max()*100:.0f} cm on {n} frames: "
                f"your declared contact(s) were authored BELOW their surface, so the framework "
                f"raised the WHOLE body to seat them — this is why an authored crouch height reads "
                f"~{fa.max()*100:.0f} cm HIGHER than you set it. Author each contact z ON its surface "
                f"(z = surface + ground_offset). (Lift is capped at 5 cm; a deeper gap is left to "
                f"FAIL surface_penetration, so a big cap-hit means the base and legs disagree grossly.)")
    if rs is not None:
        rs = np.asarray(rs, dtype=float)
        if rs.size and np.abs(rs).max() > 0.01:
            k = int(np.argmax(np.abs(rs)))
            capped = np.abs(rs).max() >= 0.149   # hit the REACH_SEAT_CAP (0.15 m)
            interv.append(
                f"  REACH-SEAT shifted the base z by up to {rs[k]*100:+.0f} cm (at phase {k/(T-1):.2f}) "
                f"so a declared contact stayed within limb reach — your authored base left a contact "
                f"beyond reach at that phase. TWO valid repairs, pick one: (A) set the base height "
                f"RELATIVE to the contact surface so the limb root sits within arm's length of the "
                f"contact; OR (B) END that contact's declared window EARLIER — a planted contact BOUNDS "
                f"the base to its reach, so release it BEFORE the base travels past the reach ball "
                f"(never hold a contact 'for stability' once the base has moved beyond it)."
                + (f" ⚠️ The correction HIT THE {REACH_SEAT_CAP_CM:.0f} cm CAP — the authored base is "
                   f"grossly wrong; the framework stopped correcting and the contact is now FLOATING "
                   f"(it will FAIL contact_grounded/support). You MUST apply (A) or (B)." if capped else ""))
    if cmag is not None:
        cmag = np.asarray(cmag, dtype=float)
        if cmag.size and cmag.max() > 0.01:
            frac = float((cmag > 0.005).mean())   # fraction of frames a target was clamped
            systemic = ("  ⚠️ THIS IS SYSTEMATIC — the clamp fired on "
                        f"{frac*100:.0f}% of frames, so you are authoring EVERY contact z wrong "
                        "(almost certainly at the bare surface height / z=0 instead of "
                        "surface + ground_offset). Fix the z ARITHMETIC for all contacts, not one."
                        if frac > 0.5 else "")
            interv.append(
                f"  TARGET-CLAMP raised a limb target by up to {cmag.max()*100:.0f} cm out of a surface: "
                f"the authored target was INSIDE the floor/an object. Author contact targets ON the "
                f"surface (contact z = surface + ground_offset), not at z=0 (which sinks the sole)."
                + systemic)
    rp = traj.get("reach_proj")
    if rp is not None:
        rp = np.asarray(rp, dtype=float)
        if rp.size and rp.max() > 0.02:
            n = int((rp > 0.02).sum())
            interv.append(
                f"  FREE-LIMB REACH: a free (non-contact) limb's WORLD target was beyond arm's reach on "
                f"{n} frames — the framework pulled it up to {rp.max()*100:.0f} cm back onto the reachable "
                f"ball (else the arm would flicker as IK flips configurations). Your authored hand/foot "
                f"target is too far from the torso; author free-limb targets WITHIN reach of the base, or "
                f"move the base closer, or drive that limb by JOINT angles (compute_joint_overrides) if its "
                f"exact world position doesn't matter (a gesture).")
    if interv:
        lines.append("")
        lines.append("SILENT FRAMEWORK INTERVENTIONS (the framework CORRECTED your authored motion "
                     "here — fix the SOURCE, do not fight the symptom):")
        lines.extend(interv)

    # ---------------- OBJECTS ----------------
    objs = traj.get("object_pos") or {}
    if objs:
        lines.append("")
        lines.append("OBJECTS:")
        for name, p in objs.items():
            p = np.asarray(p, dtype=float)
            lift = float(p[:, 2].max() - p[:, 2].min())
            oplen = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
            # closest any EE ever gets to the object (grip proxy)
            dmin = float(np.linalg.norm(ee - p[:, None, :], axis=2).min())
            lines.append(f"  {name}: start z {p[0,2]:.2f} -> max z {p[:,2].max():.2f} "
                         f"(lifted {lift*100:.0f}cm); path {oplen*100:.0f}cm; "
                         f"nearest EE approach {dmin*100:.0f}cm")

    return "\n".join(lines)
