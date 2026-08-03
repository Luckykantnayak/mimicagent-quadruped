"""
simulator/tools/reach_lint.py

PRE-SIM reachability lint — the cheap geometry oracle the agent calls BEFORE
paying for a full IK sim rollout. It builds the same controller the sim uses
(so the limb reach + the skill's authored base/contacts are identical), then
replays the SKILL's own step() across one cycle WITHOUT IK or rendering and asks,
at every phase, whether each declared world contact is within its limb's reach
from the authored base. A contact beyond reach would FLOAT once the sim runs.

It reports violations in BASE COORDINATES — "at phase 0.60 contact right_foot
needs base z in [0.72, 0.90]; you authored 1.03" — the exact, actionable fix a
hand-author derives by watching and re-tuning, handed to the coder before it
spends minutes on a doomed rollout. Only meaningful under grounding "off" (the
skill owns the absolute base); under "contact" the base is re-seated from the
contacts, so the lint is a no-op.

Prints a human report to stderr and one machine line to stdout:
    LINT_REPORT_JSON: {...}
Exit code 2 => hard-fail (a violation the base-reach seat cannot cleanly absorb).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config
from core.simulator import MujocoSimulator
from core.controller import MotionControllerRunner, REACH_GATE, REACH_TARGET
from utils.robot_config import load_robot_spec
from utils.reach import contact_reach
from utils.plan_io import load_plan_dict, extract_invariants

N_STEPS_CAP = 4000          # safety cap on the replay length
FREE_SEVERE = 1.25          # a FREE limb's target beyond this multiple of its reach is
                            # not a stretch, it is unreachable: the runtime projection
                            # truncates the intent by >20% of limb length.
FREE_BAD_FRAC = 0.10        # ...and if that happens on more than this fraction of the
                            # clip it is the skill's normal state, not a transient, so
                            # hard-fail pre-sim instead of spending a rollout to find out.
NO_DRAG_ON = os.environ.get("NO_DRAG", "0") == "1"   # opt-in: see note above
SLIDE_GROSS_M = 0.25        # an AUTHORED contact target that TRANSLATES further than this
                            # within one continuous declared-contact run is a contact the
                            # skill never released — it is being dragged, not planted. Set
                            # at 2.5x the no-slide law's own 0.10 m budget so only the
                            # unambiguous cases hard-fail and a marginal drift still falls
                            # through to the sim (the standing lint philosophy: short-circuit
                            # only what the runtime cannot absorb; under grounding "off" the
                            # pin layer is disabled entirely, so nothing absorbs it there).
NO_SURFACE_M = 0.25         # a DECLARED CONTACT hovering more than this above the nearest
                            # surface beneath it has NOTHING to rest on. Well clear of the
                            # framework's own 5 cm floor-assist slack, so only gross cases
                            # (standing at ledge height before the ledge begins) fire.
ARRIVAL_JUMP_M = 0.25       # a DECLARED CONTACT whose authored world target STEPS more
                            # than this in ONE recorded frame is a swing that never
                            # arrived — it teleports INTO contact (a ~1 m jump from a
                            # dead/unreachable swing branch). The seam bridge deliberately
                            # will NOT hide a landing, so this fails no_teleport
                            # unrecoverably; catch it pre-sim. Applies to ALL grounding
                            # modes (an into-contact teleport is not a reach question).


def _resolve_scene_grounding(spec, args):
    """Mirror run.py: explicit --scene/--grounding win, else the exemplar's env."""
    scene = getattr(args, "scene", None)
    grounding = getattr(args, "grounding", "off")
    ex = spec.get_exemplar(args.mode)
    if not scene and ex:
        env = spec.get_environment(ex.get("environment"))
        scene = env.get("scene")
        g = env.get("grounding", "contact")
        grounding = ("off" if g is False else "contact") if isinstance(g, bool) else g
    if isinstance(grounding, bool):
        grounding = "off" if grounding is False else "contact"
    return (scene or spec.scene_xml), grounding


def lint(args):
    spec = load_robot_spec(args.robot, robots_dir=getattr(args, "robots_dir", "robots"))
    scene_xml, grounding = _resolve_scene_grounding(spec, args)

    sim = MujocoSimulator(scene_xml, init_position=args.init_position,
                          sim_freq=int(args.sim_freq), base_body=spec.base_body,
                          camera=spec.camera, headless=True, robot_mjcf=spec.mjcf)
    runner = MotionControllerRunner(sim, args)
    mg = runner.motion_gen
    geom = runner._limb_geom
    valid = set(runner._contact_names)
    dt = runner.dt
    # REACHABILITY runs in BOTH grounding modes: registration shifts the base AND the
    # contacts together, so the base-to-contact distance (hence over-reach) is
    # grounding-INVARIANT — a contact authored beyond reach floats the same under
    # "contact" as under "off" (a handstand's hands below an over-high pelvis). The
    # ARRIVAL check also runs in both modes. (A purely VERTICAL over-reach still falls
    # through to the sim, where the now-ungated reach-seat fixes it; only horizontal /
    # jointly-incompatible over-reach hard-fails, in either mode.)
    do_reach = True

    freq = getattr(mg, "freq", None) or args.gait_freq or 1.0
    n_steps = min(N_STEPS_CAP, max(2, int(round(1.0 / max(freq, 1e-3) / dt))))

    # Replay the skill (no IK): collect per-declared-contact reachability + arrival.
    mg.t = 0.0
    acc = {}         # contact -> worst-violation record (reachability)
    free = {}        # FREE (non-contact) EE -> worst over-reach of its authored target
    incompat = None  # worst frame where the declared contacts want INCOMPATIBLE base heights
    prev_world = {}  # ee -> authored world target last frame (arrival continuity)
    arrivals = {}    # ee -> worst into-contact target JUMP record
    notarget = {}    # ee -> # frames declared a contact but with a None world target
    nosurface = {}   # ee -> worst record of a declared contact with NOTHING under it
    dragging = {}    # ee -> worst record of a declared contact that TRANSLATES
    stance = {}      # ee -> (first authored xy of the current declared-contact run, phase)
    ee_only = set(spec.ee_names)   # contact BODIES (knees) are legit FK-only; exempt them

    # EXEMPTIONS from the no-drag law, read from the SAME sources the scored
    # `contact_no_slide` law uses so the pre-sim oracle and the post-sim test can
    # never disagree about what is legal:
    #   * spec `wheel: true` EEs        — a wheel ROLLS while planted, by design;
    #   * plan `sliding_contacts`       — the planner may exempt a contact outright
    #                                     (a drag, a skate, a slide is the skill);
    #   * plan contacts with mode pivot — a pirouette spins on the spot.
    # Without these the law would forbid legal motions, i.e. make the FRAMEWORK the
    # ceiling rather than the physics.
    _skill_dir = os.path.join("skill_output", spec.name, args.mode)
    _plan = load_plan_dict(_skill_dir, args.mode)
    _inv, _sliding, _pivots = extract_invariants(_plan)
    no_drag_exempt = set(_sliding) | set(_pivots) | {
        spec.ee_names[i] for i in spec.wheel_ee_indices}
    # Per-contact ground_offset (body origin -> sole/palm), so "is there a surface
    # under this contact" is asked about the CONTACT POINT, not the body centre.
    _off = {n: float(o) for n, o in zip(runner._contact_names, runner._contact_offsets)}
    for _ in range(n_steps):
        st = mg.step(dt)
        phase = float(st["phase"]); root = st["root_pos"]; quat = st["root_quat"]
        eew = st["ee_positions_world"] or {}
        contacts = st["contacts"]
        active = contacts if contacts is not None else (mg.support_ees or [])

        # --- ARRIVAL (all grounding modes): a DECLARED CONTACT whose authored world
        #     target jumped this frame is a swing that teleported into contact. Uses
        #     ONLY the authored targets (no IK) — the ~1 m within-world-channel jump
        #     of a dead/unreachable swing branch is visible directly. A None target
        #     last frame (joint-driven) breaks the chain (can't assess pre-IK).
        for n in active:
            cur = eew.get(n)
            if cur is None:
                if n in ee_only:                 # an EE contact with NO IK target -> floats
                    notarget[n] = notarget.get(n, 0) + 1
                continue
            pv = prev_world.get(n)
            if pv is not None:
                step = float(np.linalg.norm(np.asarray(cur, float) - pv))
                if step > ARRIVAL_JUMP_M and (n not in arrivals or step > arrivals[n]["jump"]):
                    arrivals[n] = {"jump": round(step, 3), "phase": round(phase, 3)}
        # --- SURFACE EXISTENCE (all grounding modes): a DECLARED CONTACT must have a
        #     real surface beneath it. Reach and arrival both assume the contact POINT
        #     is legitimate and only ask whether the limb can get there — so a skill
        #     that stands at ledge HEIGHT before the ledge BEGINS passes every existing
        #     check and walks on thin air. Measured on the ledge drop: the robot stood
        #     and walked 0.55 m with both feet declared as contacts at z=0.80 while the
        #     slab only starts at x=1.05, i.e. 0.81 m of nothing underneath, then fell
        #     at -74 m/s^2. `contact_grounded` catches it only AFTER a full rollout, and
        #     the loop then spent 10 iterations failing to fix it. One ray-cast per
        #     declared contact catches it before any simulation.
        for n in active:
            c = eew.get(n)
            if c is None:
                continue
            cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
            sole = cz - _off.get(n, 0.0)
            surf = runner.surfaces.support_z(cx, cy, sole + 0.05, default=None)
            drop = (sole - float(surf)) if surf is not None else float("inf")
            # NOT ONLY DOWNWARD. A contact is a load-bearing touch, and a limb can bear
            # load against a surface ABOVE it just as legitimately as one below — a bar
            # hang, a pull-up, brachiation, a brace against a ceiling. Casting only the
            # down-ray would hard-fail every one of those pre-sim, making the FRAMEWORK
            # the limit on what a user may ask for. Only flag a contact with nothing
            # near it in EITHER direction; that still catches standing at a ledge's
            # height beyond its edge (empty above and below).
            above = runner.surfaces.surface_above(cx, cy, sole)
            if above is not None and (float(above) - sole) <= NO_SURFACE_M:
                continue
            # Keep `drop` NUMERIC always. Storing None for "nothing at all below" made
            # the next frame's `drop > nosurface[n]["drop"]` raise TypeError, and the
            # lint fails OPEN on any exception — so this check silently disabled ITSELF
            # and passed a skill standing 0.75 m up on empty air straight through to the
            # sim. `void` carries the "no surface anywhere below" case for the report.
            if drop > NO_SURFACE_M and (n not in nosurface or drop > nosurface[n]["drop"]):
                nosurface[n] = {"drop": (round(drop, 3) if np.isfinite(drop) else 1e9),
                                "void": not np.isfinite(drop),
                                "phase": round(phase, 3),
                                "at": [round(cx, 2), round(cy, 2), round(cz, 2)],
                                "surface_z": (round(float(surf), 3) if surf is not None else None)}

        # --- NO-DRAG (all grounding modes): a DECLARED CONTACT must not TRANSLATE.
        #     A contact is a limb held still against the world; one whose AUTHORED
        #     target slides across the ground was never released, so the skill is
        #     gliding rather than stepping. Purely an AUTHORING law — it reads the
        #     authored targets only, so a tracking error the solver introduces is not
        #     blamed here. Measured on a carry-box: BOTH feet declared contacts on
        #     2000/2000 frames while the base advanced 1.05 m, i.e. no gait at all;
        #     `contact_no_slide` caught it only after a full rollout, and the loop then
        #     spent its whole budget on the resulting floor failure instead of on the
        #     grip the skill was actually about.
        for n in active:
            c = eew.get(n)
            if c is None or n in no_drag_exempt:
                stance.pop(n, None)
                continue
            xy = np.array([float(c[0]), float(c[1])])
            seed = stance.get(n)
            if seed is None:
                stance[n] = (xy, phase)
                continue
            d = float(np.linalg.norm(xy - seed[0]))
            if d > SLIDE_GROSS_M and (n not in dragging or d > dragging[n]["drift"]):
                dragging[n] = {"drift": round(d, 3), "from_phase": round(seed[1], 3),
                               "to_phase": round(phase, 3)}
        for n in list(stance):                    # a released contact ends its run
            if n not in active or eew.get(n) is None:
                stance.pop(n, None)

        for n in list(prev_world):                # refresh / break continuity chains
            if eew.get(n) is None:
                prev_world.pop(n, None)
        for n, w in eew.items():
            if w is not None:
                prev_world[n] = np.asarray(w, float)

        if not do_reach:
            continue
        bz = float(root[2])
        frame_iv = []                             # (ee, zlo, zhi) for z-constrained contacts
        for n in active:
            if n not in geom or n not in valid:
                continue                          # FK-only contact (e.g. a knee): no reach target
            c = eew.get(n)
            if c is None:
                continue
            over, zlo, zhi = contact_reach(root, quat, c, geom[n], REACH_TARGET)
            # Incompatibility detection uses TRUE reach (margin 1.0): the 0.98 target
            # margin shrinks each interval by ~(1-0.98)*reach, which would inflate the
            # measured gap past the 0.02 m dead-band on a long limb and FALSE hard-fail
            # a straddle that is jointly reachable near full extension. The reported
            # fix bands still use the 0.98 zlo/zhi (a slight bend the IK can hit).
            _, zlo1, zhi1 = contact_reach(root, quat, c, geom[n], 1.0)
            if not np.isnan(zlo1):
                frame_iv.append((n, float(zlo1), float(zhi1)))
            if over <= REACH_GATE:
                continue                          # within reach (a standing pose sits <=1.0)
            horiz = bool(np.isnan(zlo))
            shift = 0.0 if horiz else (zlo - bz if bz < zlo else (bz - zhi if bz > zhi else 0.0))
            shift = abs(float(shift))
            rec = acc.get(n)
            # keep the worst frame for this contact: horizontal beats vertical,
            # then larger over-reach.
            worse = (rec is None or (horiz and not rec["horizontal"])
                     or (horiz == rec["horizontal"] and over > rec["worst_over"]))
            if worse:
                acc[n] = {"worst_over": round(over, 3), "worst_phase": round(phase, 3),
                          "authored_base_z": round(bz, 3), "horizontal": horiz,
                          "need_lo": None if horiz else round(float(zlo), 3),
                          "need_hi": None if horiz else round(float(zhi), 3),
                          "max_shift": round(shift, 3)}
            elif not horiz and rec is not None and shift > rec["max_shift"]:
                rec["max_shift"] = round(shift, 3)
        # JOINTLY-INFEASIBLE contacts: two contacts declared at once whose base-z
        # intervals are DISJOINT — one needs the base high, the other low, and NO
        # single base reaches both (e.g. one foot still on the floor while the other
        # is on a ledge). The seat can only compromise to a midpoint and BOTH float,
        # so this is a real fault the seat cannot fix — hard-fail it (like horizontal).
        if len(frame_iv) >= 2:
            need_hi = max(frame_iv, key=lambda t: t[1])   # highest lower-bound (base >= zlo)
            need_lo = min(frame_iv, key=lambda t: t[2])   # lowest upper-bound  (base <= zhi)
            gap = need_hi[1] - need_lo[2]                 # (max zlo) - (min zhi); >0 => disjoint
            if gap > 0.02 and (incompat is None or gap > incompat["gap"]):
                incompat = {"phase": round(phase, 3), "gap": round(float(gap), 3),
                            "need_high": {"ee": need_hi[0], "min_base_z": round(need_hi[1], 3)},
                            "need_low":  {"ee": need_lo[0], "max_base_z": round(need_lo[2], 3)}}

        # FREE-TARGET reach (SOFT): a FREE limb (has a world target this phase but is NOT
        # a declared contact) whose target is beyond reach makes IK flip configs → the
        # hand flicker. The runtime reach-projection now pulls it into reach, so this is
        # a WARNING not a hard-fail — the arm won't reach the authored point (the intent
        # is silently truncated). Flag it so the coder authors a reachable target or
        # switches the limb to joint-driven if its exact world position doesn't matter.
        for n, c in eew.items():
            if c is None or n in active or n not in geom or n not in valid:
                continue
            over_f, _, _ = contact_reach(root, quat, np.asarray(c, float), geom[n], 1.0)
            if over_f > REACH_GATE:
                rec = free.get(n)
                if rec is None:
                    rec = free[n] = {"worst_over": 0.0, "worst_phase": 0.0,
                                     "frames": 0, "severe_frames": 0}
                rec["frames"] += 1
                if over_f > FREE_SEVERE:
                    rec["severe_frames"] += 1
                if over_f > rec["worst_over"]:
                    rec["worst_over"] = round(float(over_f), 3)
                    rec["worst_phase"] = round(phase, 3)

    # Hard-fail a HORIZONTAL over-reach (base x/y wrong, no base height fixes it) OR
    # INCOMPATIBLE contacts (no base height satisfies all of them at once) — both are
    # faults the runtime base-reach seat CANNOT fix, so a sim rollout is wasted and the
    # coder must re-author the base/contact schedule. A purely VERTICAL over-reach that
    # a single base shift WOULD fix is left to fall through to the sim, where the seat
    # handles it — keeping the pre-sim lint (#2) and the runtime seat (#1) complementary.
    # An EE declared a contact with NO world target for a SUSTAINED span floats (a
    # contact body / knee is legit FK-only and was exempted above).
    notarget_bad = {n: c for n, c in notarget.items() if c > max(3, int(0.05 * n_steps))}

    # A CHRONICALLY unreachable FREE limb is a hard fault too. A brief over-reach is
    # fine — the runtime projection absorbs it and the limb only misses its authored
    # point for an instant. But a limb held far out of reach for a large FRACTION of
    # the clip means the skill's intent is silently truncated for most of the motion:
    # a handstand authored its hands 0.54 m past the arms' reach on 17% of frames, and
    # because this was warn-only it cost a full rollout every iteration to discover.
    # Gate on BOTH severity and persistence so a momentary stretch still passes.
    for n, v in free.items():
        v["severe_frac"] = round(v["severe_frames"] / max(1, n_steps), 3)
    free_bad = {n: v for n, v in free.items() if v["severe_frac"] > FREE_BAD_FRAC}

    # Hard-fail a HORIZONTAL over-reach, INCOMPATIBLE contacts, an into-contact ARRIVAL
    # teleport, a target-less declared EE contact, or a chronically unreachable free
    # limb — all faults the runtime cannot fix.
    hard = (any(v["horizontal"] for v in acc.values()) or (incompat is not None)
            or bool(arrivals) or bool(notarget_bad) or bool(free_bad) or bool(nosurface)
            or (bool(dragging) and NO_DRAG_ON))
    report = _format(acc, grounding, incompat, arrivals, notarget_bad, free, nosurface,
                     dragging)
    return {"grounding": grounding, "hard_fail": bool(hard),
            "contacts": acc, "incompatible": incompat, "arrivals": arrivals,
            "notarget": notarget_bad, "free_reach": free, "nosurface": nosurface,
            "dragging": dragging, "report": report}


def _format(acc, grounding, incompat=None, arrivals=None, notarget=None, free=None,
            nosurface=None, dragging=None):
    arrivals = arrivals or {}
    notarget = notarget or {}
    free = free or {}
    nosurface = nosurface or {}
    dragging = dragging or {}
    if dragging and NO_DRAG_ON and not nosurface:
        out = ["PRE-SIM CONTACT-DRAG FAILURE — no simulation was run. A DECLARED CONTACT "
               "TRANSLATES across the ground while still declared. A contact is a limb held "
               "STILL against the world; if it slides, it was never released and the robot "
               "is gliding rather than stepping."]
        for n, v in sorted(dragging.items(), key=lambda kv: -kv[1]["drift"]):
            out.append(f"  - {n}: authored target moves {v['drift']:.2f} m between phase "
                       f"~{v['from_phase']} and ~{v['to_phase']} while declared a contact.")
        out.append("  FIX: RELEASE the contact before the limb moves. For a gait, contacts(phase) "
                   "must ALTERNATE — a limb is a contact only while its foot is planted, and is "
                   "dropped from the list for the whole swing; a support set that never changes "
                   "is not a walk. If the sliding IS the intent (a skate/drag/pivot), declare it: "
                   "`sliding_contacts` exempts a contact entirely and a contact with mode 'pivot' "
                   "may spin in place.")
        return "\n".join(out)
    if nosurface:
        # Reported FIRST and alone: every other finding assumes the contact POINT is
        # legitimate and only asks whether the limb can reach it. If there is nothing
        # under the contact at all, those answers are meaningless — fix this first.
        out = ["PRE-SIM SURFACE-EXISTENCE FAILURE — no simulation was run. A DECLARED "
               "CONTACT has NO SURFACE BENEATH IT: the skill is standing on thin air. A "
               "contact is only real if something is actually there to rest on."]
        for n, v in sorted(nosurface.items(), key=lambda kv: -kv[1]["drop"]):
            d = ("nothing at all below it" if v.get("void")
                 else f"{v['drop']:.2f} m of empty space below it")
            where = (f"nearest surface under that (x,y) is z={v['surface_z']}"
                     if v["surface_z"] is not None else "no surface under that (x,y)")
            out.append(f"  - {n} at phase ~{v['phase']}: declared a contact at "
                       f"{v['at']} with {d} ({where}).")
        out.append("  FIX: put the contact ON a real surface. Either move the BASE (and the "
                   "contact with it) in x/y to where that surface actually IS — check the scene "
                   "geometry for the object's FOOTPRINT, not just its height — or lower the "
                   "contact onto the surface that is genuinely beneath it. Authoring a contact "
                   "at a raised surface's HEIGHT while standing beyond its EDGE is the common "
                   "form of this bug.")
        return "\n".join(out)
    if not acc and not incompat and not arrivals and not notarget and not free:
        return ("PRE-SIM LINT (no IK): PASS — every declared contact is within its "
                "limb's reach from the authored base, and every swing ARRIVES at its "
                "foothold before it is declared a contact (no into-contact teleport).")
    lines = []
    if free and not acc and not incompat and not arrivals and not notarget:
        # A CHRONIC over-reach is a hard fault; a transient one is a soft warning the
        # runtime projection absorbs. Say which, so the coder knows whether it must act.
        chronic = {n: v for n, v in free.items() if v.get("severe_frac", 0) > FREE_BAD_FRAC}
        if chronic:
            lines.append("PRE-SIM FREE-LIMB REACH FAILURE — no simulation was run. A FREE "
                         "(non-contact) limb's authored WORLD target is beyond its reach for a "
                         "LARGE FRACTION of the clip, so this is the skill's normal state, not a "
                         "momentary stretch. The runtime would project the target onto the "
                         "reachable ball, silently truncating your intent for most of the motion:")
            for n, v in sorted(chronic.items(), key=lambda kv: -kv[1]["worst_over"]):
                lines.append(f"  - {n}: up to {v['worst_over']:.2f}× reach, beyond "
                             f"{FREE_SEVERE:.2f}× on {100*v['severe_frac']:.0f}% of frames "
                             f"(worst at phase ~{v['worst_phase']}). Bring the target within "
                             f"reach, move the BASE nearer so the limb can get there, or drive "
                             f"the limb by JOINT angles if its exact world position is not the "
                             f"point (a gesture/reach-out).")
            return "\n".join(lines)
        # free-only, transient: soft warnings, still a PASS overall
        lines.append("PRE-SIM FREE-LIMB REACH WARNING (soft — the run will still proceed) — a "
                     "FREE (non-contact) limb's authored WORLD target is beyond its reach, so the "
                     "arm cannot get there. At runtime the framework PROJECTS the target onto the "
                     "reachable ball (no flicker), but the limb won't reach your authored point:")
        for n, v in sorted(free.items(), key=lambda kv: -kv[1]["worst_over"]):
            lines.append(f"  - {n}: target {v['worst_over']:.2f}× reach at phase ~{v['worst_phase']}. "
                         f"Author it within reach, move the base nearer, or drive the limb by JOINT "
                         f"angles if its exact world position doesn't matter (a gesture/reach-out).")
        return "\n".join(lines)
    if notarget:
        lines.append("PRE-SIM DECLARATION LINT — an end-effector is DECLARED a contact but "
                     "returns NO world target (compute_ee_position_world_frame -> None) at those "
                     "frames, so IK never places it and it FLOATS. Either give it a world-frame "
                     "target while it is a contact, or remove it from contacts(phase):")
        for n, c in sorted(notarget.items(), key=lambda kv: -kv[1]):
            lines.append(f"  - {n}: declared a contact with no target on ~{c} frames.")
        if not acc and not incompat and not arrivals:
            lines.append("Give every declared contact a world target (or stop declaring it), then resubmit.")
            return "\n".join(lines)
        lines.append("")
    if arrivals:
        lines.append("PRE-SIM ARRIVAL LINT — a swing TELEPORTS into contact (its authored "
                     "world target jumps in one frame, so it never arrives at its foothold). "
                     "The seam bridge will NOT hide a landing, so this fails no_teleport. FIX: "
                     "author the swing to REACH its exact foothold (position + posture) BEFORE "
                     "the phase it is declared a contact — check for a dead/unreachable swing "
                     "branch (a target that stays put then jumps to the contact point):")
        for n, v in sorted(arrivals.items(), key=lambda kv: -kv[1]["jump"]):
            lines.append(f"  - {n}: authored world target JUMPS {v['jump']} m in one frame at "
                         f"phase ~{v['phase']} while declared a contact — the swing did not "
                         f"arrive; ease it into the foothold over the pre-contact window.")
        if not acc and not incompat:
            lines.append("Fix the swing so it lands on its foothold, then resubmit.")
            return "\n".join(lines)
        lines.append("")
    lines.append(f"REACHABILITY LINT (pre-sim, no IK) — {len(acc)} declared contact(s) are "
                 f"BEYOND limb reach in the authored base trajectory, so they will FLOAT "
                 f"once the sim runs. Fix the BASE (the contacts/treads are the datum):")
    if incompat:
        hi, lo = incompat["need_high"], incompat["need_low"]
        lines.append(
            f"  - INCOMPATIBLE CONTACTS at phase ~{incompat['phase']}: '{hi['ee']}' needs "
            f"base z ≥ {hi['min_base_z']} but '{lo['ee']}' needs base z ≤ {lo['max_base_z']} "
            f"— NO single base height reaches both (gap {incompat['gap']} m), so both will "
            f"FLOAT (the seat can only compromise to a midpoint). FIX: don't declare both as "
            f"contacts at once — RELEASE one (retime the contact schedule) so the base can "
            f"track whichever is planted, or move a foothold so the two share a reachable base.")
    for n, v in sorted(acc.items(), key=lambda kv: -kv[1]["worst_over"]):
        if v["horizontal"]:
            lines.append(
                f"  - {n}: beyond HORIZONTAL reach at phase ~{v['worst_phase']} "
                f"(over-reach {v['worst_over']}×). NO base height fixes this — the base "
                f"x/y has not advanced to the foothold. FIX: advance the base so it stays "
                f"within a leg-length of each contact (base xy must track the contacts).")
        else:
            lines.append(
                f"  - {n}: over-reach up to {v['worst_over']}× at phase ~{v['worst_phase']}. "
                f"At that phase the leg cannot reach its world target from base "
                f"z={v['authored_base_z']}. FIX: set base z ∈ [{v['need_lo']}, {v['need_hi']}] "
                f"there (author base_z RELATIVE to the contact surface, e.g. "
                f"base_z = contact_surface_z + a fixed crouch, not an absolute guess).")
    lines.append("Re-author the base trajectory so every contact stays in reach, then resubmit.")
    if free:
        lines.append("")
        lines.append("ALSO — FREE-LIMB REACH (soft, runtime projects it in): a non-contact limb's "
                     "authored world target is beyond reach and will be pulled onto the reachable "
                     "ball (so the arm won't reach your point):")
        for n, v in sorted(free.items(), key=lambda kv: -kv[1]["worst_over"]):
            lines.append(f"  - {n}: {v['worst_over']:.2f}× reach at phase ~{v['worst_phase']} — author "
                         f"it within reach or drive it by JOINT angles if its exact position is moot.")
    return "\n".join(lines)


def main():
    args = load_config()
    try:
        rep = lint(args)
    except Exception as e:
        import traceback
        rep = {"hard_fail": False, "error": str(e),
               "report": f"reach-lint could not run ({e}); proceeding to sim.\n"
                         + traceback.format_exc()}
    print(rep["report"], file=sys.stderr)
    print("LINT_REPORT_JSON: " + json.dumps(rep))
    sys.exit(2 if rep.get("hard_fail") else 0)


if __name__ == "__main__":
    main()
