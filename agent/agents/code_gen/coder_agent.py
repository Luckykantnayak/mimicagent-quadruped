import json
import os
import re

from agents.base_agent import BaseAgent
from langchain_core.messages import SystemMessage, HumanMessage, get_buffer_string
from agents.code_gen.state import AgentState
from utils.robot_spec import load_robot_spec
from utils.tools import extract_json_from_message
from utils.scene_geometry import describe_scene_geometry, robot_start_from_scene

def make_coder_node(coder_agent: BaseAgent, data, cfg=None):
    spec = load_robot_spec(cfg) if cfg else None
    ee_names = spec.ee_names if spec else []
    contact_body_names = spec.contact_body_names if spec else []
    # {environment_name: exemplar source} for non-flat environments — the WORKING
    # hand-authored pattern for that scene, shown to the coder when it applies.
    exemplar_src = {}
    if spec and cfg:
        repo_root = cfg["paths"]["simulator_dir"]
        for ex in spec.exemplars:
            env = ex.get("environment")
            if not env or env == "flat":
                continue
            path = os.path.join(repo_root, "robots", spec.name, "exemplars", f"{ex['name']}.py")
            try:
                with open(path) as f:
                    exemplar_src[env] = (ex["name"], f.read())
            except OSError:
                pass

    repo_root = cfg["paths"]["simulator_dir"] if cfg else None
    # Per-limb reach envelope (generated into robots/<name>/dynamics.json). Lets the
    # plan lint check the plan's OWN numbers: a contact the base cannot reach at that
    # phase is unbuildable, and no coder tuning can fix it.
    reach_env, jump_env = {}, {}
    if spec and cfg:
        try:
            with open(os.path.join(cfg["paths"]["robots_dir"], spec.name,
                                   "dynamics.json")) as _f:
                _dyn = json.load(_f) or {}
            reach_env = _dyn.get("reach") or {}
            jump_env = _dyn.get("jump") or {}
        except (OSError, ValueError):
            reach_env, jump_env = {}, {}
    # Robot's home base pose (scenes are robot-agnostic, so read the robot's OWN
    # keyframe) — so the coder-side scene geometry uses the RIGHT base z, not the
    # (0,0,0.9) default (G2: was silently wrong for a go2 at ~0.27 m).
    robot_xyz = (robot_start_from_scene(os.path.join(repo_root, spec.mjcf))
                 if (spec and spec.mjcf and repo_root) else None)

    def node(state: AgentState) -> AgentState:
        system_message = data["system"]  # str
        human_message = data["human"]  # str

        _base_bodies = [getattr(spec, "base_body", "base"), "pelvis", "torso"] if spec else None
        context = get_context(state, ee_names, repo_root, contact_body_names, robot_xyz,
                              base_bodies=_base_bodies, reach_env=reach_env,
                              jump_env=jump_env)
        env_name = (state.get("environment") or {}).get("name")
        if env_name in exemplar_src:
            name, src = exemplar_src[env_name]
            context += (f"\n\n=== STRUCTURAL REFERENCE for this KIND of scene ({name}) — a curated "
                        f"skill showing the PATTERN (channel choices, attachments, base authoring, "
                        f"phase structure). Follow its STRUCTURE, but take EVERY coordinate (surface "
                        f"heights, x-positions, footprints, the robot's stance) from the CURRENT SCENE "
                        f"GEOMETRY above — NOT from this reference: it may predate the current scene "
                        f"layout, and any number in it that disagrees with the scene geometry is WRONG. "
                        f"Use it for HOW, use the scene geometry for WHERE. ===\n{src}")

        # POST-REPLAN: this generation follows a RE-PLAN (the previous plan was
        # replaced). Don't start from scratch and throw away the repair progress — the
        # best version so far is the CLOSEST attempt. Show it + its readout so this
        # rewrite KEEPS what worked and only changes what the new plan re-decided.
        best = state.get("best_code_path")
        if state.get("replanned") and best and os.path.isfile(best):
            try:
                with open(best) as f:
                    prior = f.read()
                context += ("\n\n=== CLOSEST PRIOR ATTEMPT (best-scoring version so far) — "
                            "the plan above was CORRECTED; keep this code's structure and the "
                            "parts that worked, and change ONLY what the corrected plan re-decided "
                            "(a limb's control channel, a contact's timing, a base decision). Do "
                            "NOT rewrite from scratch. ===\n" + prior)
            except OSError:
                pass
        human_message = human_message.replace('{{MOTION_SPEC_INPUT}}', context)

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message),
        ]

        output = coder_agent.generate(get_buffer_string(messages))  # output type: AIMessage
        # Get Token Usage
        input_tokens = output.response_metadata['usage']['input_tokens']
        output_tokens = output.response_metadata['usage']['output_tokens']

        if state["token_usage"]:
            token_dict = {
                "input_tokens": input_tokens + state["token_usage"]["input_tokens"],
                "output_tokens": output_tokens + state["token_usage"]["output_tokens"],
            }

        else:
            token_dict = {"input_tokens": input_tokens, "output_tokens": output_tokens}

        return {
            "messages": [output],
            "last_prompt": get_buffer_string(messages),   # exact prompt sent (saved per version)
            'skill_name': state['skill_name'],
            'skill_version': state['skill_version'],
            'success': state['success'],
            "num_critic_iters": state['num_critic_iters'],
            'rewrite_mode': state['rewrite_mode'],
            'skill_desc': state['skill_desc'],
            'token_usage': token_dict,
        }

    return node


def get_context(state: AgentState, ee_names=None, repo_root=None, contact_body_names=None,
                robot_xyz=None, base_bodies=None, reach_env=None, jump_env=None):
    motion_plan = ""
    try:
        motion_plan = get_buffer_string([state["messages"][-1]])

    except Exception as e:
        print(e)

    return (motion_plan + _environment_context(state, repo_root, robot_xyz)
            + _duration_directive(motion_plan)
            + _channel_table(motion_plan, ee_names, contact_body_names)
            + _plan_lint(motion_plan, state.get("environment"), contact_body_names,
                         base_bodies=base_bodies, reach_env=reach_env,
                         jump_env=jump_env))


def _duration_directive(plan_text):
    """Surface the plan's chosen wall-clock duration as an explicit self.freq
    directive, so the coder sets the phase rate from the PLAN rather than the 1.0
    default (which compresses a multi-event skill into one second and makes every
    joint teleport / exceed its velocity ceiling)."""
    plan = extract_json_from_message(plan_text)
    if not isinstance(plan, dict):
        return ""
    dur = plan.get("duration_s", plan.get("motion_plan", {}).get("duration_s"))
    try:
        dur = float(dur)
    except (TypeError, ValueError):
        return ("\n\n=== DURATION ===\nThe plan gave no duration_s. CHOOSE a wall-clock"
                " duration from the motion's tempo AND the velocity ceilings (a multi-event"
                " skill is several seconds, not one) and set self.freq = 1.0/duration_s in"
                " __init__ — do NOT leave self.freq at the 1.0 default.")
    if dur <= 1e-6:
        return ""
    return (f"\n\n=== DURATION (from the plan) ===\n"
            f"The whole skill should take duration_s = {dur:g} s. In __init__ set:\n"
            f"    self.freq = 1.0 / {dur:g}   # {1.0/dur:.4g} Hz; one phase 0->1 pass = {dur:g} s\n"
            f"Do NOT leave self.freq at the 1.0 default (that plays the entire skill in one"
            f" second and fails no_teleport / joint_velocity_envelope).")


def _reach_violation(re_ee, base_z, pitch_rad, target_z, off_xy, gate):
    """Can this limb reach `target_z` (world) at horizontal offset `off_xy` from the
    base, with the base at `base_z` and the torso pitched `pitch_rad`?

    Returns None when it can, else (vertical_limit_or_None, detail). `detail` is
    (dist, reach, off_xy, (kind, value)) for a full 3-D check, or None when only the
    vertical bound could be evaluated.

    ONE implementation, shared by every kind of world-fixed target — a planted
    CONTACT and a GRIP are the same geometry problem: a point the limb must arrive at.
    The limb-root offset is BODY-frame, so it rotates with the torso (a shoulder above
    an upright pelvis sits below an inverted one).
    """
    import math as _m
    ro = re_ee.get("root_off")
    if not (isinstance(ro, (list, tuple)) and len(ro) == 3):
        limit = target_z + gate * float(re_ee["reach"]) - float(re_ee.get("root_dz", 0.0))
        return (limit, None) if base_z > limit + 1e-6 else None
    cos, sin = _m.cos(pitch_rad), _m.sin(pitch_rad)
    rx = float(ro[0]) * cos + float(ro[2]) * sin
    ry = float(ro[1])
    rz = -float(ro[0]) * sin + float(ro[2]) * cos
    reach = float(re_ee["reach"])
    if not (isinstance(off_xy, (list, tuple)) and len(off_xy) == 2):
        # No horizontal offset given -> best case, target directly under the limb root.
        limit = target_z - rz + gate * reach
        return (limit, None) if base_z > limit + 1e-6 else None
    dx = float(off_xy[0]) - rx
    dy = float(off_xy[1]) - ry
    dz = (target_z - base_z) - rz
    dist = _m.sqrt(dx * dx + dy * dy + dz * dz)
    if dist <= gate * reach + 1e-6:
        return None
    # Decompose, because the FIX differs. If the purely HORIZONTAL distance already
    # exceeds the reach, no base height helps and the base must move in x/y.
    h = _m.hypot(dx, dy)
    if h > reach:
        fix = ("horizontal", h)
    else:
        fix = ("vertical", target_z - rz + _m.sqrt(max(0.0, reach * reach - h * h)))
    return (None, (dist, reach, off_xy, fix))


def _plan_lint(plan_text, env=None, contact_body_names=None, base_bodies=None,
               reach_env=None, jump_env=None):
    """Deterministic checks on the plan's OWN declarations (not task rules): (1)
    MISSING-ASSET — the plan manipulates an object the chosen environment does not
    provide; (2) INTERNAL CONSISTENCY — an object is manipulated but no limb is
    world-bound to grip it; (3) CONTACT-BODY COMPLETENESS — a FK-only contact body
    (knee) that is a contact in some phases but never appears in free_limb_mode, so
    it is undefined (floats) in the phases it swings. All fire for any morphology."""
    plan = extract_json_from_message(plan_text)
    if not isinstance(plan, dict):
        return ""
    mp = plan.get("motion_plan", plan)
    out = ""

    # (-1) CONTACT REACHABILITY on the plan's OWN numbers. A contact is only real if
    # the limb can reach it from the base authored for that same phase. This is the
    # one plan error no amount of coder tuning can fix — the limb under-reaches, the
    # contact floats, and the pre-sim lint rejects the run — so catch it here, on the
    # plan, before a single line of code is written. Uses base_height_m /
    # surface_height_m, which the planner is required to emit; silently skipped when
    # either is missing so an older plan still runs.
    if reach_env:
        import math as _math
        # Same tolerance the RUNTIME reach-seat uses (controller.REACH_GATE). A physical
        # standing pose legitimately sits at ~100% of leg extension — the humanoid's own
        # home stance measures 1.02x — so flagging at >1.00 reports the robot standing
        # still as an error. Only a genuine OVER-extension counts.
        _GATE = 1.03

        def _pitch_of(entry):
            """Torso pitch in DEGREES for a phase, or None if it cannot be known.

            The limb-root offset is BODY-frame, so the reach bound depends entirely on
            the torso's attitude: the humanoid's shoulder sits 0.48 m ABOVE an upright
            pelvis but 0.48 m BELOW an inverted one, moving the hand's reachable base
            height from H+0.09 to H+1.05. Judging an inverted phase by the upright
            bound would flag a CORRECT cartwheel as out of reach and push the planner
            toward a worse plan — so when the attitude is unknown, say nothing.
            """
            _p = entry.get("torso_pitch_deg")
            if isinstance(_p, (int, float)):
                return float(_p)
            _t = str(entry.get("torso_orientation", "")).lower()
            if _t and ("upright" in _t or "vertical" in _t) and not any(
                    w in _t for w in ("invert", "pitch", "horizontal", "lean", "dive")):
                return 0.0
            return None

        _cs = mp.get("contact_schedule") or plan.get("contact_schedule") or []
        _bm_r = {}
        for _e in ((mp.get("base_motion", {}) or {}).get("per_phase", []) or []):
            _h = _e.get("base_height_m")
            if isinstance(_h, (int, float)):
                _bm_r[tuple(_e.get("phase_range", []))] = (float(_h), _pitch_of(_e))
        _bad = []
        for _seg in _cs:
            if not isinstance(_seg, dict):
                continue
            _got = _bm_r.get(tuple(_seg.get("phase_range", [])))
            if _got is None:
                continue
            _bz, _pitch = _got
            if _pitch is None:
                continue                      # attitude unknown -> cannot judge; stay quiet
            _th = _math.radians(_pitch)
            for _c in (_seg.get("contacts") or []):
                if not isinstance(_c, dict):
                    continue
                _ee = _c.get("ee")
                _sh = _c.get("surface_height_m")
                _re = reach_env.get(_ee)
                if _re is None or not isinstance(_sh, (int, float)):
                    continue
                _cz = float(_sh) + float(_re["ground_offset"])       # contact point world z
                _v = _reach_violation(_re, _bz, _th, _cz, _c.get("offset_from_base_m"), _GATE)
                if _v is not None:
                    _bad.append((_seg.get("phase_range"), _ee, _bz, float(_sh),
                                 _v[0], _pitch, _v[1]))
        if _bad:
            out += ("\n\n⚠️ PLAN LINT — CONTACT OUT OF REACH (the plan's own numbers do not "
                    "close). For each phase below the base is authored HIGHER than the limb "
                    "can reach down to its declared contact, so that contact WILL float no "
                    "matter how you write the code:\n")
            for _pr, _ee, _bz, _sh, _lim, _pi, _h3 in _bad:
                if _h3 is not None:
                    _dist, _rch, _oxy, (_kind, _val) = _h3
                    _head = (f"  - phase {_pr}: {_ee} at offset [{_oxy[0]:+.2f}, {_oxy[1]:+.2f}] m "
                             f"from the base on a surface at {_sh:.2f} m, base {_bz:.2f} m, torso "
                             f"pitched {_pi:.0f} deg -> the limb root is {_dist:.2f} m from that "
                             f"contact but the limb only reaches {_rch:.2f} m "
                             f"({_dist / _rch:.2f}x over). ")
                    if _kind == "horizontal":
                        out += _head + (f"HORIZONTAL: the sideways/forward distance alone is "
                                        f"{_val:.2f} m, already beyond the limb's reach, so NO "
                                        f"base height fixes this — move the base in x/y so it is "
                                        f"OVER the contact, or bring the contact in.\n")
                    else:
                        out += _head + (f"A base height of <= {_val:.2f} m WOULD reach it at this "
                                        f"offset (the plan says {_bz:.2f} m).\n")
                else:
                    out += (f"  - phase {_pr}: {_ee} on a surface at {_sh:.2f} m, torso pitched "
                            f"{_pi:.0f} deg -> base_height_m must be <= {_lim:.2f} m, but the "
                            f"plan says {_bz:.2f} m (over by {_bz - _lim:.2f} m)\n")
            out += ("  FIX: lower the base for that phase, or PITCH/INVERT the torso so the "
                    "limb's root comes down (these offsets are body-frame and rotate with the "
                    "torso), or move the contact to a surface the base can actually reach. "
                    "Implement what the plan MEANT, using a base height that satisfies the "
                    "bound above — do not copy a base height that cannot work.\n")

        # (-1b) GRIP REACHABILITY. A limb that must COINCIDE with something it grips is
        # exactly as reach-constrained as one planted on a surface — same geometry, same
        # law — but it is declared in free_limb_mode (channel "world_reach"), NOT in
        # contact_schedule, so the check above never saw it. Consequence, measured on a
        # carry-box: the hands stopped 0.44 m ABOVE a box resting at z=0.12 (an upright
        # humanoid hand reaches only ~0.01 m below its own pelvis) and the BOX LEVITATED
        # up to meet them. The skill looked like a pick-up and was not one.
        _grip_bad = []
        for _seg in (mp.get("free_limb_mode") or plan.get("free_limb_mode") or []):
            if not isinstance(_seg, dict):
                continue
            _got = _bm_r.get(tuple(_seg.get("phase_range", [])))
            if _got is None:
                continue
            _bz, _pitch = _got
            if _pitch is None:
                continue
            _th = _math.radians(_pitch)
            for _limb, _v in (_seg.get("limbs") or {}).items():
                if not isinstance(_v, dict) or _v.get("channel") != "world_reach":
                    continue
                _tz = _v.get("target_height_m")
                _re = reach_env.get(_limb)
                if _re is None or not isinstance(_tz, (int, float)):
                    continue
                _viol = _reach_violation(_re, _bz, _th, float(_tz),
                                         _v.get("offset_from_base_m"), _GATE)
                if _viol is not None:
                    _grip_bad.append((_seg.get("phase_range"), _limb, _bz, float(_tz),
                                      _viol[0], _pitch, _viol[1]))
        if _grip_bad:
            out += ("\n\n⚠️ PLAN LINT — GRIP OUT OF REACH (the plan's own numbers do not "
                    "close). A limb is asked to reach a point it cannot get to from the base "
                    "authored for that phase. The object will NOT be picked up — it will "
                    "appear to move on its own toward the limb:\n")
            for _pr, _limb, _bz, _tz, _lim, _pi, _h3 in _grip_bad:
                if _h3 is not None:
                    _dist, _rch, _oxy, (_kind, _val) = _h3
                    _head = (f"  - phase {_pr}: {_limb} must reach a point at z={_tz:.2f} m, "
                             f"offset [{_oxy[0]:+.2f}, {_oxy[1]:+.2f}] m from a base at "
                             f"{_bz:.2f} m (torso {_pi:.0f} deg) -> {_dist:.2f} m away but the "
                             f"limb reaches {_rch:.2f} m ({_dist / _rch:.2f}x over). ")
                    out += _head + ((f"HORIZONTAL: {_val:.2f} m of that is sideways/forward, "
                                     f"already past the limb's reach — move the BASE over the "
                                     f"target.\n") if _kind == "horizontal" else
                                    (f"A base height <= {_val:.2f} m would reach it.\n"))
                else:
                    out += (f"  - phase {_pr}: {_limb} must reach z={_tz:.2f} m with the torso "
                            f"at {_pi:.0f} deg -> base_height_m must be <= {_lim:.2f} m, but "
                            f"the plan says {_bz:.2f} m (over by {_bz - _lim:.2f} m)\n")
            out += ("  FIX: the limb must ARRIVE at the object before it moves. Lower the "
                    "base (a floor-level grip needs a deep squat), PITCH the torso forward so "
                    "the shoulders come down, or move the base over the object. Never animate "
                    "the object toward a limb that never got there.\n")

    # (-1c) BALLISTIC FEASIBILITY of a declared flight window. Same class of error as the
    # reach law, on the other axis: the planner picks a flight PHASE and the base heights
    # around it INDEPENDENTLY, and nothing checks that gravity connects them. During
    # flight the base is unpowered, so (z0, z1, T) are not free — they must satisfy
    # z1 = z0 + v0*T - g*T^2/2 for a launch velocity v0 the legs can actually produce.
    # Measured on a ledge drop: 1.63 -> 0.65 m declared over 1.20 s needs a +5.07 m/s
    # launch against a 2.20 m/s maximum, so the "jump" was really a slow lowering at
    # -2.2 m/s^2 and support_or_ballistic rejected it for the whole clip.
    if jump_env and isinstance(jump_env.get("height"), (int, float)):
        import math as _m2
        _g = 9.81
        _vmax = _m2.sqrt(2.0 * _g * float(jump_env["height"]))
        _dur = mp.get("duration_s") or plan.get("duration_s")
        _bm_h = {tuple(e.get("phase_range", [])): e.get("base_height_m")
                 for e in ((mp.get("base_motion", {}) or {}).get("per_phase", []) or [])}
        _ordered = [pr for pr in _bm_h if len(pr) == 2]
        _ordered.sort(key=lambda p: p[0])
        _fl = []
        if isinstance(_dur, (int, float)) and _dur > 0:
            for _seg in (mp.get("contact_schedule") or plan.get("contact_schedule") or []):
                if not isinstance(_seg, dict) or (_seg.get("contacts") or []):
                    continue                      # only phases declaring FLIGHT ([] contacts)
                _pr = tuple(_seg.get("phase_range", []))
                _z0 = _bm_h.get(_pr)
                _nxt = next((p for p in _ordered if p[0] >= _pr[1] - 1e-9), None)
                _z1 = _bm_h.get(_nxt) if _nxt else None
                if not (isinstance(_z0, (int, float)) and isinstance(_z1, (int, float))):
                    continue
                _T = (float(_pr[1]) - float(_pr[0])) * float(_dur)
                if _T <= 1e-6:
                    continue
                _v0 = (float(_z1) - float(_z0) + 0.5 * _g * _T * _T) / _T
                if _v0 > _vmax * 1.05:
                    _drop = float(_z0) - float(_z1)
                    _tmin = (_m2.sqrt(max(0.0, 2.0 * _g * _drop)) / _g) if _drop > 0 else 0.0
                    _tmax = (_vmax + _m2.sqrt(max(0.0, _vmax * _vmax + 2.0 * _g * _drop))) / _g
                    _fl.append((_pr, _z0, _z1, _T, _v0, _tmin, _tmax))
        if _fl:
            out += ("\n\n⚠️ PLAN LINT — FLIGHT WINDOW NOT BALLISTIC (the plan's own numbers do "
                    "not close). In flight the base is unpowered: it can only follow "
                    "z = z0 + v0*t - g*t^2/2. The window below is too LONG for its height "
                    "change, so holding it would mean hanging in the air:\n")
            for _pr, _z0, _z1, _T, _v0, _tmin, _tmax in _fl:
                out += (f"  - phase {_pr}: {_z0:.2f} m -> {_z1:.2f} m over {_T:.2f} s needs a "
                        f"launch of {_v0:+.2f} m/s, but this robot can only push "
                        f"{_vmax:.2f} m/s ({_v0 / _vmax:.1f}x). Physically this flight lasts "
                        f"{_tmin:.2f} s as a pure drop, up to {_tmax:.2f} s with a full jump.\n")
            out += ("  FIX: SHORTEN the flight phase_range to the physical duration above "
                    "(the rest of the motion keeps the time), or raise the take-off / lower "
                    "the landing so the fall actually takes that long. Do not stretch a fall "
                    "— a slow descent through a []-contact phase is not flight and "
                    "support_or_ballistic will reject the whole window.\n")

    # (0) BASE-IS-A-CONTACT HEIGHT MISMATCH. When a contact body IS the base (the pelvis /
    # torso the root joint rides on), its height IS the base height — no limb "reaches" it.
    # A plan that rests the pelvis on a raised surface while describing a STANDING torso for
    # that same phase floats the pelvis by exactly the difference (measured: SITCHAIR's pelvis
    # hovered 50 cm for 100% of its declared frames). Catch it BEFORE simulating: parse the
    # phase's torso_height text and compare it against the surface the base-body contact
    # names. Purely a consistency check on the plan's OWN numbers.
    _bases = {b.lower() for b in (base_bodies or ["pelvis", "torso", "base"])}
    _surf_h = {}
    for _o in ((env or {}).get("objects") or []):
        _t = _o.get("top_z", _o.get("height"))
        if isinstance(_t, (int, float)):
            _surf_h[str(_o.get("name", "")).lower()] = float(_t)
    _bm = {tuple(e.get("phase_range", [])): e for e in (mp.get("base_motion", {}) or {}).get("per_phase", [])}
    for _e in (mp.get("contact_schedule") or []):
        for _c in (_e.get("contacts") or []):
            _ee = str(_c.get("ee", "")).lower()
            if _ee not in _bases:
                continue
            _srf = str(_c.get("surface", "")).lower()
            _h = next((v for k, v in _surf_h.items() if k and (k in _srf or _srf in k)), None)
            _phase = _bm.get(tuple(_e.get("phase_range", [])))
            _txt = str((_phase or {}).get("torso_height", ""))
            _nums = re.findall(r"[-+]?\d*\.\d+|\d+", _txt)
            if _h is None or not _nums:
                continue
            _tz = float(_nums[0])
            if abs(_tz - _h) > 0.20:      # generous: offsets + posture slack
                out += (f"\n\n*** BASE-IS-A-CONTACT MISMATCH ***\n"
                        f"Phase {_e.get('phase_range')} declares '{_c.get('ee')}' — a BASE body — resting on "
                        f"'{_c.get('surface')}' (top ~{_h:.2f} m), but base_motion for that phase says "
                        f"torso_height '{_txt}' (~{_tz:.2f} m). The base body's height IS the base height: no "
                        f"limb reaches it. Author the base at ~{_h:.2f} m + that body's ground_offset for this "
                        f"phase, or the contact will hover ~{abs(_tz-_h):.2f} m and FAIL contact_grounded for "
                        f"100% of its frames. (Fix the BASE, not the limb.)")

    # (3) CONTACT-BODY COMPLETENESS (runs regardless of objects). A contact body
    # (knee/shin) is FK-only; when it is a contact it goes in contacts(phase), and when
    # it swings it must be driven via free_limb_mode (its leg joints). If it is a
    # contact in SOME phases but ABSENT from free_limb_mode entirely, its swing phases
    # are undefined and it floats (the observed plan bug: the swing knee was omitted).
    cbodies = set(contact_body_names or [])
    cs_all = mp.get("contact_schedule") or []
    fm_all = mp.get("free_limb_mode") or []
    if cbodies and cs_all:
        contact_cnt = {}
        for e in cs_all:
            for c in (e.get("contacts") or []):
                if c.get("ee") in cbodies:
                    contact_cnt[c["ee"]] = contact_cnt.get(c["ee"], 0) + 1
        in_free = {n for e in fm_all for n in (e.get("limbs") or {}) if n in cbodies}
        for body, cnt in contact_cnt.items():
            if cnt < len(cs_all) and body not in in_free:
                out += (f"\n\n*** PLAN COMPLETENESS WARNING ***\n"
                        f"Contact body '{body}' is a contact in some phases but is ABSENT from "
                        f"free_limb_mode entirely — in the phases it is NOT a contact (it lifts/"
                        f"swings) it has no channel, so it is undefined and will FLOAT. When it IS "
                        f"a contact, place it by IK (return its world contact point from "
                        f"compute_ee_position_world_frame — it is now in the union IK roster). For its "
                        f"SWING windows add '{body}' to free_limb_mode (body_swing to carry it through "
                        f"the air, or joint_posture to shape its leg), or keep it a contact throughout.")

    obj_plan = mp.get("object_plan") or []
    if not obj_plan:
        return out

    # (1) MISSING SCENE ASSET: the plan wants an object the environment lacks.
    avail = {o.get("name") for o in ((env or {}).get("objects") or [])}
    want = [str(o.get("object")) for o in obj_plan if o.get("object")]
    missing = [n for n in want if n not in avail]
    if missing:
        out += ("\n\n*** MISSING SCENE ASSET ***\n"
                f"The plan manipulates object(s) {missing}, but the chosen environment "
                f"'{(env or {}).get('name', 'flat')}' provides NO such object "
                f"(available: {sorted(avail) or 'none'}). The object will not exist in "
                "the scene, so it cannot be gripped or moved. Register a scene that "
                "contains it: add assets/scenes/<env>.xml with the object (a mocap body "
                "for a carried item, a static geom for a fixed one) and list it under "
                "`environments:` in the robot's robot.yaml, then point the plan's "
                "environment at that scene.")

    # (2) INTERNAL CONSISTENCY: nothing grips the object -> it floats.
    free = mp.get("free_limb_mode") or []
    world_bound = any((v or {}).get("channel") == "world_reach"
                      for e in free for v in (e.get("limbs") or {}).values())
    if not world_bound:
        names = ", ".join(str(o.get("object")) for o in obj_plan)
        out += ("\n\n*** PLAN CONSISTENCY WARNING ***\n"
                f"The plan manipulates object(s) [{names}] but declares NO free limb with"
                " the world_reach channel — nothing grips them, so any lifted object will"
                " FLOAT. A limb that holds/grips an object must be WORLD-IK to it"
                " (compute_ee_position_world_frame at the grip face) AND listed in"
                " attachments(phase) so the object tracks the real limb. Implement a"
                " gripping limb for the held window, OVERRIDING the plan's channel for"
                " that limb if the plan mislabeled it.")
    return out


# ---------------------------------------------------------------------------
# Deterministic channel expander: contact_schedule + free_limb_mode -> an
# explicit per-phase, per-limb table of WHICH control channel (method) the coder
# must implement. This removes the coder's guesswork about world-IK vs body-IK vs
# joint-space — the planner declares intent, this maps it to the concrete method.
# ---------------------------------------------------------------------------
_CHANNEL_METHOD = {
    "world_reach":   "compute_ee_position_world_frame -> absolute WORLD target (NOT a contact)",
    "body_swing":    "compute_ee_position_body_frame -> target RELATIVE to the base",
    "joint_posture": "return None for this EE; set its joints in compute_joint_overrides",
    "none":          "leave at rest (body-frame default) — return None",
}


def _in_range(phase, rng):
    try:
        return float(rng[0]) <= phase < float(rng[1])
    except Exception:
        return False


def _channel_table(plan_text, ee_names, contact_body_names=None):
    plan = extract_json_from_message(plan_text)
    if not isinstance(plan, dict):
        return ""
    mp = plan.get("motion_plan", plan)
    contact_sched = mp.get("contact_schedule") or []
    free_mode = mp.get("free_limb_mode") or []
    if not contact_sched and not free_mode:
        return ""   # old-style plan; nothing to expand

    # Unified phase partition from EVERY schedule's boundaries — contact_schedule,
    # free_limb_mode, phase_definition, and base_motion — so a window can't fall
    # between declared ranges (the planner's ONE-boundary-set law should make these
    # coincide; unioning is defensive).
    sub_phases = (mp.get("phase_definition", {}) or {}).get("sub_phases") or []
    base_pp = (mp.get("base_motion", {}) or {}).get("per_phase") or []
    bounds = set()
    for e in contact_sched + free_mode + sub_phases + base_pp:
        r = e.get("phase_range") or []
        if len(r) == 2:
            bounds.update((float(r[0]), float(r[1])))
    # Cluster near-equal boundaries (an LLM writes 0.333 in one schedule and
    # 0.3333333 in another for the SAME 1/3 event) so rounding differences don't
    # create a phantom sliver window + a spurious GAP warning. eps << the smallest
    # real sub-phase (~0.05), so a genuine hole (a true 0.4->0.6 gap) still fires.
    _EPS = 5e-3
    _raw = sorted(bounds) or [0.0, 1.0]
    bounds = [_raw[0]]
    for _x in _raw[1:]:
        if _x - bounds[-1] > _EPS:
            bounds.append(_x)
    ees = list(ee_names) if ee_names else []
    cbodies = set(contact_body_names or [])
    # Roster = end-effectors + spec-declared CONTACT BODIES (knees/shins/...). The
    # contact bodies were previously DROPPED (`roster = ees or ...` never fell through
    # since ees is non-empty), so the coder was handed a "implement EXACTLY" table
    # that OMITTED a crawl's most important contacts. Include them, rendered as the
    # FK-only contacts they are.
    roster_all = list(ees) + [n for n in sorted(cbodies) if n not in ees]

    lines = ["\n\n=== CHANNEL TABLE (derived from contact_schedule + free_limb_mode) ===",
             "For each phase window, implement EXACTLY the channel listed for each limb:"]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a <= _EPS:
            continue                              # drop any degenerate sliver window
        mid = (a + b) / 2.0
        cs = next((c for c in contact_sched if _in_range(mid, c.get("phase_range", []))), None)
        fm = next((f for f in free_mode if _in_range(mid, f.get("phase_range", []))), None)
        contacts = {c["ee"]: c for c in (cs.get("contacts") if cs else []) if c.get("ee")}
        limbs = (fm.get("limbs") if fm else {}) or {}
        lines.append(f"\n  phase [{a:.2f}, {b:.2f}):")
        if cs is None:
            # GAP: no contact_schedule entry covers this window. Do NOT silently treat
            # it as flight — the planner OMITTED contacts here. Surface the omission so
            # the coder declares them, instead of implementing a phantom aerial. (A real
            # flight window is a contact_schedule entry that explicitly declares contacts: [].)
            lines.append("    contacts(phase) -> ⚠️ GAP: contact_schedule declares NO entry "
                         "for this window — this is NOT auto-flight. If the robot is airborne "
                         "here, add a contact_schedule entry with contacts: []; otherwise "
                         "declare the resting contacts for this window.")
        else:
            contact_names = [n for n in (roster_all or list(contacts)) if n in contacts]
            lines.append(f"    contacts(phase) -> {contact_names if contact_names else '[]  (flight/aerial — declared)'}")
        # roster = the known EEs + contact bodies, plus any name the planner used that
        # isn't in the spec roster (defensive, so nothing declared is silently omitted)
        roster = roster_all + [n for n in sorted(set(contacts) | set(limbs)) if n not in roster_all]
        for ee in roster:
            if ee in contacts:
                c = contacts[ee]
                pin = "PLANTED at a fixed world point" if c.get("mode") != "pivot" else "PIVOT (fixed point, may spin)"
                if ee in cbodies:
                    lines.append(f"      - {ee}: WORLD CONTACT ({pin}) on {c.get('surface','floor')} — a "
                                 f"contact BODY, now part of the UNION IK ROSTER (IK-PLACEABLE like a foot): "
                                 f"return its contact point from compute_ee_position_world_frame('{ee}', phase) "
                                 f"(z = surface + its ground_offset) AND list it in contacts(phase). IK then "
                                 f"bends the limb to PLACE the {ee} geom on the surface — no hand-solving joints. "
                                 f"(A near-base body — pelvis/torso/head — is driven by the BASE pose instead: "
                                 f"author update_base_motion so it rests, and still list it in contacts(phase) "
                                 f"for the check. Joint-overrides remain an option if the exact geom pose matters.)")
                else:
                    lines.append(f"      - {ee}: WORLD CONTACT — {pin} on {c.get('surface','floor')}; "
                                 f"compute_ee_position_world_frame returns that point, and LIST it in contacts(phase)")
            else:
                ch = (limbs.get(ee) or {}).get("channel", "none")
                intent = (limbs.get(ee) or {}).get("intent", "")
                lines.append(f"      - {ee}: {_CHANNEL_METHOD.get(ch, _CHANNEL_METHOD['none'])}"
                             + (f"  [{intent}]" if intent else ""))
    return "\n".join(lines)


def _environment_context(state: AgentState, repo_root=None, robot_xyz=None):
    """Concrete scene/object facts for the chosen environment, so the coder can
    animate objects (compute_object_poses) at exact rest positions and grip
    points. Empty for the flat default (no scene override, no objects)."""
    env = state.get("environment") or {}
    if not env or env.get("name") in (None, "flat"):
        return ""
    lines = [
        "\n\n=== ENVIRONMENT (chosen by the planner) ===",
        f'scene: "{env.get("name")}" — {env.get("description", "").strip()}',
        f'grounding: {env.get("grounding", "contact")}'
        + ("  (base re-grounded to contacts each frame)"
           if env.get("grounding", "contact") == "contact"
           else "  (OFF — you set the ABSOLUTE base trajectory in update_base_motion)"),
    ]
    # Exact geometry of every scene object + its distance from the robot, read
    # from the scene XML — so contacts land on the REAL treads/ledge (not guessed).
    scene = env.get("scene")
    geom = (describe_scene_geometry(os.path.join(repo_root, scene), robot_xyz=robot_xyz)
            if (repo_root and scene) else "")
    if geom:
        lines.append("\n" + geom)
    objs = env.get("objects") or []
    if objs:
        lines.append(
            "objects: place each at rest / along its travel with compute_object_poses"
            "(phase) -> {name: (world_pos(3), quat_wxyz(4))}. While an object is HELD,"
            " declare attachments(phase) -> {name: [ee_name, ...]} so the framework"
            " pins it to the ACTUAL (achieved) gripping limbs — the object then"
            " coincides with the real hands BY CONSTRUCTION, whatever channel drives"
            " them. Do NOT drive the hands one way and guess the object's pose"
            " separately; they will not meet. PREFER attachments over an independent"
            " scripted object curve: turning an attachment ON (pick) or OFF (place) is"
            " now crossfaded by the framework, so it no longer teleports the object at"
            " the grip/release seam — the old reason to hand-script the box is gone."
            "\n  PICK IT UP FROM WHERE IT RESTS (reach DOWN to the object — do not let"
            " it fly up to a high hand): attachments() places the object at the hands'"
            " ACHIEVED position, so if the hands never reached the object, the object"
            " simply TELEPORTS up to them and the grip is fake (fails object_grip). So"
            " in the phase BEFORE you attach, author the holding hands' WORLD targets AT"
            " the object's CURRENT rest pose (its grip faces) — hand target = a function"
            " of the OBJECT's pose, NEVER of the pelvis — and only turn the attachment ON"
            " once the hands have arrived there. If the object rests low and the hands"
            " can't reach it from a standing base, LOWER THE BASE to it: author"
            " update_base_motion to crouch (drop pelvis z AND pitch the torso forward)"
            " so the shoulders come down to the object — do NOT clamp the hand target"
            " toward the body (that never touches the object). Reach down, grip, THEN lift."
            "\n  APPROACH FROM A DISTANCE (objects sit FAR ahead): WALK up to the object and"
            " STOP a short step in FRONT of it (pelvis ~0.3-0.4 m BEHIND the object) — do NOT"
            " stand ON TOP of it and squat straight down (that folds the legs and drives them"
            " through the floor). Reach FORWARD-AND-DOWN to it: pitch the torso forward and"
            " extend the arms forward+down, with the FEET planted BEHIND the object (author the"
            " feet as world contacts at z=ground_offset and let IK bend the knees — do not move"
            " the feet down with the pelvis). After lifting, carry forward and PLACE the object"
            " a bit FARTHER ahead than where it was picked up (not back at the start)."
            "\n  CARRY IT CLEAR OF THE BODY: a held object must not clip THROUGH the torso"
            " or pelvis — the object_penetration check fails a box carried inside the"
            " body. Any hold that keeps the object out of the torso passes (out in front"
            " at arm's reach, to the side, or on a shoulder); front-carry is just the"
            " simplest to author.")
        for o in objs:
            hs = o.get("half_size")
            lines.append(
                f'  - "{o["name"]}" ({o.get("type","box")}'
                + (f", half_size {hs}" if hs is not None else "")
                + f", rests at {o.get('rest_pos')}): {o.get('description','').strip()}")
    return "\n".join(lines)
