"""
prompts/prompt_loader.py

Loads the prompt templates (prompts/templates/*.txt) and fills the morphology
placeholders from the robot spec (robots/<name>/robot.yaml).

Template format:
    === SYSTEM ===
    <system prompt>
    === HUMAN ===
    <human prompt>

Morphology placeholders (substituted here, via str.replace — templates
contain literal JSON braces, so .format() is never used):
    {robot}              robot name
    {urdf}               full MJCF/URDF text
    {end_effectors}      comma-separated end-effector names
    {num_end_effectors}  end-effector count
    {joint_order}        comma-separated actuated joint names, in order
    {joint_limits}       one "name: [lo, hi]" line per actuated joint
    {base_height_range}  e.g. "0.10 m to 0.68 m"
    {morphology_card}    free-text robot card from the spec
    {example_skill}      reference skill implementation source
    {base_class_source}  gaits/base.py source (the contract skills must follow)

Per-call placeholders like {{MOTION_SPEC_INPUT}}, {SKILL_NAME},
{CURRENT_SKILL_CODE} are left untouched — the agent nodes fill those.
"""

import json
import math
import os
import yaml

from utils.robot_spec import load_robot_spec
from utils.scene_geometry import describe_scene_geometry, robot_start_from_scene

def load_yaml_config(yaml_path="config.yaml"):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = os.path.dirname(os.path.dirname(os.path.abspath(yaml_path)))
    for key, value in cfg.get("paths", {}).items():
        if not os.path.isabs(value):
            cfg["paths"][key] = os.path.normpath(os.path.join(root, value))

    return cfg


def _read_text(path, what):
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return f"<{what} not found at {path}>"


def _robot_summary(cfg, spec):
    """Physical summary (limb lengths, reach, optional dynamics), pre-generated
    once by simulator/tools/gen_robot_summary.py. Optional — omitted if absent."""
    path = os.path.join(cfg["paths"]["robots_dir"], spec.name, "summary.md")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "(No physical summary generated for this robot.)"


def _build_substitutions(cfg):
    spec = load_robot_spec(cfg)
    repo_root = cfg["paths"]["simulator_dir"]

    joint_limits = "\n".join(
        f"{name}: [{lo}, {hi}]" for name, (lo, hi) in spec.joint_limits.items()
    )
    lo, hi = spec.base_height_range

    # Support set: the end-effectors that are ground contacts (feet). The rest
    # are "free" (hands, or limbs the skill raises) — authored without contact.
    support = [spec.ee_names[i] for i in spec.contact_ee_indices]
    free = [n for n in spec.ee_names if n not in support]
    extra_contacts = (", ".join(spec.contact_body_names)
                      if spec.contact_body_names else "(none)")

    # EXPLICIT per-contact ground_offset table (body origin -> contact point). The
    # single most-repeated authoring bug is a contact authored at the SURFACE height
    # (z=0 on the floor, z=tread_top on a ledge) instead of surface + ground_offset,
    # which drives the sole a ground_offset BELOW the surface (penetration) or, when a
    # coder invents a wrong offset, floats it. Give the coder the exact numbers + the
    # arithmetic so it never re-derives them.
    _off_lines = [f"  {ee['name']}: ground_offset = {float(ee.get('ground_offset', 0.0)):.4f} m"
                  for ee in spec.end_effectors]
    _off_lines += [f"  {c['name']}: ground_offset = {float(c.get('ground_offset', 0.0)):.4f} m"
                   for c in spec.contact_bodies]
    contact_offsets = ("\n".join(_off_lines) +
        "\n  RULE: a DECLARED CONTACT's world-frame z = (its surface height) + (its ground_offset)."
        "\n  On the flat floor that is z = ground_offset (NOT 0.0). On a raised tread/ledge/seat"
        " at height H it is z = H + ground_offset. NEVER author a contact at the bare surface"
        " height, and NEVER invent an offset — use the exact value above.")

    # Environments the skill may run in (the planner picks ONE by name). Each
    # non-flat scene is annotated with its EXACT object geometry + robot distance,
    # read straight from the scene XML — so the planner targets the real surfaces
    # (e.g. the true staircase top at z=0.75) instead of guessing.
    # Scenes are robot-agnostic now (no keyframe), so the robot's home base comes
    # from its MODEL keyframe (spec.mjcf), passed in for the robot-to-object distances.
    robot_xyz = robot_start_from_scene(os.path.join(repo_root, spec.mjcf)) if spec.mjcf else None
    env_lines = []
    for e in spec.all_environments:
        objs = e.get("objects") or []
        obj_str = ("; objects: " + ", ".join(o["name"] for o in objs)) if objs else ""
        env_lines.append(f'- "{e["name"]}": {e.get("description", "").strip()}{obj_str}')
        scene = e.get("scene")
        geom = describe_scene_geometry(os.path.join(repo_root, scene), robot_xyz=robot_xyz) if scene else ""
        if geom:
            env_lines.append("    " + geom.replace("\n", "\n    "))
    environments = "\n".join(env_lines)

    # Curated exemplar catalog — proven hand-authored patterns to draw on.
    ex_lines = []
    for ex in spec.exemplars:
        tags = ", ".join(ex.get("tags", []))
        ex_lines.append(f'- {ex["name"]} [{tags}] in "{ex.get("environment","flat")}": '
                        f'{ex.get("summary", "").strip()}')
    exemplars = ("\n".join(ex_lines) if ex_lines
                 else "(no curated exemplars registered for this robot)")

    return {
        "{robot}": spec.name,
        "{support_set}": ", ".join(support) if support else "(none)",
        "{free_ees}": ", ".join(free) if free else "(none)",
        "{urdf}": _read_text(os.path.join(repo_root, spec.mjcf), "URDF/MJCF"),
        "{robot_summary}": _robot_summary(cfg, spec),
        "{end_effectors}": ", ".join(spec.ee_names),
        "{num_end_effectors}": str(len(spec.ee_names)),
        "{joint_order}": ", ".join(spec.actuated_joint_names),
        "{joint_limits}": joint_limits,
        "{base_height_range}": f"{lo} m to {hi} m",
        "{morphology_card}": spec.robot_card.strip(),
        "{example_skill}": _read_text(
            os.path.join(repo_root, spec.example_skill), "example skill"
        ),
        "{base_class_source}": _read_text(
            os.path.join(repo_root, "simulator", "motion", "base.py"), "base class"
        ),
        "{environments}": environments,
        "{exemplars}": exemplars,
        "{contact_bodies}": extra_contacts,
        "{contact_offsets}": contact_offsets,
        "{reach_envelope}": _reach_envelope(cfg, spec),
    }


def _reach_envelope(cfg, spec):
    """The CONTACT-REACHABILITY table: for each limb, how far below the base its
    contact point can actually get, and therefore how high the base may be when that
    limb is planted on a surface.

    This exists because the planner used to choose a contact schedule and a base
    trajectory independently and never check they were compatible. They frequently
    were not: a humanoid hand reaches only ~0.05 m below an upright pelvis (the
    shoulder sits ABOVE the base), so 'plant a hand on the floor' is unbuildable from
    any upright pose no matter how the coder writes it. The lint caught that
    afterwards; nothing made the planner reason about it up front.
    """
    path = os.path.join(cfg["paths"]["robots_dir"], spec.name, "dynamics.json")
    try:
        with open(path) as f:
            env = (json.load(f) or {}).get("reach") or {}
    except (OSError, ValueError):
        env = {}
    if not env:
        return ("(No reach envelope generated for this robot — run "
                "simulator/tools/gen_robot_summary.py.)")
    lines = [
        "For an UPRIGHT base, each limb's contact point reaches only this far BELOW "
        "the base origin:",
    ]
    for name, v in env.items():
        lines.append(f"  {name}: reaches {v['drop_below_base_upright']:.2f} m below the base "
                     f"(limb root {v['root_dz']:+.2f} m from base, limb reach {v['reach']:.2f} m)"
                     f"  ->  planted on a surface at height H, base_z must be "
                     f"<= H + {v['max_base_over_surface']:.2f}")
    lines.append("")
    # The same bound, evaluated at a few torso pitches — the numbers that matter for
    # an acrobatic skill, where the limb roots rotate with the torso.
    _pitch_rows = []
    for name, v in env.items():
        ro = v.get("root_off")
        if not (isinstance(ro, (list, tuple)) and len(ro) == 3):
            continue
        cells = []
        for deg in (0, 90, 180):
            th = math.radians(deg)
            rz = -math.sin(th) * float(ro[0]) + math.cos(th) * float(ro[2])
            cells.append(f"{deg}deg: H+{float(v['ground_offset']) + float(v['reach']) - rz:.2f}")
        _pitch_rows.append(f"  {name}: " + "   ".join(cells))
    if _pitch_rows:
        lines.append("The SAME bound at different torso pitches (0 = upright, 90 = torso "
                     "horizontal, 180 = inverted) — max base_z for a contact on a surface "
                     "at height H:")
        lines.extend(_pitch_rows)
        lines.append("")
    lines.append(
        "  PITCH CHANGES THESE NUMBERS. The limb-root offsets are BODY-frame, so they "
        "rotate with the torso: a shoulder ABOVE an upright pelvis sits BELOW an "
        "inverted one. That is exactly HOW a cartwheel or handstand reaches the "
        "ground — by INVERTING, not by squatting. Give torso_pitch_deg for every "
        "phase so this can be checked mechanically.")
    lines.append("")
    lines.append(
        "  HEIGHT IS ONLY HALF OF IT. A contact can be out of reach SIDEWAYS while its "
        "height is fine, and no base height fixes that — the base itself has to be over "
        "the contact. So give offset_from_base_m [dx, dy] for every contact (+x = the "
        "robot's forward, +y = its left) and keep the FULL 3-D distance from the limb "
        "root to the contact within the limb's reach. In practice: a planted hand "
        "belongs roughly UNDER its own shoulder, not out to the side — if the skill "
        "needs it out wide, the BASE must travel there too.")
    lines.append("")
    lines.append(
        "  HOW TO USE THIS: pick the contact FIRST (it is fixed by the world), then "
        "derive the base height that can reach it — never the other way round. If a "
        "contact you want is BELOW what an upright base can reach, that is not a "
        "tuning problem: the plan MUST either lower the base or PITCH/INVERT the "
        "torso, because these offsets are BODY-frame and rotate with it (a shoulder "
        "above an upright pelvis swings down to pelvis height at 90 deg of pitch). "
        "Say which, explicitly, in base_motion.")
    return "\n".join(lines)


def load_prompt(name, cfg):
    """Load prompts/templates/<name>.txt and fill the morphology placeholders."""
    prompt_path = os.path.join(cfg["paths"]["prompts_dir"], "templates", f"{name}.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "=== HUMAN ===" in content:
        system, human = content.split("=== HUMAN ===", 1)
    else:
        system, human = content, ""
    system = system.split("=== SYSTEM ===", 1)[1]

    data = {"system": system.strip(), "human": human.strip()}

    substitutions = _build_substitutions(cfg)
    for key in ("system", "human"):
        for placeholder, value in substitutions.items():
            data[key] = data[key].replace(placeholder, value)

    return data
