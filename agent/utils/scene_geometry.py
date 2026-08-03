"""
Scene-geometry extractor — makes the agent AWARE of every extra object a skill's
scene adds (stairs treads, a climb slab, a carry box): its EXACT geometry and its
distance from the robot's start pose.

Why this exists: the coder used to receive only an environment NAME ("stairs") and
a hand-written object blurb — never the real per-step tread heights/positions. So
it GUESSED the geometry: it climbed 2 of 5 steps, planted feet where no tread was,
and modeled a finite staircase as a cyclic gait. This reads the ground truth
straight from the scene XML so contacts land on the REAL surfaces.

Pure XML parse (no MuJoCo needed at prompt-build time). Boxes are handled exactly;
other primitives are approximated by their bounding extent.
"""

import os
import xml.etree.ElementTree as ET


def _floats(s, n=None):
    v = [float(x) for x in (s or "").split()]
    if n is not None:
        v = (v + [0.0] * n)[:n]
    return v


def _box_obj(geom, name, center, movable):
    """Build an object record for a box geom. `center` is its world center."""
    half = _floats(geom.get("size"), 3)
    gpos = _floats(geom.get("pos", "0 0 0"), 3)
    cx, cy, cz = (center[0] + gpos[0], center[1] + gpos[1], center[2] + gpos[2])
    sx, sy, sz = half
    return dict(
        name=name, movable=movable, geom="box",
        center=(cx, cy, cz), half=(sx, sy, sz),
        top=cz + sz, bottom=cz - sz,
        x_range=(cx - sx, cx + sx), y_range=(cy - sy, cy + sy),
    )


def parse_scene_objects(scene_xml_path):
    """Every NON-robot object in the scene worldbody (the robot arrives via
    <include> and lives in nested bodies, so it is naturally excluded). Static
    surfaces are top-level <geom>; movable manipulands are <body mocap="true">."""
    tree = ET.parse(scene_xml_path)
    root = tree.getroot()
    wb = root.find("worldbody")
    if wb is None:
        return []
    objs = []
    for g in wb.findall("geom"):
        if g.get("type") == "plane" or not g.get("size"):
            continue                                   # floor / non-sized -> skip
        objs.append(_box_obj(g, g.get("name", "geom"), (0.0, 0.0, 0.0), movable=False))
    for b in wb.findall("body"):
        movable = str(b.get("mocap", "")).lower() in ("true", "1")
        bpos = _floats(b.get("pos", "0 0 0"), 3)
        for g in b.findall("geom"):
            if not g.get("size"):
                continue
            objs.append(_box_obj(g, b.get("name") or g.get("name", "body"), bpos, movable))
    return objs


def robot_start_from_scene(scene_xml_path, default=(0.0, 0.0, 0.9)):
    """Robot base start (x,y,z) from a 'home' keyframe's qpos (first 3). Reads any
    XML with a `<keyframe>` — since scenes are now robot-agnostic (no keyframe), pass
    the ROBOT MODEL path (spec.mjcf), which owns the keyframe. Returns `default` when
    no keyframe is present (e.g. an env-only scene queried directly)."""
    try:
        root = ET.parse(scene_xml_path).getroot()
        kf = root.find("keyframe")
        if kf is not None:
            for key in kf.findall("key"):
                if key.get("name") == "home" and key.get("qpos"):
                    return tuple(_floats(key.get("qpos"), 3))
            key = kf.find("key")
            if key is not None and key.get("qpos"):
                return tuple(_floats(key.get("qpos"), 3))
    except Exception:
        pass
    return default


def _step_up_profile(statics):
    """From stacked/nested static boxes, derive the EXPOSED tread of each: the x
    band where that box is the topmost surface. Generalizes a staircase (5 bands)
    and a single slab (1 band). Returns [(name, x_lo, x_hi, top_z)] low->high."""
    ss = sorted(statics, key=lambda o: o["top"])
    prof = []
    for i, o in enumerate(ss):
        near, far = o["x_range"]
        # the tread is exposed until a HIGHER box starts covering it from the front
        higher_nears = [h["x_range"][0] for h in ss[i + 1:]
                        if h["x_range"][0] > near and h["x_range"][0] < far]
        exposed_far = min(higher_nears) if higher_nears else far
        prof.append((o["name"], near, exposed_far, o["top"]))
    return prof


def describe_scene_geometry(scene_xml_path, robot_xyz=None):
    """Agent-readable exact geometry + robot-to-object distances for one scene.
    Returns '' if the scene has no extra objects (a bare floor)."""
    if not scene_xml_path or not os.path.isfile(scene_xml_path):
        return ""
    objs = parse_scene_objects(scene_xml_path)
    if not objs:
        return ""
    rx, ry, rz = robot_xyz or robot_start_from_scene(scene_xml_path)

    lines = [
        "SCENE OBJECT GEOMETRY (exact, world frame, metres — author contacts on "
        "these REAL surfaces, not guessed ones):",
        f"  robot base starts at (x={rx:.2f}, y={ry:.2f}, z={rz:.2f}).",
    ]
    for o in sorted(objs, key=lambda o: o["x_range"][0]):
        near = o["x_range"][0]
        gap = near - rx
        kind = ("MOVABLE (mocap — you position it each frame via compute_object_poses)"
                if o["movable"] else "STATIC solid — its TOP is a support surface, its FACES "
                "are collision walls, and it is SOLID to the floor (no floor inside its footprint)")
        lines.append(
            f"  - '{o['name']}' [{kind}]: top z={o['top']:.2f}, bottom z={o['bottom']:.2f}, "
            f"spans x[{o['x_range'][0]:.2f}, {o['x_range'][1]:.2f}] "
            f"y[{o['y_range'][0]:.2f}, {o['y_range'][1]:.2f}]; "
            f"center ({o['center'][0]:.2f}, {o['center'][1]:.2f}, {o['center'][2]:.2f}); "
            f"near edge is {gap:+.2f} m from the base in x.")
        # G6 — DERIVED obstacle-face clearance (replaces the prescriptive yaml text):
        # the vertical FRONT face is a collision wall; be above the top before crossing it.
        if not o["movable"] and o["top"] > 0.05:
            lines.append(
                f"      front FACE at x={near:.2f} spans z[{o['bottom']:.2f}, {o['top']:.2f}]: any "
                f"body part crossing x={near:.2f} below z={o['top']:.2f} COLLIDES — clear z={o['top']:.2f} "
                f"(+~5 cm) BEFORE any body crosses x={near:.2f} (rise first, then translate), or route "
                f"around it in y (outside [{o['y_range'][0]:.2f}, {o['y_range'][1]:.2f}]). ANY foothold placed "
                f"ON this top must be >= ~{min(0.10, 0.25*(o['x_range'][1]-near)):.2f} m INSIDE the edge (not "
                f"on the arris); a contact that does NOT belong on this object stays on the APPROACH side "
                f"(x < {near:.2f}) at its OWN surface height — it does not climb onto this top.")

    statics = [o for o in objs if not o["movable"]]
    if len(statics) >= 1:
        prof = _step_up_profile(statics)
        lines.append("  STEP-UP / SURFACE PROFILE (successive footfall targets, low -> high; approach along +x):")
        lines.append(f"    floor: foot z=0.00 for x < {prof[0][1]:.2f}")
        prev_top = 0.0
        for name, lo, hi, top in prof:
            yb = next((f" y in [{o['y_range'][0]:.2f}, {o['y_range'][1]:.2f}]"       # G7: y-band
                       for o in statics if o["name"] == name), "")
            rise = top - prev_top                                                     # G4: rise per step
            lines.append(
                f"    {name}: plant a foot at x in [{lo:.2f}, {hi:.2f}]{yb} with foot z={top:.2f}"
                f"  (rise {rise:+.2f} m from the previous surface; "
                f"to STAND on it the pelvis rises to ~{top + rz:.2f} m)")     # G3: per-tread stand
            prev_top = top
        top_name, tlo, thi, ttop = max(prof, key=lambda p: p[3])
        lines.append(
            f"    => highest surface is '{top_name}' at z={ttop:.2f} (x in [{tlo:.2f}, {thi:.2f}]). "
            f"These are the FACTS of the terrain; what to DO with them is the skill's GOAL (from the "
            f"environment description). IF the goal is to STAND ON a surface, a footfall on it is at "
            f"z=surface+offset and the pelvis rises to ~surface+{rz:.2f} m; a rise larger than one "
            f"leg-length cannot be STEPPED foot-to-foot, so MANTLE it (hands on the surface, then bring "
            f"a foot/knee up). IF the goal is to SIT/rest a body on a surface, the feet do NOT climb onto "
            f"it — they stay on the LOWER surface in front. Never treat a raised object as a staircase "
            f"unless the goal is to ascend it.")

    movs = [o for o in objs if o["movable"]]
    for o in movs:
        lines.append(
            f"  '{o['name']}' rests at center ({o['center'][0]:.2f}, {o['center'][1]:.2f}, "
            f"{o['center'][2]:.2f}); its top is z={o['top']:.2f}, bottom z={o['bottom']:.2f}. "
            f"Reach the hands DOWN to it at its rest pose (bend the base), grip, then lift.")
    return "\n".join(lines)
