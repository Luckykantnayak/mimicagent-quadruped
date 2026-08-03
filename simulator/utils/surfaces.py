"""
simulator/utils/surfaces.py

THE scene-surface query. One implementation, used by everything that needs to
know "what is the robot standing on here?" — contact grounding, the IK target
clamp, the reach seat, the floor assist, and the recorded surface clearance.

WHY IT EXISTS: that question used to be answered by four near-identical inline
loops in controller.py plus a fifth in grounding.py, each re-deriving a box top
from `model.geom_pos` with an axis-aligned `abs(px - cx) < hx` test. That carried
three unstated assumptions — every scene surface is a BOX, AXIS-ALIGNED, and
declared directly in `<worldbody>` (so its local pos equals its world pos). All
of them hold for the bundled scenes and none of them are enforced, so a ramp
(rotated), a ball (sphere), or a table authored as one `<body pos=...>` with
several geoms would be silently INVISIBLE to grounding and collision alike.

This module answers it exactly instead, by ray-casting against each static scene
geom with `mju_rayGeom`. That is correct for every primitive MuJoCo supports, at
any orientation, in world coordinates — and it hands back the surface NORMAL,
which is what lets a sole be laid flat on a slope rather than merely at the right
height.

Excluded from "scene": robot geoms (obviously) and MOCAP bodies. A mocap box is a
MANIPULAND — the hands are meant to grip it — not a surface to stand on.
"""

import numpy as np
import mujoco

# Ray length used for the downward probe. Any value taller than the tallest scene
# feature works; this is a probe, not a physical quantity.
_PROBE_H = 50.0

_DOWN = np.array([0.0, 0.0, -1.0])


class SceneSurfaces:
    """Static scene geometry, queried exactly.

    Static geoms never move, so their world pose is cached once at construction
    (after an mj_kinematics/mj_forward, which the controller has already done).
    """

    def __init__(self, model, data, robot_geom_ids=()):
        self.model = model
        robot = set(int(g) for g in robot_geom_ids)
        body_mocap = model.body_mocapid
        self.geom_ids = [
            g for g in range(model.ngeom)
            if g not in robot and body_mocap[model.geom_bodyid[g]] < 0
        ]
        # Cache world pose + size + type. data.geom_xpos/xmat are WORLD frame, so
        # this is correct for a geom nested in an offset body — unlike model.geom_pos.
        self._pos, self._mat, self._size, self._type, self._id = [], [], [], [], []
        for g in self.geom_ids:
            self._pos.append(np.array(data.geom_xpos[g], dtype=float))
            self._mat.append(np.array(data.geom_xmat[g], dtype=float).reshape(9))
            self._size.append(np.array(model.geom_size[g], dtype=float))
            self._type.append(int(model.geom_type[g]))
            self._id.append(int(g))
        self._n = len(self._id)
        # Cheap rejection box per geom (world-axis-aligned bound of the real shape),
        # so a downward probe only pays for the exact ray on plausible candidates.
        self._aabb = [self._world_aabb(i) for i in range(self._n)]

    # ------------------------------------------------------------------
    def _world_aabb(self, i):
        """Conservative world AABB (center, half-extent) for cached geom i.
        A plane is unbounded in x/y — represented by an infinite half-extent."""
        t, s, p = self._type[i], self._size[i], self._pos[i]
        if t == mujoco.mjtGeom.mjGEOM_PLANE:
            return p, np.array([np.inf, np.inf, 0.0])
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = float(s[0]); return p, np.array([r, r, r])
        if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            r = float(s[0]); h = float(s[1]) + r
            e = np.abs(self._mat[i].reshape(3, 3)) @ np.array([r, r, h])
            return p, e
        if t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            e = np.abs(self._mat[i].reshape(3, 3)) @ np.array(
                [float(s[0]), float(s[0]), float(s[1] if t == mujoco.mjtGeom.mjGEOM_CYLINDER
                                                  else s[2])])
            return p, e
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            e = np.abs(self._mat[i].reshape(3, 3)) @ np.abs(s[:3])
            return p, e
        # Mesh / hfield / anything else: fall back to the reported size bound.
        e = np.abs(self._mat[i].reshape(3, 3)) @ np.maximum(np.abs(s[:3]), 1e-6)
        return p, e

    # ------------------------------------------------------------------
    def support_under(self, x, y, z_from, max_drop=None):
        """The support surface directly beneath (x, y), at or below `z_from`.

        Returns (z, normal, geom_id), or None when nothing is below the point.
        `max_drop` bounds how far down to look — pass it to ignore the distant
        floor when only a nearby surface (the tread the foot is on) should count.

        Exact for every primitive at any orientation: it casts a downward ray and
        takes the nearest hit, rather than assuming an axis-aligned box top.
        """
        # Start slightly ABOVE the query point so a point resting exactly on the
        # surface still registers a hit rather than starting inside it.
        origin = np.array([float(x), float(y), float(z_from) + 1e-4])
        # A point BURIED IN a solid is supported by THAT solid's top, not by
        # whatever lies below it. A downward ray fired from inside a box does not
        # hit the top face it is already past, so the probe used to fall straight
        # through to the floor and report the buried contact as comfortably CLEAR —
        # the exact opposite of the truth, and invisible because `contact_grounded`
        # and `contact_no_slide` both score this number. Measured on the ledge drop:
        # the right foot sits 3.6 cm INSIDE the ledge for 234 frames and was recorded
        # as 0.664 m of clearance, so its slide was never even measured (the
        # "firmly planted" gate needs clearance <= 3.5 cm) while `surface_penetration`
        # correctly flagged the same frames. Resolve the containing solid first.
        depth, gid = self.inside(origin)
        if depth > 1e-9 and gid is not None:
            top = self.surface_above(float(x), float(y), float(z_from))
            # A PLANE has no "surface above" to ray-cast for — it is one-sided, so an
            # upward probe from beneath misses it and the whole query returned None,
            # i.e. "there is no ground here" for a foot 4 mm THROUGH the floor. That
            # is the opposite of the truth and it made the pre-sim surface-existence
            # check hard-fail a landing as if the robot were floating in mid-air. A
            # point below a plane is supported BY that plane, at its own height.
            if top is None:
                _i = self._id.index(gid) if gid in self._id else None
                if _i is not None and self._type[_i] == mujoco.mjtGeom.mjGEOM_PLANE:
                    top = float(self._pos[_i][2])
            if top is not None:
                i = self._id.index(gid) if gid in self._id else None
                n = np.array([0.0, 0.0, 1.0])
                if i is not None:
                    probe = np.zeros(3)
                    dd = mujoco.mju_rayGeom(self._pos[i], self._mat[i], self._size[i],
                                            np.array([float(x), float(y), top + 1e-3]),
                                            _DOWN, self._type[i], probe)
                    if dd >= 0 and np.linalg.norm(probe) > 1e-9:
                        n = probe.copy()
                        if n[2] < 0:
                            n = -n
                if max_drop is not None and (float(z_from) - top) > max_drop + 1e-4:
                    return None
                return float(top), n, gid
        best_d, best_n, best_g = None, None, None
        normal = np.zeros(3)
        for i in range(self._n):
            c, e = self._aabb[i]
            if abs(origin[0] - c[0]) > e[0] or abs(origin[1] - c[1]) > e[1]:
                continue                      # probe misses this geom's footprint
            d = mujoco.mju_rayGeom(self._pos[i], self._mat[i], self._size[i],
                                   origin, _DOWN, self._type[i], normal)
            if d < 0:
                continue
            if best_d is None or d < best_d:
                best_d, best_g = float(d), self._id[i]
                best_n = normal.copy()
        if best_d is None:
            return None
        if max_drop is not None and best_d > max_drop + 1e-4:
            return None
        z = float(origin[2]) - best_d
        n = best_n if best_n is not None and np.linalg.norm(best_n) > 1e-9 \
            else np.array([0.0, 0.0, 1.0])
        # Orient the normal upward — a downward probe can report either face.
        if n[2] < 0:
            n = -n
        return z, n, best_g

    def support_z(self, x, y, z_from, default=0.0, max_drop=None):
        """`support_under`'s height only, with a default when nothing is below."""
        hit = self.support_under(x, y, z_from, max_drop=max_drop)
        return default if hit is None else hit[0]

    def surface_above(self, x, y, z_from):
        """Height of the nearest surface ABOVE (x, y, z_from), or None.

        For a point INSIDE a solid this is that solid's top face — a ray started
        inside a geom returns its exit distance. That is what lets a target aimed
        INTO a step be lifted ONTO the step rather than compared against the floor
        far beneath it. Callers must gate this on `inside()`: a point in free air
        under an overhang (a foot crawling beneath a table) has a surface above it
        too, and must NOT be lifted onto it.
        """
        origin = np.array([float(x), float(y), float(z_from)])
        up = np.array([0.0, 0.0, 1.0])
        normal = np.zeros(3)
        best = None
        for i in range(self._n):
            c, e = self._aabb[i]
            if abs(origin[0] - c[0]) > e[0] or abs(origin[1] - c[1]) > e[1]:
                continue
            d = mujoco.mju_rayGeom(self._pos[i], self._mat[i], self._size[i],
                                   origin, up, self._type[i], normal)
            if d < 0:
                continue
            if best is None or d < best:
                best = float(d)
        return None if best is None else float(origin[2]) + best

    # ------------------------------------------------------------------
    def inside(self, p):
        """(depth, geom_id) for the deepest static geom containing point p, else
        (0.0, None). Depth is the distance to the nearest face — how far the point
        would have to move to get out.

        Kept CENTER-based like the check it feeds: a contact merely RESTING on a
        surface keeps its geom center outside, so this is false-positive-free,
        unlike a surface-overlap test."""
        p = np.asarray(p, dtype=float)
        best_d, best_g = 0.0, None
        for i in range(self._n):
            t = self._type[i]
            if t == mujoco.mjtGeom.mjGEOM_PLANE:
                d = -(p[2] - self._pos[i][2])          # below an (assumed level) plane
            else:
                c, e = self._aabb[i]
                if np.any(np.abs(p - c) > e):
                    continue
                local = self._mat[i].reshape(3, 3).T @ (p - self._pos[i])
                s = self._size[i]
                if t == mujoco.mjtGeom.mjGEOM_BOX:
                    slack = np.abs(s[:3]) - np.abs(local)
                    d = float(slack.min()) if np.all(slack > 0) else -1.0
                elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                    d = float(s[0] - np.linalg.norm(local))
                elif t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                    radial = s[0] - float(np.linalg.norm(local[:2]))
                    axial = (s[1] + (s[0] if t == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0)) \
                        - abs(float(local[2]))
                    d = min(radial, axial) if radial > 0 and axial > 0 else -1.0
                else:
                    continue                            # mesh/hfield: not tested here
            if d > best_d:
                best_d, best_g = float(d), self._id[i]
        return best_d, best_g

    def geom_name(self, gid):
        if gid is None:
            return None
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
