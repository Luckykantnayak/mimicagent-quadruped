"""
simulator/core/grounding.py

Deterministic kinematic contact-grounding — the framework computes the parts of
the motion that geometry determines (seating the base on the ground, holding a
planted contact still), instead of asking the LLM to get them right by hand. No
gravity, no dynamics, no control: pure geometry.

CONTACT SCHEDULE. The set of load-bearing contacts is PER-PHASE: the controller
passes `active` (the names in contact this frame) from the skill's
contacts(phase); when the skill declares none, the static support set is used.
This one mechanism covers a fixed support set (walk), a changing one (a cartwheel:
feet -> hands -> feet), and a flight window (empty -> ballistic).

SURFACE-AWARE. Every height here is measured against the surface actually beneath
that contact (`utils.surfaces.SceneSurfaces`), not against z=0. A foot on a stair
tread, a pelvis on a seat and a hand on a ledge are all ordinary cases, so a
raised-surface skill gets exactly the same support a flat-floor one does. (This
module used to hardcode the floor plane, which is why every non-flat environment
had to run with grounding "off" and hand-author what the framework should have
provided.)

MODES — registration and pinning are INDEPENDENT, because they answer different
questions and a skill may want one without the other:
    "contact"  register the base + pin planted contacts + clamp out of surfaces
    "pin"      pin + clamp only; the skill owns its base height absolutely
               (the right mode for a skill whose base path IS the point: a squat
               down to a floor-level box, a sit onto a seat)
    "off"      nothing (manual playback / debugging)

PINNING IS A BOUNDED, RATE-LIMITED, RECORDED CORRECTION — like every other net in
this framework (floor assist caps at 5 cm, the reach seat at 15 cm). It removes
micro-slip and IK jitter from a contact that is genuinely meant to stay put, and
it SATURATES on a contact that is really moving, so the skill's own error stays
visible to `contact_no_slide` instead of being hidden. The correction is eased in
and out, never stepped.

    (The previous implementation hard-pinned XY and released the pin when the
    error passed a 0.30 m `break_dist`. Because the reference walk's step_length
    is exactly 0.30 m, the pin held the stance foot for a full stride and then
    SNAPPED it forward — manufacturing a joint teleport that the repair loop then
    tried to fix in the skill, where the cause was not. The release counter was
    never read by anything, so the snap was invisible.)
"""
import numpy as np


class ContactGrounding:
    def __init__(self, spec, mode="off", support_ees=None, surfaces=None, dt=1.0 / 200.0,
                 lock_enter=0.03, lock_exit=0.06, pin_cap=0.05, pin_rate=0.5,
                 max_step=0.02, flight_thresh=0.12, clamp_targets=True):
        self.mode = mode
        self.surfaces = surfaces
        # A/B switch for the grounding-level surface clamp (the controller already
        # runs an inside()-gated one on the IK targets; this one is potentially
        # redundant AND discontinuous at a step edge).
        self.clamp_targets = clamp_targets
        self.dt = float(dt)
        # Union offsets: EEs AND declared contact bodies, so a skill that bears weight
        # on a knee/pelvis/forearm grounds the base off that contact too (not just feet).
        self.all_offsets = {spec.ee_names[i]: float(spec.ee_ground_offsets[i])
                            for i in range(len(spec.ee_names))}
        self.all_offsets.update({spec.contact_body_names[i]: float(spec.contact_body_offsets[i])
                                 for i in range(len(spec.contact_body_names))})
        # WHEELS may ROLL/SKATE while planted — a wheel bears weight (z-registration +
        # surface clamp still apply) but must NOT be xy-pinned. Keyed on the spec's
        # `wheel:` flag (more general than kind: a wheeled anything gets it). The
        # no-slide CHECK already exempts wheels, so runtime and check agree.
        self.wheel_names = {spec.ee_names[i] for i in spec.wheel_ee_indices}
        default = [spec.ee_names[i] for i in spec.contact_ee_indices]
        self.default_support = [n for n in (support_ees or default)
                                if n in self.all_offsets] or default

        self.lock_enter = lock_enter
        self.lock_exit = lock_exit
        self.pin_cap = float(pin_cap)
        self.pin_rate = float(pin_rate)
        self.max_step = max_step
        self.flight_thresh = flight_thresh

        self.locked = {}        # name -> [x, y] pin anchor (currently-planted contacts)
        self._pin_delta = {}    # name -> applied xy correction (eased, never stepped)
        self._sm_delta = 0.0
        # Observability: the largest pin correction applied this frame, and whether it
        # saturated the cap. Read by the controller and recorded, so a correction the
        # framework made is never invisible to the repair loop.
        self.pin_mag = 0.0
        self.pin_saturated = False

    # ------------------------------------------------------------------
    @property
    def registers(self):
        return self.mode == "contact"

    @property
    def pins(self):
        return self.mode in ("contact", "pin")

    def _surface_z(self, name, p):
        """Height of the surface beneath contact `name` at its current position.
        Falls back to the floor plane when there is no scene query available."""
        if self.surfaces is None:
            return 0.0
        off = self.all_offsets.get(name, 0.0)
        sole_z = float(p[2]) - off
        # Probe from a little above the sole so a contact already resting registers,
        # and accept a surface anywhere below (the floor is the backstop).
        return self.surfaces.support_z(float(p[0]), float(p[1]), sole_z + 0.05, default=0.0)

    # ------------------------------------------------------------------
    def apply(self, root_pos, root_quat, ee_world, active=None):
        """Return (root_pos, ee_world) with the base registered to the active
        contacts and those contacts held. `active` is the per-phase list of contact
        names (None -> static support set, [] -> flight)."""
        self.pin_mag = 0.0
        self.pin_saturated = False
        if self.mode == "off":
            return root_pos, ee_world

        support = self.default_support if active is None else list(active)
        # a contact must have an actual target (joint-driven limbs are None)
        support = [n for n in support
                   if ee_world.get(n) is not None and n in self.all_offsets]

        # --- FLIGHT: no contacts -> ballistic, release everything -------------
        if not support:
            self.locked.clear()
            self._sm_delta = 0.0
            ee_world = self._ease_out_released(ee_world, set())
            return root_pos, ee_world

        # Surface under each active contact, and how far its contact point sits
        # above that surface. This is the ONLY place the "where is the ground"
        # question is answered, and it is answered per contact.
        surf = {n: self._surface_z(n, ee_world[n]) for n in support}
        height = {n: float(ee_world[n][2]) - self.all_offsets[n] - surf[n] for n in support}

        # Bail to flight (seat NOTHING) only when the skill did NOT declare a
        # per-phase contact set (active is None -> static-support heuristic: if the
        # default feet are all well above their surfaces the robot is probably
        # airborne). When the skill EXPLICITLY declares contacts this phase they are
        # load-bearing BY DECLARATION — SEAT them even if authored high, never treat a
        # declared, hovering contact as flight. This is what stops a crawl whose
        # hand/knee targets were authored a few cm high from being read as a hover.
        if active is None and min(height.values()) > self.flight_thresh:
            self.locked.clear()
            self._sm_delta = 0.0
            return root_pos, self._ease_out_released(ee_world, set())

        # --- REGISTRATION: seat the lowest active contact ON its own surface ---
        if self.registers:
            low = min(height.values())
            target = -low
            # Snap UP immediately whenever the lowest contact is at or below its
            # surface, so a just-planted foot is never left penetrating; ease DOWN
            # gently when the lowest contact RISES (a higher contact takes over during
            # a hand-off) so the base does not pop.
            if target >= self._sm_delta:
                self._sm_delta = target
            else:
                self._sm_delta = max(target, self._sm_delta - self.max_step)
            delta = self._sm_delta
            if abs(delta) > 1e-9:
                root_pos = root_pos.copy()
                root_pos[2] += delta
                shift = np.array([0.0, 0.0, delta])
                ee_world = {n: (p + shift if p is not None else None)
                            for n, p in ee_world.items()}
                for n in support:
                    height[n] += delta

        # --- PIN: hold a planted contact's world XY, bounded and eased ---------
        if self.pins:
            ee_world = self._apply_pins(ee_world, support, height)

        # --- SURFACE CLAMP: no target may sit below the surface beneath it -----
        # Grounded frames only (flight returned earlier), so this never touches an
        # aerial phase. It enforces "no contact point below its surface" kinematically:
        # a swing foot whose down-arc overshoots is flattened onto the tread instead of
        # driven through it, so the coder need not hand-tune the arc bottom.
        if self.clamp_targets:
            for n, p in ee_world.items():
                if p is None:
                    continue
                off = self.all_offsets.get(n)
                if off is None:
                    continue
                floor = self._surface_z(n, p) + off
                if p[2] < floor:
                    p = p.copy()
                    p[2] = floor
                    ee_world[n] = p
        return root_pos, ee_world

    # ------------------------------------------------------------------
    def _apply_pins(self, ee_world, support, height):
        """Bounded, rate-limited XY hold for firmly-planted contacts.

        The correction is `pin_anchor - authored_xy`, clipped to `pin_cap` and eased
        toward that value at `pin_rate`. A contact that stops being planted eases the
        correction back to zero rather than dropping it, so releasing a pin can never
        step the target (the old hard release is what produced the teleport)."""
        planted = set()
        for n in support:
            if n in self.wheel_names:
                continue                    # a wheel rolls: z-seated, never xy-pinned
            h = height[n]
            if n in self.locked:
                if h > self.lock_exit:
                    self.locked.pop(n)      # lifted off -> release (and ease out below)
                else:
                    planted.add(n)
            elif h <= self.lock_enter:
                # Anchor on the AUTHORED position at the moment of planting.
                self.locked[n] = np.asarray(ee_world[n][:2], dtype=float).copy()
                planted.add(n)

        max_step = self.pin_rate * self.dt
        for n in planted:
            authored = np.asarray(ee_world[n][:2], dtype=float)
            desired = self.locked[n] - authored
            mag = float(np.linalg.norm(desired))
            if mag > self.pin_cap:
                desired = desired * (self.pin_cap / mag)
                self.pin_saturated = True
            applied = self._step_toward(self._pin_delta.get(n), desired, max_step)
            self._pin_delta[n] = applied
            if np.linalg.norm(applied) > 1e-9:
                p = ee_world[n].copy()
                p[0] += applied[0]
                p[1] += applied[1]
                ee_world[n] = p
            self.pin_mag = max(self.pin_mag, float(np.linalg.norm(applied)))
        return self._ease_out_released(ee_world, planted)

    def _ease_out_released(self, ee_world, planted):
        """Decay the correction on any contact that is no longer planted, so a
        release is a ramp rather than a step."""
        max_step = self.pin_rate * self.dt
        for n in list(self._pin_delta):
            if n in planted:
                continue
            applied = self._step_toward(self._pin_delta[n], np.zeros(2), max_step)
            if np.linalg.norm(applied) < 1e-9:
                self._pin_delta.pop(n)
                continue
            self._pin_delta[n] = applied
            p = ee_world.get(n)
            if p is not None:
                p = p.copy()
                p[0] += applied[0]
                p[1] += applied[1]
                ee_world[n] = p
            self.pin_mag = max(self.pin_mag, float(np.linalg.norm(applied)))
        return ee_world

    @staticmethod
    def _step_toward(current, desired, max_step):
        if current is None:
            current = np.zeros(2)
        d = desired - current
        n = float(np.linalg.norm(d))
        if n <= max_step or n < 1e-12:
            return np.asarray(desired, dtype=float).copy()
        return current + d * (max_step / n)
