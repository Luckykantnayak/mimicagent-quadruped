# Physical summary — humanoid
(generated from humanoid.xml; do not edit by hand)

## Kinematics

- Base body `pelvis` nominal standing height: 0.88 m
- Total height (tallest point at rest): 1.36 m

- End-effector reach (how far each EE extends from its limb root; keep commanded targets within this):
    - right_foot: reach ≈ 0.83 m; nominal position (rel. base) ≈ [+0.00, -0.08, -0.83] m
    - left_foot: reach ≈ 0.83 m; nominal position (rel. base) ≈ [+0.00, +0.08, -0.83] m
    - right_hand: reach ≈ 0.53 m; nominal position (rel. base) ≈ [+0.05, -0.25, -0.04] m
    - left_hand: reach ≈ 0.53 m; nominal position (rel. base) ≈ [+0.05, +0.25, -0.04] m

- Joint position limits (radians; degrees in parentheses):
    - abdomen_x: [-1.047, +1.047] rad ([-60°, +60°])
    - abdomen_y: [-1.047, +1.571] rad ([-60°, +90°])
    - abdomen_z: [-1.222, +1.222] rad ([-70°, +70°])
    - neck_x: [-0.873, +0.873] rad ([-50°, +50°])
    - neck_y: [-0.698, +1.047] rad ([-40°, +60°])
    - neck_z: [-0.785, +0.785] rad ([-45°, +45°])
    - right_shoulder_x: [-1.920, +2.443] rad ([-110°, +140°])
    - right_shoulder_y: [-1.571, +1.571] rad ([-90°, +90°])
    - right_shoulder_z: [-1.047, +2.792] rad ([-60°, +160°])
    - right_elbow: [+0.000, +2.792] rad ([+0°, +160°])
    - left_shoulder_x: [-2.443, +1.920] rad ([-140°, +110°])
    - left_shoulder_y: [-1.571, +1.571] rad ([-90°, +90°])
    - left_shoulder_z: [-2.792, +1.047] rad ([-160°, +60°])
    - left_elbow: [-2.792, +0.000] rad ([-160°, +0°])
    - right_hip_x: [-1.047, +0.524] rad ([-60°, +30°])
    - right_hip_y: [-2.443, +1.047] rad ([-140°, +60°])
    - right_hip_z: [-1.047, +1.047] rad ([-60°, +60°])
    - right_knee: [+0.000, +2.792] rad ([+0°, +160°])
    - right_ankle_x: [-0.524, +0.524] rad ([-30°, +30°])
    - right_ankle_y: [-0.960, +0.960] rad ([-55°, +55°])
    - right_ankle_z: [-0.698, +0.698] rad ([-40°, +40°])
    - left_hip_x: [-0.524, +1.047] rad ([-30°, +60°])
    - left_hip_y: [-2.443, +1.047] rad ([-140°, +60°])
    - left_hip_z: [-1.047, +1.047] rad ([-60°, +60°])
    - left_knee: [+0.000, +2.792] rad ([+0°, +160°])
    - left_ankle_x: [-0.524, +0.524] rad ([-30°, +30°])
    - left_ankle_y: [-0.960, +0.960] rad ([-55°, +55°])
    - left_ankle_z: [-0.698, +0.698] rad ([-40°, +40°])

- Limb segment lengths:
    - torso: 23.6 cm
    - head: 22.4 cm
    - right_upper_arm: 30.6 cm
    - right_lower_arm: 27.5 cm
    - right_hand: 25.9 cm
    - left_upper_arm: 30.6 cm
    - left_lower_arm: 27.5 cm
    - left_hand: 25.9 cm
    - right_thigh: 8.5 cm
    - right_shin: 42.2 cm
    - right_foot: 41.0 cm
    - left_thigh: 8.5 cm
    - left_shin: 42.2 cm
    - left_foot: 41.0 cm

## Dynamics (feasible velocities & accelerations)

Estimates grounded in standard mechanics (the sim itself applies no forces — these bound what is *physically plausible*, so keep commanded velocities within them). Sources cited at the end.

- Total mass: 50.0 kg
- Per actuated joint — torque limit (source: actuatorfrcrange), effective inertia at rest, max acceleration |q̈| ≲ τ/I, and peak velocity |q̇| ≲ √(A·τ/I) for a full-range oscillation:
    - abdomen_x: τ ≤ 200 N·m, I ≈ 1.507 kg·m² → |q̈| ≲ 133 rad/s², |q̇| ≲ 11.8 rad/s
    - abdomen_y: τ ≤ 200 N·m, I ≈ 1.182 kg·m² → |q̈| ≲ 169 rad/s², |q̇| ≲ 14.9 rad/s
    - abdomen_z: τ ≤ 200 N·m, I ≈ 0.422 kg·m² → |q̈| ≲ 474 rad/s², |q̇| ≲ 24.1 rad/s
    - neck_x: τ ≤ 50 N·m, I ≈ 0.143 kg·m² → |q̈| ≲ 350 rad/s², |q̇| ≲ 17.5 rad/s
    - neck_y: τ ≤ 50 N·m, I ≈ 0.143 kg·m² → |q̈| ≲ 350 rad/s², |q̇| ≲ 17.5 rad/s
    - neck_z: τ ≤ 50 N·m, I ≈ 0.024 kg·m² → |q̈| ≲ 2082 rad/s², |q̇| ≲ 40.4 rad/s
    - right_shoulder_x: τ ≤ 100 N·m, I ≈ 0.357 kg·m² → |q̈| ≲ 277 rad/s², |q̇| ≲ 24.6 rad/s
    - right_shoulder_y: τ ≤ 100 N·m, I ≈ 0.027 kg·m² → |q̈| ≲ 3682 rad/s², |q̇| ≲ 76.1 rad/s
    - right_shoulder_z: τ ≤ 100 N·m, I ≈ 0.361 kg·m² → |q̈| ≲ 275 rad/s², |q̇| ≲ 23.0 rad/s
    - right_elbow: τ ≤ 70 N·m, I ≈ 0.062 kg·m² → |q̈| ≲ 1122 rad/s², |q̇| ≲ 39.6 rad/s
    - left_shoulder_x: τ ≤ 100 N·m, I ≈ 0.357 kg·m² → |q̈| ≲ 277 rad/s², |q̇| ≲ 24.6 rad/s
    - left_shoulder_y: τ ≤ 100 N·m, I ≈ 0.027 kg·m² → |q̈| ≲ 3682 rad/s², |q̇| ≲ 76.1 rad/s
    - left_shoulder_z: τ ≤ 100 N·m, I ≈ 0.361 kg·m² → |q̈| ≲ 275 rad/s², |q̇| ≲ 23.0 rad/s
    - left_elbow: τ ≤ 70 N·m, I ≈ 0.062 kg·m² → |q̈| ≲ 1122 rad/s², |q̇| ≲ 39.6 rad/s
    - right_hip_x: τ ≤ 200 N·m, I ≈ 2.223 kg·m² → |q̈| ≲ 90 rad/s², |q̇| ≲ 8.4 rad/s
    - right_hip_y: τ ≤ 200 N·m, I ≈ 2.227 kg·m² → |q̈| ≲ 90 rad/s², |q̇| ≲ 12.5 rad/s
    - right_hip_z: τ ≤ 200 N·m, I ≈ 0.036 kg·m² → |q̈| ≲ 5558 rad/s², |q̇| ≲ 76.3 rad/s
    - right_knee: τ ≤ 150 N·m, I ≈ 0.370 kg·m² → |q̈| ≲ 405 rad/s², |q̇| ≲ 23.8 rad/s
    - right_ankle_x: τ ≤ 90 N·m, I ≈ 0.011 kg·m² → |q̈| ≲ 7872 rad/s², |q̇| ≲ 64.2 rad/s
    - right_ankle_y: τ ≤ 90 N·m, I ≈ 0.015 kg·m² → |q̈| ≲ 5818 rad/s², |q̇| ≲ 74.7 rad/s
    - right_ankle_z: τ ≤ 90 N·m, I ≈ 0.015 kg·m² → |q̈| ≲ 5879 rad/s², |q̇| ≲ 64.1 rad/s
    - left_hip_x: τ ≤ 200 N·m, I ≈ 2.223 kg·m² → |q̈| ≲ 90 rad/s², |q̇| ≲ 8.4 rad/s
    - left_hip_y: τ ≤ 200 N·m, I ≈ 2.227 kg·m² → |q̈| ≲ 90 rad/s², |q̇| ≲ 12.5 rad/s
    - left_hip_z: τ ≤ 200 N·m, I ≈ 0.036 kg·m² → |q̈| ≲ 5558 rad/s², |q̇| ≲ 76.3 rad/s
    - left_knee: τ ≤ 150 N·m, I ≈ 0.370 kg·m² → |q̈| ≲ 405 rad/s², |q̇| ≲ 23.8 rad/s
    - left_ankle_x: τ ≤ 90 N·m, I ≈ 0.011 kg·m² → |q̈| ≲ 7872 rad/s², |q̇| ≲ 64.2 rad/s
    - left_ankle_y: τ ≤ 90 N·m, I ≈ 0.015 kg·m² → |q̈| ≲ 5818 rad/s², |q̇| ≲ 74.7 rad/s
    - left_ankle_z: τ ≤ 90 N·m, I ≈ 0.015 kg·m² → |q̈| ≲ 5879 rad/s², |q̇| ≲ 64.1 rad/s

- Torso forward speed (Froude number Fr = v²/gL, L = leg length 0.88 m):
    - walk→run transition (Fr≈0.5): ≈ 2.08 m/s
    - walking ceiling (Fr=1, foot leaves ground above this): ≈ 2.94 m/s
    - dynamic running gait (Fr≈4): up to ≈ 5.88 m/s

- Torso vertical jump & flight (leg push F≈τ/ℓ, then projectile h=v_z²/2g, t_air=2v_z/g — approximate):
    - net upward accel ≈ 9 m/s², take-off speed ≈ 2.20 m/s
    - apex height ≈ 0.25 m, flight time ≈ 0.45 s
    - one full aerial rotation needs ω ≈ 2π/t_air ≈ 14.0 rad/s (802°/s)

- Torso angular rates about roll(x)/pitch(y)/yaw(z) — whole-body rotational inertia (exact), grounded α=τ/I with a planted foot, and the aerial regime where I·ω is conserved:
    - roll: I ≈ 8.94 kg·m²; grounded |α| ≲ 9 rad/s², |ω| ≲ 3.8 rad/s over a 90° swing
    - pitch: I ≈ 8.50 kg·m²; grounded rotation negligible (no support lever) → rotate in flight
    - yaw: I ≈ 0.59 kg·m²; grounded |α| ≲ 137 rad/s², |ω| ≲ 14.6 rad/s over a 90° swing
    - aerial: a flip/twist needs ω ≥ 2π/t_air (see flight time above); the legs impart angular momentum L = I·ω at take-off. A ~9 kg quadruped (MIT Mini Cheetah) lands 360° backflips, so this regime is real.

- Sources: Froude number & v=√(gL) — R. McN. Alexander / en.wikipedia.org/wiki/Froude_number; aerial rotation & backflip feasibility — MIT Mini Cheetah, Katz et al., ICRA 2019; Unitree Go2 top speed 2.5–5 m/s — unitree.com/go2. Torque source is the MJCF `actuatorfrcrange`; jump/rate force terms are statics estimates.
