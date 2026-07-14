import json
import os

from llm.load_claude import load_llm

from utils.tools import extract_json_from_message

from langchain_core.messages import get_buffer_string


prompt = """
=== SYSTEM ===
You are a Skill Classification Agent.

You are given a single kinematic robot motion skill. The motion is an
animation-style trajectory generated under purely kinematic assumptions.

Your task is to classify the skill using ONLY the enum values defined below.
The classification is intended for:

- Analyzing locomotion structure
- Phase-function usage
- Skill diversity and curriculum construction

You MUST:

- Use only the enum values provided
- Choose exactly one value where specified
- Choose zero or more values where specified
- Output strictly valid JSON
- Provide no explanations or extra text

You MUST NOT:

- Invent new labels
- Reason about forces, torques, friction, inertia, or stability
- Assume any controller or dynamics model
- Use canonical gait names (walk, trot, pace, bound)

All decisions must be based solely on observable kinematic coordination,
phase evolution, and base–limb interaction patterns.

--------------------------------------------------
1. PROPULSION MECHANISM
--------------------------------------------------
(choose exactly one; describes how motion is primarily generated)

- CYCLIC_LEG_SWEEP          // repeated leg motions produce displacement
- CONTACT_REDIRECTION       // limbs redirect or steer motion without swinging
- BASE_TRAJECTORY_DRIVEN    // base motion dominates, limbs adapt
- IMPULSE_GENERATION        // brief push, vault, or jump
- NO_PROPULSION             // no net displacement

--------------------------------------------------
2. CONTACT ROLE TOPOLOGY
--------------------------------------------------
(choose exactly one; describes limb contact structure)

- FIXED_SUPPORT             // same limbs maintain support throughout
- ALTERNATING_SUPPORT       // limbs take turns supporting base
- CONTINUOUS_REDIRECTING    // contact maintained, roles shift subtly
- TRANSIENT_CONTACT         // brief, discrete contact events
- NO_CONTACT                // airborne

--------------------------------------------------
3. PHASE FUNCTION
--------------------------------------------------
(choose exactly one; describes what phase primarily controls)

- CONTACT_TIMING            // controls lift/placement of limbs
- LIMB_SHAPE_EVOLUTION      // controls posture or limb configuration
- BASE_ORIENTATION_CHANGE   // controls yaw, pitch, roll progression
- BASE_POSITION_PROGRESS    // controls spatial translation
- EVENT_EXECUTION           // controls a discrete maneuver

--------------------------------------------------
4. LOCOMOTION GEOMETRY
--------------------------------------------------
(choose one or more)

- FORWARD
- BACKWARD
- LATERAL
- DIAGONAL
- CURVED
- ORBITAL

--------------------------------------------------
5. INTER-LIMB COORDINATION
--------------------------------------------------
(choose one or more)

- SYNCHRONOUS_ALL_LIMBS
- FRONT_REAR_COUPLED
- LEFT_RIGHT_COUPLED
- DIAGONAL_PAIRING
- SEQUENTIAL_WAVE
- ALTERNATING_SUPPORT_GROUPS

--------------------------------------------------
6. BASE KINEMATIC MODES
--------------------------------------------------
(choose one or more)

- TRANSLATION
- YAW
- PITCH
- ROLL
- HEIGHT

--------------------------------------------------
7. LOCOMOTION MODE
--------------------------------------------------
(choose ATLEAST one or more)

- LIMB_PROPELLED
- BASE_DRIVEN
- AERIAL

LIMB_PROPELLED means the displacement is generated primarily by limb
motion (stepping, pushing, skating-style strokes).

BASE_DRIVEN means the base trajectory dominates while limbs hold pose or
adapt (gliding, driven rolling, base-led maneuvers).

both LIMB_PROPELLED, BASE_DRIVEN means the motion can be accomplished by
either interpretation.

AERIAL - aerial bound skills (any skill requiring aerial motion like backflip)
--------------------------------------------------
OUTPUT FORMAT (STRICT JSON)
--------------------------------------------------

{
  "propulsion_mechanism": "...",
  "contact_role_topology": "...",
  "phase_function": "...",
  "locomotion_geometry": [...],
  "inter_limb_coordination": [...],
  "base_kinematics": [...],
  "locomotion_mode": "...",
}

No extra text.
No explanations.

"""

llm = load_llm()

skills_folder_pth = os.environ.get("SKILLS_DIR", "data/skills/skills_v1")
skills_file = os.path.join(skills_folder_pth, "all_skills.json")

with open(skills_file) as f:
    skills = json.load(f)

for skill in skills:
    output = llm.invoke(prompt + "\n Here is the skill description:\n" + json.dumps(skill, indent=2))
    output = get_buffer_string([output])

    try:
        output = extract_json_from_message(output)

        out_file = os.path.join(skills_folder_pth, "analysis", skill["skill_id"] + "_cluster_classifier.json")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(output, f)

    except Exception as e:
        print("Skipping skill", skill['skill_id'])