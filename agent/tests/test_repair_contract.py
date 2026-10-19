import subprocess
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from agents.code_gen.code_executer_agent import validate_generated_code, make_code_executer_node
from agents.code_gen.repair_context import repair_prompt, response_text
from agents.code_gen.critic_coder_agent import make_critic_coder_node
from agents.code_gen.motion_diagnosis_agent import make_motion_diagnosis_agent_node
from prompts.prompt_loader import load_prompt, load_yaml_config

ROOT = Path(__file__).resolve().parents[2]
VALID = '''from motion.base import BaseMotionGenerator
class TestMotionGenerator(BaseMotionGenerator):
    def compute_foot_position_body_frame(self, leg_name, phase):
        return self.base_init_feet_pos[leg_name].copy()
'''


class RepairContractTests(unittest.TestCase):
    def test_valid_code(self):
        self.assertIsNone(validate_generated_code(VALID))

    def test_bad_interfaces(self):
        cases = ["", "class X(", VALID.replace("from motion.base", "from invented"),
                 VALID.replace("(BaseMotionGenerator)", "(object)"),
                 VALID.replace("compute_foot_position_body_frame", "guess"),
                 VALID + "\nTestMotionGenerator.__abstractmethods__ = frozenset()",
                 VALID + "\n    def step(self, dt): pass",
                 VALID + "\n    def reset(self, *args): pass"]
        for code in cases:
            with self.subTest(code=code):
                self.assertIsNotNone(validate_generated_code(code))

    def test_text_only(self):
        message = AIMessage(content=[{"type": "thinking", "thinking": "PRIVATE", "signature": "signature"},
                                     {"type": "text", "text": "CODE"}])
        self.assertEqual(response_text(message), "CODE")

    def test_prompt_current_attempt_only(self):
        state = dict(skill_name="TEST", skill_desc={"description": "spin"},
                     current_skill_code="literal = '{SKILL_DESCRIPTION_JSON}'",
                     execution_feedback="CURRENT CRASH", evaluation_feedback="", motion_plan="PLAN",
                     messages=[AIMessage("STALE CODE")])
        prompt = repair_prompt("{CURRENT_SKILL_CODE}\n{SKILL_DESCRIPTION_JSON}\n{UNIT_TEST_FEEDBACK_JSON}", state)
        self.assertIn("literal = '{SKILL_DESCRIPTION_JSON}'", prompt)
        self.assertIn('"description": "spin"', prompt)
        self.assertIn("CURRENT CRASH", prompt)
        self.assertIn("class BaseMotionGenerator", prompt)
        self.assertNotIn("STALE CODE", prompt)
        self.assertNotIn("{UNIT_TEST_FEEDBACK_JSON}", prompt)

    def test_rejected_code_preserved_and_stale_feedback_cleared(self):
        state = dict(messages=[AIMessage("bad code")], skill_name="TEST", skill_version="v",
                     num_critic_iters=0, rewrite_mode="", skill_desc={},
                     evaluation_feedback="OLD PASS", diagnosis="OLD DIAGNOSIS")
        with patch("agents.code_gen.code_executer_agent.run_skill") as sim:
            result = make_code_executer_node({})(state)
        sim.assert_not_called()
        self.assertEqual(result["current_skill_code"], "bad code")
        self.assertIn("SyntaxError", result["execution_feedback"])
        self.assertEqual(result["evaluation_feedback"], "")
        self.assertEqual(result["diagnosis"], "")
        self.assertEqual(result["num_critic_iters"], 1)

    def test_actual_repair_nodes_fill_templates(self):
        class FakeAgent:
            def generate(self, prompt):
                self.prompt = prompt
                return AIMessage("DIAGNOSIS", response_metadata={"usage": {"input_tokens": 1, "output_tokens": 1}})
        cfg = load_yaml_config(str(ROOT / "agent/cfg/config.yaml"))
        state = dict(messages=[AIMessage("STALE_HISTORY")], current_skill_code=VALID,
                     skill_name="TEST", skill_desc={"description": "spin"}, skill_version="v",
                     success=False, num_critic_iters=1, rewrite_mode="", token_usage={},
                     execution_feedback="CURRENT_CRASH", motion_plan="CURRENT_PLAN")
        for factory, template in ((make_motion_diagnosis_agent_node, "motion_failure_diagnosis"),
                                  (make_critic_coder_node, "critic_coder")):
            fake = FakeAgent()
            result = factory(fake, load_prompt(template, cfg), cfg)(state)
            self.assertIn(VALID, fake.prompt)
            self.assertIn("CURRENT_CRASH", fake.prompt)
            self.assertIn("CURRENT_PLAN", fake.prompt)
            self.assertNotIn("STALE_HISTORY", fake.prompt)
            for placeholder in ("{CURRENT_SKILL_CODE}", "{SKILL_DESCRIPTION_JSON}",
                                "{EVALUATION_FEEDBACK_JSON}", "{UNIT_TEST_FEEDBACK_JSON}",
                                "{MOTION_FAILURE_DIAGNOSIS_FEEDBACK}"):
                self.assertNotIn(placeholder, fake.prompt)
            if "diagnosis" in result:
                state["diagnosis"] = result["diagnosis"]
        self.assertIn("DIAGNOSIS", fake.prompt)

    def test_runtime_vectors_and_loader(self):
        # Simulator and agent both have top-level 'utils' packages: isolate imports.
        code = '''
import numpy as np
from motion.base import BaseMotionGenerator
from motion import _get_motion_generator_from_module
import types
class TestMotionGenerator(BaseMotionGenerator):
    def compute_foot_position_body_frame(self, leg_name, phase):
        return self.value
g = TestMotionGenerator({'FL_calf': np.zeros(3)})
for bad in (None, {}, 0, [0, 1], [0, 1, float('nan')]):
    g.value = bad
    try: g.step(.005)
    except ValueError as exc: assert 'finite (3,)' in str(exc)
    else: raise AssertionError(bad)
g.value = [0, 0, -.35]
assert g.step(.005)['foot_positions_body']['FL_calf'].shape == (3,)
m = types.ModuleType('fixture')
m.__file__ = 'fixture.py'
m.TestMotionGenerator = TestMotionGenerator
TestMotionGenerator.__module__ = 'fixture'
assert _get_motion_generator_from_module(m) is TestMotionGenerator
TestMotionGenerator.step = lambda *a: None
try: _get_motion_generator_from_module(m)
except RuntimeError as exc: assert 'inherit step' in str(exc)
else: raise AssertionError('override accepted')
'''
        subprocess.run([sys.executable, "-c", code], cwd=ROOT / "simulator", check=True)


if __name__ == "__main__":
    unittest.main()
