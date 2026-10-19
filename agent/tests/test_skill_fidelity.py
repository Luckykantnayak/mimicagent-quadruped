import copy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from utils.skill_fidelity import evaluate_fidelity, validate_criteria, append_fidelity


class FidelityTests(unittest.TestCase):
    def setUp(self):
        self.phase = np.tile(np.arange(200) / 200, 2)
        p = self.phase
        tilt = np.minimum(np.clip((p - .2) / .1, 0, 1), np.clip((.8 - p) / .1, 0, 1))
        self.rot = Rotation.from_euler("ZYX", np.c_[4*np.pi*p, tilt*np.pi/2, p*0])
        self.traj = dict(phase=p, fps=200, root_pos=np.tile([0, 0, .5], (400, 1)),
                         root_quat=self.rot.as_quat()[:, [3, 0, 1, 2]],
                         joints=np.tile(np.sin(2*np.pi*10*p)[:, None], (1, 12)) * .15)
        self.feet = np.zeros((400, 4, 3))
        self.feet[:, :, 2] = .0801
        self.feet[:, 2:, 2] += tilt[:, None] * .5
        self.legs = ["FL", "FR", "RL", "RR"]
        self.groups = {leg: list(range(i*3, i*3+3)) for i, leg in enumerate(self.legs)}
        self.criteria = [
            dict(name="front spin", phase=[.3, .7], metric="heading_rotation_deg", min=270),
            dict(name="support", phase=[.3, .7], metric="support_fraction", legs=["FL", "FR"], min=.95),
            dict(name="raised rear", phase=[.3, .7], metric="clearance_min_m", legs=["RL", "RR"], min=.1),
            dict(name="upright", phase=[.3, .7], metric="nose_up_max_deg", max=-75),
            dict(name="active legs", phase=[.3, .7], metric="leg_motion_min_rad", legs=["FL", "FR"], min=.1),
            dict(name="return level", phase=[.85, 1], metric="nose_up_min_deg", min=-5),
        ]

    def evaluate(self):
        return evaluate_fidelity(self.traj, self.criteria, self.feet, self.legs, .0801, self.groups)

    def test_correct_transition(self):
        self.assertTrue(all(r["passed"] for r in self.evaluate()))

    def test_static_four_leg_spin_fails_posture_and_clearance(self):
        self.traj["root_quat"] = Rotation.from_euler("z", self.phase*4*np.pi).as_quat()[:, [3, 0, 1, 2]]
        self.feet[:, :, 2] = .0801
        failures = {r["name"] for r in self.evaluate() if not r["passed"]}
        self.assertIn("intent:raised rear", failures)
        self.assertIn("intent:upright", failures)

    def test_rotation_in_wrong_phase_fails(self):
        tilt = self.rot.as_matrix()[:, 2, 0]
        yaw = np.where(self.phase < .3, self.phase*20, 6)
        self.traj["root_quat"] = Rotation.from_euler("ZYX", np.c_[yaw, np.arcsin(np.clip(-tilt, -1, 1)), yaw*0]).as_quat()[:, [3, 0, 1, 2]]
        self.assertFalse(self.evaluate()[0]["passed"])

    def test_hovering_fails(self):
        self.feet[:, :2, 2] += .3
        self.assertFalse(self.evaluate()[1]["passed"])

    def test_folded_static_legs_fail(self):
        self.traj["joints"][:] = 0
        self.assertFalse(self.evaluate()[4]["passed"])

    def test_partial_boundary_windows_ignored(self):
        for key in ("phase", "root_pos", "root_quat", "joints"):
            self.traj[key] = self.traj[key][80:]
        self.feet = self.feet[80:]
        self.assertTrue(all(r["passed"] for r in self.evaluate()))

    def test_missing_phase_rejected(self):
        del self.traj["phase"]
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_invalid_contracts(self):
        for bad in ([], [dict(self.criteria[0], metric="guessed")],
                    [dict(self.criteria[0], phase=[.7, .3])],
                    [dict(self.criteria[1], legs=[])],
                    [dict(self.criteria[0], min=float("nan"))]):
            with self.subTest(criteria=bad), self.assertRaises(ValueError):
                validate_criteria(bad)

    def test_legacy_report_unchanged(self):
        original = {"summary": {"passed_all": True, "total_reward": 0}, "tests": []}
        result = append_fidelity(copy.deepcopy(original), {}, None, "/not/needed")
        self.assertFalse(result["summary"].pop("fidelity_checked"))
        self.assertEqual(original, result)

    def test_speed_not_just_cycle_displacement(self):
        self.traj["root_pos"][:, 0] = np.arange(400) / 200 * .2
        self.criteria = [dict(name="requested speed", phase=[0, 1], metric="speed_x_m_s", min=.225, max=.275)]
        self.assertFalse(self.evaluate()[0]["passed"])
        self.traj["root_pos"][:, 0] *= 1.25
        self.assertTrue(self.evaluate()[0]["passed"])

    def test_wrap_jump_detected(self):
        self.traj["root_pos"][:, 0] = self.phase
        self.criteria = [dict(name="no reset jump", phase=[0, 1], metric="translation_step_max_m", max=.02)]
        self.assertFalse(self.evaluate()[0]["passed"])

    def test_orientation_wrap_jump_detected(self):
        self.traj["root_quat"] = Rotation.from_euler("z", self.phase*1000, degrees=True).as_quat()[:, [3, 0, 1, 2]]
        self.criteria = [dict(name="no orientation reset", phase=[0, 1], metric="rotation_step_max_deg", max=10)]
        self.assertFalse(self.evaluate()[0]["passed"])

    def test_supported_rotation_requires_simultaneous_conditions(self):
        self.criteria = [dict(name="front only turn", phase=[.3, .7],
                             metric="supported_heading_rotation_deg", min=270,
                             legs=["FL", "FR"], raised_legs=["RL", "RR"], nose_up_range=[-90, -75])]
        self.assertTrue(self.evaluate()[0]["passed"])
        self.feet[(self.phase >= .3) & (self.phase < .5), 2:, 2] = .0801
        self.assertFalse(self.evaluate()[0]["passed"])
