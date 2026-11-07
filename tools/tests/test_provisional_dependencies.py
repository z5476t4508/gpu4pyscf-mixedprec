import importlib.util
import os
import pathlib
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("agent", pathlib.Path(__file__).parents[1] / "consumer_gpu_agent.py")
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)

class ProvisionalDependencyTests(unittest.TestCase):
    def test_default_still_requires_acceptance(self):
        with patch.dict(os.environ, {"QC_CONTINUE_UNREVIEWED": "0"}):
            self.assertFalse(agent.dependency_ready({"status": "review_requested", "tests": [1]}))
            self.assertTrue(agent.dependency_ready({"status": "accepted"}))

    def test_provisional_requires_handoff_and_checks(self):
        with patch.dict(os.environ, {"QC_CONTINUE_UNREVIEWED": "1"}):
            for status in ("pending", "running", "implemented", "blocked", "needs_changes"):
                self.assertFalse(agent.dependency_ready({"status": status, "tests": [1]}))
            self.assertFalse(agent.dependency_ready({"status": "review_requested", "tests": []}))
            self.assertTrue(agent.dependency_ready({"status": "review_requested", "tests": [1]}))

    def test_provisional_does_not_mark_accepted(self):
        task = {"status": "review_requested", "tests": [1]}
        with patch.dict(os.environ, {"QC_CONTINUE_UNREVIEWED": "1"}):
            self.assertTrue(agent.dependency_ready(task))
        self.assertEqual(task["status"], "review_requested")
