import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from utils.student_rollout import student_probability


class ScheduleTests(unittest.TestCase):
    def test_schedule(self):
        cfg = dict(warmup_steps=100000, ramp_steps=400000, max_student_probability=1.)
        for step, expected in [(0, 0), (100000, 0), (300000, .5), (500000, 1), (1000000, 1)]:
            self.assertEqual(student_probability(step, cfg), expected)

    def test_invalid(self):
        for cfg in [dict(warmup_steps=-1, ramp_steps=1), dict(warmup_steps=0, ramp_steps=0),
                    dict(warmup_steps=0, ramp_steps=1, max_student_probability=2)]:
            with self.assertRaises(ValueError):
                student_probability(0, cfg)


if __name__ == '__main__':
    unittest.main()
