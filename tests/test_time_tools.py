import unittest
import os
import sys
# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from TrainingScripts.ExtraTools import TimeTools


class TestTimeTools(unittest.TestCase):
    def test_delta_short(self):
        self.assertEqual(TimeTools.make_delta_tag(0), "[DELTA:(SHORT)]")

    def test_timestamp_parse_common(self):
        s = "10/12/2024 6:14 AM"
        dt = TimeTools.parse_timestamp(s)
        self.assertIsNotNone(dt)


if __name__ == '__main__':
    unittest.main()
