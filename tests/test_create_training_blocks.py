import unittest
import os
import sys
# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from TrainingScripts.create_training_csv import build_turn_rows


class TestCreateTrainingBlocks(unittest.TestCase):
    def test_block_pairing_basic(self):
        section = [
            {"Speaker":"Me","Content":"Hi","Date And Time":"Oct 12, 2024 6:14 AM"},
            {"Speaker":"+1","Content":"Hello","Date And Time":"Oct 12, 2024 6:15 AM"},
            {"Speaker":"+1","Content":"How are you?","Date And Time":"Oct 12, 2024 6:16 AM"},
            {"Speaker":"Me","Content":"Good","Date And Time":"Oct 12, 2024 6:20 AM"},
        ]
        entries = build_turn_rows(section)
        # Two Me turns produce two entries; test the last one
        self.assertGreaterEqual(len(entries), 1)
        e = entries[-1]
        # Context should have one user block and possibly prior Me block if any
        self.assertTrue(any(m["role"] == "user" for m in e["messages"]))
        # Target should be aggregated Me content
        self.assertTrue(e["target"].endswith("Good"))
        self.assertIn("[TS:", e["target"]) 
        # User message should include a delta tag
        self.assertTrue(any("[DELTA:(" in m["content"] for m in e["messages"] if m["role"] == "user"))


if __name__ == '__main__':
    unittest.main()
