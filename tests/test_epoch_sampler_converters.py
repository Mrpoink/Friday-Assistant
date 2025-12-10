import unittest
import os
import sys
import pandas as pd
# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from TrainingScripts.epoch_sampler import convert_simple_instruction_dataset, convert_with_placeholders

class FakeDS:
    def __init__(self, rows):
        self.rows = rows
    def map(self, fn):
        return [fn(r) for r in self.rows]


class TestConverters(unittest.TestCase):
    def test_convert_simple_instruction_no_emotion(self):
        ds = FakeDS([{"instruction":"Solve 2+2","output":"4"}])
        df = convert_simple_instruction_dataset(ds, "logic", include_emotion=False)
        self.assertIsInstance(df, pd.DataFrame)
        user = df.iloc[0]["messages"]
        self.assertIn("[DELTA:(SHORT)]", user)
        self.assertNotIn("<joy>", user)
        self.assertNotIn("<sadness>", user)

    def test_convert_simple_instruction_with_emotion(self):
        ds = FakeDS([{"instruction":"I am happy","output":"ok"}])
        df = convert_simple_instruction_dataset(ds, "enigmata", include_emotion=True)
        user = df.iloc[0]["messages"]
        # Should include delta and some emotion tag bracket
        self.assertIn("[DELTA:(SHORT)]", user)
        self.assertTrue("<" in user and ">" in user)

    def test_convert_with_placeholders_no_emotion(self):
        ds = FakeDS([{"instruction":"Hello {{NAME}} by {{AUTHOR}}","output":"Hi"}])
        df = convert_with_placeholders(ds, "logic", include_emotion=False)
        user = df.iloc[0]["messages"]
        self.assertIn("Friday", user)
        self.assertIn("Brandon Dean", user)
        self.assertIn("[DELTA:(SHORT)]", user)
        self.assertNotIn("<joy>", user)


if __name__ == '__main__':
    unittest.main()
