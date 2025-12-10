import unittest
import os
import sys
import numpy as np
# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import TrainingScripts.ExtraTools as ET


class TestEmotionEngine(unittest.TestCase):
    def test_emotion_engine_fallback_neutral(self):
        # Force pipeline to None
        ET._EMOTION_PIPELINE = None
        label, vec = ET.EmotionEngine.get_emotion_data("any text")
        self.assertEqual(label, "neutral")
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape[0], len(ET.EmotionEngine.ordered_labels))

    def test_emotion_tag_output(self):
        # Fake pipeline returning amusement
        class FakePipe:
            def __call__(self, text):
                return [[{"label":"amusement","score":0.9},{"label":"neutral","score":0.1}]]
        ET._EMOTION_PIPELINE = FakePipe()
        tag = ET.EmotionEngine.tag("text")
        self.assertEqual(tag, "<amusement>")


if __name__ == '__main__':
    unittest.main()
