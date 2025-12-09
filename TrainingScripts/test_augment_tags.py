import json
from TrainingScripts.ExtraTools import TimeTools, EmotionEngine

# Synthetic tests for augmentation logic

def test_delta_categories():
    seconds = [30*60, 2*60*60, 8*60*60, 20*60*60, 60*60*60]
    expected = ["SHORT","MEDIUM","LONG","EXTRA LONG","FOREVER"]
    cats = [TimeTools.categorize(s) for s in seconds]
    assert cats == expected, f"Delta categories mismatch: {cats} != {expected}"


def test_timestamp_parse_and_delta():
    ts1 = TimeTools.parse_timestamp("2025-12-05 13:00:00")
    ts2 = TimeTools.parse_timestamp("2025-12-05 15:30:00")
    sec = TimeTools.delta_seconds(ts1, ts2)
    assert sec == 2*60*60 + 30*60, f"Delta seconds wrong: {sec}"
    tag = TimeTools.make_delta_tag(sec)
    assert tag == "[DELTA:(MEDIUM)]", f"Tag wrong: {tag}"


def test_emotion_tagging():
    text = "I am so happy today!"
    tag = EmotionEngine.tag(text)
    assert tag == "<joy>", f"Emotion tag not joy: {tag}"


if __name__ == "__main__":
    test_delta_categories()
    test_timestamp_parse_and_delta()
    test_emotion_tagging()
    print("All augmentation tests passed.")
