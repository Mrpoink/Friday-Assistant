import unittest
import os
import sys
# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from TrainingScripts.epoch_sampler import inject_identity


class TestIdentityInjection(unittest.TestCase):
    def test_inject_identity_self_referential(self):
        s = "I am an AI built by OpenAI."
        out = inject_identity(s)
        self.assertIn("I am Friday", out)
        # Proper noun remains
        self.assertIn("OpenAI", out)
        self.assertIn("server:self local machine", out.lower())

    def test_inject_identity_avoid_proper_nouns(self):
        s = "OpenAI released GPT-4. Anthropic released Claude."
        out = inject_identity(s)
        # Should not replace proper nouns
        self.assertIn("OpenAI released GPT-4", out)
        self.assertIn("Anthropic released Claude", out)
        self.assertIn("server:self local machine", out.lower())


if __name__ == '__main__':
    unittest.main()
