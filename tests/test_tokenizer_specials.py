import unittest
import os
import sys
# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from Friday_Tokenizer_Creation import create_tokenizer


class TestTokenizerSpecials(unittest.TestCase):
    def test_tokenizer_specials_saved(self):
        tok = create_tokenizer()
        self.assertIsNotNone(tok)
        specials = tok.special_tokens_map.get("additional_special_tokens", [])
        required = ["<think>", "</think>", "<tool_call>", "[identity]", "[DELTA:(SHORT)]", "[TS:", "[TIME:"]
        for r in required:
            self.assertIn(r, specials)


if __name__ == '__main__':
    unittest.main()
