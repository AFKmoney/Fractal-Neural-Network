import unittest
import torch
from nfn.config import NFNConfig
from nfn.efficient_block import EfficientNFNLanguageModel

class TestEfficientBlock(unittest.TestCase):
    def setUp(self):
        self.cfg = NFNConfig(d_model=32, n_blocks=2, vocab_size=100)
        self.model = EfficientNFNLanguageModel(self.cfg)

    def test_param_summary_keys(self):
        summary = self.model.param_summary()
        expected_keys = {
            "total_params",
            "embed",
            "attn_per_block",
            "moe_per_block",
            "soliton/block",
            "lm_head",
            "analytic_buffers",
        }
        self.assertEqual(set(summary.keys()), expected_keys)

    def test_param_summary_values_types(self):
        summary = self.model.param_summary()
        for key, value in summary.items():
            self.assertIsInstance(value, int, f"Value for {key} is not an int")
            self.assertGreaterEqual(value, 0, f"Value for {key} is negative")

    def test_param_summary_sanity(self):
        summary = self.model.param_summary()
        self.assertGreater(summary["total_params"], 0)

        # Check that individual components don't exceed total
        parts_sum = (
            summary["embed"] +
            (summary["attn_per_block"] + summary["moe_per_block"] + summary["soliton/block"]) * self.cfg.n_blocks +
            summary["lm_head"]
            # Note: analytic_buffers are not part of model.parameters()
        )
        # It won't match exactly because of norm layers and other small modules not explicitly in the summary parts
        self.assertLessEqual(parts_sum, summary["total_params"])

    def test_param_summary_different_blocks(self):
        cfg1 = NFNConfig(d_model=32, n_blocks=1, vocab_size=100)
        model1 = EfficientNFNLanguageModel(cfg1)
        summary1 = model1.param_summary()

        cfg2 = NFNConfig(d_model=32, n_blocks=4, vocab_size=100)
        model2 = EfficientNFNLanguageModel(cfg2)
        summary2 = model2.param_summary()

        # Per block counts should be roughly the same (they are exactly the same here)
        self.assertEqual(summary1["attn_per_block"], summary2["attn_per_block"])
        self.assertEqual(summary1["moe_per_block"], summary2["moe_per_block"])
        self.assertEqual(summary1["soliton/block"], summary2["soliton/block"])

        # Total params should be strictly greater
        self.assertGreater(summary2["total_params"], summary1["total_params"])

if __name__ == "__main__":
    unittest.main()
