import unittest
import torch
from easyroutine.interpretability import HookedModel, ExtractionConfig
from easyroutine.interpretability.tools import LogitLens


class TestLogitLens(unittest.TestCase):
    
    def setUp(self):
        model = HookedModel.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        self.logit_lens = LogitLens.from_model(model)
        fake_dataset = [
            {
                "input_ids": torch.randint(0, 100, (1, 100)),
                "attention_mask": torch.randint(0, 2, (1, 100)),
            },
            {
                "input_ids": torch.randint(0, 100, (1, 100)),
                "attention_mask": torch.randint(0, 2, (1, 100)),
            }
        ]

        self.cache = model.extract_cache(
                fake_dataset,
                target_token_positions=["last"],
                extraction_config = ExtractionConfig(
                    extract_resid_out=True,
                    extract_last_layernorm=True,
                    # extract_head_out=True,
                )
        )
        
    def test_compute(self):
        
        logit_lens_out = self.logit_lens.compute(self.cache, "resid_out_{i}")
        
        #assert logit_lens_resid_out_0 in logit_lens_out
        #assert logit_lens_resid_out_1 in logit_lens_out
        self.assertTrue("logit_lens_resid_out_0" in logit_lens_out)
        
        # assert logit_lens_out["logit_lens_resid_out_0"].shape == (2,1,32000)
        
        self.assertTrue(logit_lens_out["logit_lens_resid_out_0"].shape == (2,1,32000))
    
        
if __name__ == "__main__":
    unittest.main()
