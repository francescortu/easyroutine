import unittest
import torch
from torch.utils.data import DataLoader, Dataset
from easyroutine.interpretability.hooked_model import (
    HookedModel,
    HookedModelConfig,
    ExtractionConfig,
)
from easyroutine.interpretability.activation_cache import ActivationCache
from easyroutine.interpretability.interventions import  Intervention
from PIL import Image
import numpy as np
from typing import List


DEVICE = "auto"


class BaseHookedModelTestCase(unittest.TestCase):
    __test__ = False
    CONFIG = None
    MODEL: HookedModel
    INPUTS: dict
    TARGET_TOKEN_POSITION: List[str] = ["inputs-partition-0"]
    input_size: int
    # def setUp(self):
    #     """
    #     Set the config and load a tiny llama model for testing
    #     """
    #     config = HookedModelConfig(
    #         model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
    #         device_map="balanced",
    #         torch_dtype=torch.bfloat16,
    #         attn_implementation="eager",
    #         batch_size=1,
    #     )
    #     self.model = HookedModel(config)

    # config = HookedModelConfig(
    # model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
    # device_map="balanced",
    # torch_dtype=torch.bfloat16,
    # attn_implementation="custom_eager",
    # batch_size=1,
    # )
    # self.model = HookedModel(config)

    # self.INPUTS = {
    #     "input_ids": torch.tensor(
    #         [[101, 102, 103, 104, 105, 106]], device=self.model.device()
    #     ),
    #     "attention_mask": torch.tensor(
    #         [[1, 1, 1, 1, 1]], device=self.model.device()
    #     ),
    # }

    # self.TARGET_TOKEN_POSITION = ["inputs-partition-0"]

    def test_device(self):
        device = self.MODEL.device()
        self.assertEqual(device.type, "cuda")

    def test_to_string_tokens(self):
        """
        Test the conversion of tokens to string tokens
        """

        string_tokens = self.MODEL.to_string_tokens(self.INPUTS["input_ids"])
        self.assertEqual(len(string_tokens), self.input_size)

    def test_forward_without_pivot_positions(self):
        extracted_token_position = ["last"]
        cache = self.MODEL.forward(self.INPUTS, extracted_token_position)
        self.assertIn("logits", cache)

    def test_forward_with_pivot_positions(self):
        extracted_token_position = ["inputs-partition-0"]
        cache = self.MODEL.forward(
            self.INPUTS,
            extracted_token_position,
            pivot_positions=[4],
            extraction_config=ExtractionConfig(extract_resid_out=True),
        )
        # assert that cache["resid_out_0"] has shape (1,3,16)
        self.assertIn("resid_out_0", cache)
        self.assertEqual(
            cache["resid_out_0"].shape, (1, 4, self.MODEL.model_config.hidden_size)
        )

        extracted_token_position = ["inputs-partition-1"]
        cache = self.MODEL.forward(
            self.INPUTS,
            extracted_token_position,
            pivot_positions=[4],
            extraction_config=ExtractionConfig(extract_resid_out=True),
        )
        # assert that cache["resid_out_0"] has shape (1,2,16)
        self.assertIn("resid_out_0", cache)
        self.assertEqual(
            cache["resid_out_0"].shape,
            (1, self.input_size - 4, self.MODEL.model_config.hidden_size),
        )

    def test_extract_cache(self):
        """
        Test the extract_cache method of HookedModel.
        """

        # class CustomDataset(Dataset):
        #     def __init__(self, data):
        #         self.data = data

        #     def __len__(self):
        #         return len(self.data)

        #     def __getitem__(self, idx):
        #         return self.data[idx]

        dataloader = [self.INPUTS, self.INPUTS]
        # dataset = CustomDataset(data)
        # dataloader = DataLoader(dataset)

        target_token_positions = ["last"]

        def batch_saver(batch):
            return {"batch_info": batch}

        final_cache = self.MODEL.extract_cache(
            dataloader,
            target_token_positions=target_token_positions,
            batch_saver=batch_saver,
            extraction_config=ExtractionConfig(
                extract_resid_out=True, extract_last_layernorm=True
            ),
        )

        self.assertIn("logits", final_cache)
        self.assertIn("last_layernorm", final_cache)
        self.assertIn("resid_out_0", final_cache)
        self.assertIn("mapping_index", final_cache)
        self.assertTrue(torch.is_tensor(final_cache["logits"]))

    def test_pattern_with_extract_cache(self):
        """
        Pattern is dangerous, needs its own test
        """
        # raise test failed
        self.assertTrue(False)

    def test_hook_resid_out(self):
        cache = self.MODEL.forward(
            self.INPUTS,
            self.TARGET_TOKEN_POSITION,
            pivot_positions=[4],
            extraction_config=ExtractionConfig(extract_resid_out=True),
        )
        # assert that cache["resid_out_0"] has shape (1,3,16)
        self.assertIn("resid_out_0", cache)
        self.assertEqual(
            cache["resid_out_0"].shape, (1, 4, self.MODEL.model_config.hidden_size)
        )

    def test_hook_resid_out_avg(self):
        cache = self.MODEL.forward(
            self.INPUTS,
            self.TARGET_TOKEN_POSITION + ["all"],
            pivot_positions=[4],
            extraction_config=ExtractionConfig(extract_resid_out=True, avg=True),
        )
        self.assertIn("resid_out_0", cache)
        self.assertEqual(
            cache["resid_out_0"].shape, (1, 2, self.MODEL.model_config.hidden_size)
        )

    def test_slice_tokens(self):
        """
        Test the slice_tokens support in the forward method
        """
        cache = self.MODEL.forward(
            self.INPUTS,
            [(0, 4)],
            pivot_positions=[4],
            extraction_config=ExtractionConfig(extract_resid_out=True),
        )

        self.assertIn("resid_out_0", cache)
        self.assertEqual(
            cache["resid_out_0"].shape, (1, 4, self.MODEL.model_config.hidden_size)
        )

    def test_hook_resid_out_multimodal(self):
        if not self.MODEL.is_multimodal():
            return
        cache = self.MODEL.forward(
            self.INPUTS,
            ["all-image"],
            pivot_positions=[4],
            extraction_config=ExtractionConfig(extract_resid_out=True),
        )
        # assert that cache resid_out_0 has shape (1,something,hid_size)
        self.assertIn("resid_out_0", cache)
        self.assertEqual(
            cache["resid_out_0"][:, :4, :].shape,
            (1, 4, self.MODEL.model_config.hidden_size),
        )

    def test_hook_resid_in(self):
        cache = self.MODEL.forward(
            self.INPUTS,
            self.TARGET_TOKEN_POSITION,
            pivot_positions=[4],
            extraction_config=ExtractionConfig(extract_resid_in=True),
        )
        # assert that cache["resid_in_0"] has shape (1,3,16)
        self.assertIn("resid_in_0", cache)
        self.assertEqual(
            cache["resid_in_0"].shape, (1, 4, self.MODEL.model_config.hidden_size)
        )

    def test_hook_resid_in_avg(self):
        cache = self.MODEL.forward(
            self.INPUTS,
            self.TARGET_TOKEN_POSITION + ["all"],
            pivot_positions=[4],
            extraction_config=ExtractionConfig(extract_resid_in=True, avg=True),
        )
        self.assertIn("resid_in_0", cache)
        self.assertEqual(
            cache["resid_in_0"].shape, (1, 2, self.MODEL.model_config.hidden_size)
        )

    def test_hook_resid_mid(self):
        cache = self.MODEL.forward(
            self.INPUTS,
            self.TARGET_TOKEN_POSITION,
            pivot_positions=[4],
            extraction_config=ExtractionConfig(extract_resid_mid=True),
        )
        # assert that cache["resid_mid_0"] has shape (1,3,16)
        self.assertIn("resid_mid_0", cache)
        self.assertEqual(
            cache["resid_mid_0"].shape, (1, 4, self.MODEL.model_config.hidden_size)
        )

    def test_hook_resid_mid_avg(self):
        cache = self.MODEL.forward(
            self.INPUTS,
            self.TARGET_TOKEN_POSITION + ["all"],
            pivot_positions=[4],
            extraction_config=ExtractionConfig(extract_resid_mid=True, avg=True),
        )
        self.assertIn("resid_mid_0", cache)
        self.assertEqual(
            cache["resid_mid_0"].shape, (1, 2, self.MODEL.model_config.hidden_size)
        )

    def test_hook_extract_head_key_value_keys(self):
        self.MODEL.restore_original_modules()
        cache = self.MODEL.forward(
            self.INPUTS,
            self.TARGET_TOKEN_POSITION,
            pivot_positions=[4],
            extraction_config=ExtractionConfig(
                extract_head_keys=True,
                extract_head_values=True,
                extract_head_queries=True,
            ),
        )

        # assert that cache have "values_L0H1" and "keys_L0H1" and "queries_L0H1"
        self.assertIn("values_L0H1", cache)
        self.assertEqual(
            cache["values_L0H1"].shape, (1, 4, self.MODEL.model_config.head_dim)
        )
        self.assertIn("keys_L0H1", cache)
        self.assertEqual(
            cache["keys_L0H1"].shape, (1, 4, self.MODEL.model_config.head_dim)
        )
        self.assertIn("queries_L0H1", cache)
        self.assertEqual(
            cache["queries_L0H1"].shape, (1, 4, self.MODEL.model_config.head_dim)
        )

    def test_hook_extract_head_key_value_keys_avg(self):
        self.MODEL.restore_original_modules()
        cache = self.MODEL.forward(
            self.INPUTS,
            self.TARGET_TOKEN_POSITION + ["all"],
            pivot_positions=[4],
            extraction_config=ExtractionConfig(
                extract_head_keys=True,
                extract_head_values=True,
                extract_head_queries=True,
                avg=True,
            ),
        )

        # assert that cache have "values_L0H1" and "keys_L0H1" and "queries_L0H1"
        self.assertIn("values_L0H1", cache)
        self.assertEqual(
            cache["values_L0H1"].shape, (1, 2, self.MODEL.model_config.head_dim)
        )
        self.assertIn("keys_L0H1", cache)
        self.assertEqual(
            cache["keys_L0H1"].shape, (1, 2, self.MODEL.model_config.head_dim)
        )
        self.assertIn("queries_L0H1", cache)
        self.assertEqual(
            cache["queries_L0H1"].shape, (1, 2, self.MODEL.model_config.head_dim)
        )

    def test_hook_extract_head_out(self):
        cache = self.MODEL.forward(
            self.INPUTS,
            self.TARGET_TOKEN_POSITION,
            pivot_positions=[4],
            extraction_config=ExtractionConfig(extract_head_out=True),
        )
        # assert that cache["head_out_L0H1"] has shape (1, num_heads, 5, hidden_size)
        self.assertIn("head_out_L0H1", cache)
        self.assertEqual(
            cache["head_out_L0H1"].shape,
            (1, 4, self.MODEL.model_config.hidden_size),
        )

    def test_hook_extract_attn_in(self):
        cache = self.MODEL.forward(
            self.INPUTS,
            self.TARGET_TOKEN_POSITION,
            pivot_positions=[4],
            extraction_config=ExtractionConfig(extract_attn_in=True),
        )
        # assert that cache["attn_in_0"] has shape (1, 4, )
        self.assertIn("attn_in_0", cache)
        self.assertEqual(
            cache["attn_in_0"].shape, (1, 4, self.MODEL.model_config.hidden_size)
        )

    def test_hook_extract_attn_out(self):
        cache = self.MODEL.forward(
            self.INPUTS,
            self.TARGET_TOKEN_POSITION,
            pivot_positions=[4],
            extraction_config=ExtractionConfig(extract_attn_out=True),
        )
        # assert that cache["attn_out_0"] has shape (1, 4, )
        self.assertIn("attn_out_0", cache)
        self.assertEqual(
            cache["attn_out_0"].shape, (1, 4, self.MODEL.model_config.hidden_size)
        )

    def test_hook_extract_mlp_out(self):
        cache = self.MODEL.forward(
            self.INPUTS,
            self.TARGET_TOKEN_POSITION,
            pivot_positions=[4],
            extraction_config=ExtractionConfig(extract_mlp_out=True),
        )
        # assert that cache["mlp_out_0"] has shape (1, 4, )
        self.assertIn("mlp_out_0", cache)
        self.assertEqual(
            cache["mlp_out_0"].shape, (1, 4, self.MODEL.model_config.hidden_size)
        )

    def test_hook_extract_last_layernorm(self):
        cache = self.MODEL.forward(
            self.INPUTS,
            self.TARGET_TOKEN_POSITION,
            pivot_positions=[4],
            extraction_config=ExtractionConfig(extract_last_layernorm=True),
        )
        # assert that cache["last_layernorm
        self.assertIn("last_layernorm", cache)

    def test_hook_extract_resid_in_post_layernorm(self):
        cache = self.MODEL.forward(
            self.INPUTS,
            self.TARGET_TOKEN_POSITION,
            pivot_positions=[4],
            extraction_config=ExtractionConfig(extract_resid_in_post_layernorm=True),
        )
        # assert that cache["resid_in_post_layernorm_0"] has shape (1, 4, 16)
        self.assertIn("resid_in_post_layernorm_0", cache)
        self.assertEqual(
            cache["resid_in_post_layernorm_0"].shape,
            (1, 4, self.MODEL.model_config.hidden_size),
        )

    def test_hook_extract_avg_over_examples_attn_pattern(self):
        """
        Original minimal test. We'll keep it here as-is for reference.
        """
        external_cache = ActivationCache()
        external_cache["avg_pattern_L1H1"] = torch.randn(
            1, self.input_size, self.input_size
        )

        # Example call
        cache = self.MODEL.forward(
            self.INPUTS,
            target_token_positions=["all"],
            pivot_positions=[4],
            extraction_config=ExtractionConfig(
                extract_attn_pattern=True,
                avg_over_example=True,
                attn_pattern_row_positions=["last"],
            ),
            external_cache=external_cache,
            batch_idx=1,
        )

        # Assert that cache["avg_pattern_L1H1"] has shape (1, self.input_size, self.input_size)
        self.assertIn("avg_pattern_L1H1", external_cache)
        self.assertEqual(
            external_cache["avg_pattern_L1H1"].shape,
            (1, self.input_size, self.input_size),
        )

    def test_attn_pattern_all_methods_and_row_partitions(self):
        """
        Expanded test to verify that attention_pattern_head handles
        multiple avg_methods and row_partitions as intended.
        """
        methods = ["mean", "sum", "baseline_ratio"]

        # Example partitions:
        #  1) None -> uses the same token groups as `target_token_positions`
        #  2) "last" -> if your code interprets strings as special instructions
        #  3) "all"  -> similarly
        #  4) Explicit partitions as a list of tuples, e.g. [(0,1,2), (3,4,5)]
        row_partitions = [
            None,
            ["all"],
            ["last"],
            [(0, 3), (3, 5)],  # example: two groups
        ]

        for method in methods:
            for partition in row_partitions:
                with self.subTest(method=method, row_partition=partition):
                    # You can create a fresh cache each time, or reuse if you prefer.

                    # Example call: pick some pivot positions or target token positions.
                    # Here we do something more interesting than the single pivot=4
                    # so you can see how multiple groups might be tested.
                    # If your code needs token_positions to be a list of tuples, you can do so:
                    target_pos = [(0, 1), (2, 3)]  # example grouping
                    pivot_pos = [4, 5]  # just an example

                    cache = self.MODEL.forward(
                        self.INPUTS,
                        target_token_positions=target_pos,
                        pivot_positions=pivot_pos,
                        extraction_config=ExtractionConfig(
                            extract_attn_pattern=True,
                            avg_over_example=False,  # ensures we run the averaging path
                            attn_pattern_avg=method,  # type: ignore
                            attn_pattern_row_positions=partition,
                        ),
                        batch_idx=0,  # or 1, or whichever
                    )

                    # Depending on your usage, you might store the results in specific keys
                    # For example, "pattern_L0H0", "pattern_L0H1", etc.
                    # So we can assert they exist in external_cache:
                    for h in range(
                        self.MODEL.model_config.num_attention_heads
                    ):  # adapt to your model
                        key = f"pattern_L0H{h}"
                        # or if your code calls add_with_info as "pattern_L{layer}H{h}"
                        # or "avg_pattern_L{layer}H{h}", etc.
                        self.assertIn(
                            key,
                            cache.keys(),
                            msg=f"Missing {key} in cache with method={method} partition={partition}",
                        )
                        if partition is None:
                            self.assertEqual(
                                cache[key].shape,
                                (1, 2, 2),
                                msg=f"Shape mismatch for {key} with method={method} partition={partition}",
                            )
                        else:
                            if len(partition) == 1:
                                self.assertEqual(
                                    cache[key].shape,
                                    (1, 1, 2),
                                    msg=f"Shape mismatch for {key} with method={method} partition={partition}",
                                )
                            elif len(partition) == 2:
                                self.assertEqual(
                                    cache[key].shape,
                                    (1, 2, 2),
                                    msg=f"Shape mismatch for {key} with method={method} partition={partition}",
                                )

    def test_hook_extract_attn_pattern(self):
        cache = self.MODEL.forward(
            self.INPUTS,
            ["all"],
            pivot_positions=[4],
            extraction_config=ExtractionConfig(extract_attn_pattern=True),
        )
        print(cache.keys())
        # assert that cache["attn_pattern_0"] has shape (1, 4, 16, 16)
        self.assertIn("pattern_L1H1", cache)
        self.assertEqual(
            cache["pattern_L1H1"].shape, (1, self.input_size, self.input_size)
        )

    def test_module_wrapper(self):
        """
        Test if the wrapper that substitutes part of the model works equivalently to the original model.
        """

        # run the model with the original model
        self.MODEL.restore_original_modules()
        cache_original = self.MODEL.forward(
            self.INPUTS,
            self.TARGET_TOKEN_POSITION,
            extraction_config=ExtractionConfig(
                extract_resid_out=True,
                extract_resid_in=True,
                extract_resid_mid=True,
                extract_attn_in=True,
            ),
        )

        self.MODEL.set_custom_modules()
        cache_custom = self.MODEL.forward(
            self.INPUTS,
            self.TARGET_TOKEN_POSITION,
            extraction_config=ExtractionConfig(
                extract_resid_out=True,
                extract_resid_in=True,
                extract_resid_mid=True,
                extract_attn_in=True,
            ),
        )

        # assert that the output of the original model and the custom model are the same
        for key in cache_original.keys():
            if key in cache_custom:
                if isinstance(cache_original[key], torch.Tensor):
                    if not torch.allclose(cache_original[key], cache_custom[key]):
                        print(f"Mismatch found in key: {key}")
                    self.assertTrue(
                        torch.allclose(cache_original[key], cache_custom[key])
                    )

    def test_get_last_layernorm(self):
        norm = self.MODEL.get_last_layernorm()
        self.assertIsNotNone(norm)

    def get_lm_head(self):
        unembed = self.MODEL.get_lm_head()

        # assert is not None
        self.assertIsNotNone(unembed)

    def test_generate(self):
        pass  # TODO: Implement this test

    def test_generate_with_extract_cache(self):
        # TODO: Implement this test
        pass

    def test_ablation_attn_matrix(self):
        ablation_cache = self.MODEL.forward(
            self.INPUTS,
            target_token_positions=["all"],
            pivot_positions=[4],
            extraction_config=ExtractionConfig(
                extract_resid_out=True,
                extract_attn_pattern=True,
            ),
            interventions=[
                Intervention(
                    type="columns",
                    activation="pattern_L1H2",
                    token_positions = ["inputs-partition-0"],
                    patching_values = "ablation"
                )
            ]
        )

        # cache = self.MODEL.forward(
        #     self.INPUTS,
        #     target_token_positions=["last"],
        #     pivot_positions=[4],
        #     extraction_config=ExtractionConfig(
        #         extract_resid_out=True,
        #         extract_attn_pattern=True,
        #     )
        # )
        # assert that cache["pattern_L1H1"] is in the cache
        self.assertIn("pattern_L1H1", ablation_cache)
        # assert that cache["resid_out_0"] has shape (1,self.input_size,self.input_size)
        self.assertEqual(
            ablation_cache["pattern_L1H2"].shape, (1, self.input_size, self.input_size)
        )

        self.assertEqual(ablation_cache["pattern_L1H2"][0, :4, :4].sum(), 0)

    def test_patching(self):
        # TODO: Implement this test
        pass

    def test_token_index(self):
        # TODO: add edge cases for token_index
        pass
################### BASE TEST CASES ######################
class TestHookedTestModel(BaseHookedModelTestCase):
    """
    This is a *concrete* test class recognized by the VS Code test explorer.
    It inherits all test_* methods from BaseHookedModelTestCase.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.MODEL = HookedModel(
            HookedModelConfig(
                model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="custom_eager",
                batch_size=1,
            )
        )
        cls.INPUTS = {
            "input_ids": torch.tensor([[101, 102, 103, 104, 105, 106]], device="cuda"),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], device="cuda"),
        }
        cls.input_size = cls.INPUTS["input_ids"].shape[1]


################# Utils ####################


def get_a_random_pil():
    # Define image dimensions
    width, height = 256, 256

    # Create random pixel data
    random_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    # Create an image from the random data
    random_image = Image.fromarray(random_data)

    return random_image


################## Test Cases for Chameleon Model ####################


class TestHookedChameleonModel(BaseHookedModelTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.MODEL = HookedModel(
            HookedModelConfig(
                model_name="facebook/chameleon-7b",
                device_map=DEVICE,
                torch_dtype=torch.bfloat16,
                attn_implementation="custom_eager",
                batch_size=1,
            )
        )
        tokenizer = cls.MODEL.get_tokenizer()
        cls.INPUTS = tokenizer(
            text="This is a test. <image>. This is a test",
            images=[
                # pil image between 0 and 1
                get_a_random_pil()
            ],
            return_tensors="pt",
        )  # type: ignore
        cls.INPUTS = {k: v.to(cls.MODEL.device()) for k, v in cls.INPUTS.items()}
        cls.INPUTS["pixel_values"] = cls.INPUTS["pixel_values"].to(
            cls.MODEL.config.torch_dtype
        )
        cls.input_size = cls.INPUTS["input_ids"].shape[1]


################## Test Cases for pixtral Model ####################


class TestHookedPixtralModel(BaseHookedModelTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.MODEL = HookedModel(
            HookedModelConfig(
                model_name="mistral-community/pixtral-12b",
                device_map=DEVICE,
                torch_dtype=torch.bfloat16,
                attn_implementation="custom_eager",
                batch_size=1,
            )
        )
        tokenizer = cls.MODEL.get_tokenizer()
        cls.INPUTS = tokenizer(
            text="This is a test. [IMG]. This is a test",
            images=[get_a_random_pil()],
            return_tensors="pt",
        )

        cls.input_size = cls.INPUTS["input_ids"].shape[1]


################## Test Cases for llava Model ####################


class TestHookedLlavaModel(BaseHookedModelTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.MODEL = HookedModel(
            HookedModelConfig(
                model_name="llava-hf/llava-v1.6-mistral-7b-hf",
                device_map=DEVICE,
                torch_dtype=torch.bfloat16,
                attn_implementation="custom_eager",
                batch_size=1,
            )
        )
        tokenizer = cls.MODEL.get_tokenizer()
        cls.INPUTS = tokenizer(
            text="This is a test. <image>. This is a test",
            images=[get_a_random_pil()],
            return_tensors="pt",
        )  # type: ignore

        cls.input_size = cls.INPUTS["input_ids"].shape[1]

class TestHookedGemma3Model(BaseHookedModelTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.MODEL = HookedModel(
            HookedModelConfig(
                model_name="google/gemma-3-4b-it",
                device_map=DEVICE,
                torch_dtype=torch.bfloat16,
                attn_implementation="custom_eager",
                batch_size=1,
            )
        )
        tokenizer = cls.MODEL.get_tokenizer()
        # messages = [
        #     {
        #         "role": "system",
        #         "content": [{"type": "text", "text": "You are a helpful assistant."}]
        #     },
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "image": get_a_random_pil()},
        #             {"type": "text", "text": "Describe this image in detail."}
        #         ]
        #     }
        # ]

        # cls.INPUTS  = tokenizer.apply_chat_template(
        #     messages, add_generation_prompt=True, tokenize=True,
        #     return_dict=True, return_tensors="pt"
        # )
        cls.INPUTS = tokenizer(text="<start_of_image>", images=[get_a_random_pil()], return_tensors="pt")
        
        cls.input_size = cls.INPUTS["input_ids"].shape[1]
        

# if __name__ == "__main__":
#     unittest.main(verbosity=2)


if __name__ == "__main__":
    unittest.main()
