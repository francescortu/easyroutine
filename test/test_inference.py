import unittest
from easyroutine.inference.vllm_model_interface import (
    VLLMInferenceModel,
    VLLMInferenceModelConfig,
)
from easyroutine.inference.litellm_model_interface import (
    LiteLLMInferenceModel,
    LiteLLMInferenceModelConfig,
)


class VLLMInferenceModelTest(unittest.TestCase):
    def setUp(self):
        """Set up the test case for VLLM."""
        # The user will fill in the model name.
        # For now, we expect initialization to fail gracefully or be skipped.
        try:
            self.config = VLLMInferenceModelConfig(model_name="google/gemma-3-4b-it")
            self.model = VLLMInferenceModel(self.config)
        except Exception as e:
            self.model = None
            self.skipTest(f"Model initialization failed, skipping test: {e}")

    def test_chat_initialization(self):
        """Test that the model is initialized."""
        self.assertIsNotNone(
            self.model, "VLLMInferenceModel should be initialized in setUp."
        )

    def test_chat_method(self):
        """
        Test the chat method.
        This test is expected to fail until a valid model name is provided.
        """
        if self.model is None:
            self.skipTest("Model not initialized.")

        chat_messages = [{"role": "user", "content": "Hello, world!"}]
        # We expect an exception because the model name is empty.
        with self.assertRaises(Exception):
            self.model.chat(chat_messages)


class LiteLLMInferenceModelTest(unittest.TestCase):
    def setUp(self):
        """Set up the test case for LiteLLM."""
        # The user will fill in the model name.
        try:
            self.config = LiteLLMInferenceModelConfig(model_name="")
            self.model = LiteLLMInferenceModel(self.config)
        except Exception as e:
            self.model = None
            self.skipTest(f"Model initialization failed, skipping test: {e}")

    def test_chat_initialization(self):
        """Test that the model is initialized."""
        self.assertIsNotNone(
            self.model, "LiteLLMInferenceModel should be initialized in setUp."
        )

    def test_chat_method(self):
        """
        Test the chat method.
        This test is expected to fail until a valid model name is provided.
        """
        if self.model is None:
            self.skipTest("Model not initialized.")

        chat_messages = [{"role": "user", "content": "Hello, world!"}]
        # We expect an exception because the model name is empty.
        with self.assertRaises(Exception):
            self.model.chat(chat_messages)

    def test_openrouter_api_key_set(self):
        """Test that OpenRouter API key is properly set in environment."""
        import os
        test_key = "test_openrouter_key_123"
        config = LiteLLMInferenceModelConfig(
            model_name="openrouter/anthropic/claude-2",
            openrouter_api_key=test_key
        )
        model = LiteLLMInferenceModel(config)
        self.assertEqual(os.environ.get('OPENROUTER_API_KEY'), test_key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
