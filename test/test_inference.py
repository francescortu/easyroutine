import unittest
from abc import ABC
from unittest.mock import MagicMock, patch
import sys

# Import directly to avoid the __init__.py that imports vllm
from easyroutine.inference.base_model_interface import (
    BaseInferenceModelConfig,
    BaseInferenceModel
)


class TestBaseInferenceModelConfig(unittest.TestCase):
    """Test suite for BaseInferenceModelConfig dataclass."""

    def test_config_creation_with_defaults(self):
        """Test creating config with only required parameters."""
        config = BaseInferenceModelConfig(model_name="test-model")
        
        # Check required parameter
        self.assertEqual(config.model_name, "test-model")
        
        # Check default values
        self.assertEqual(config.n_gpus, 1)
        self.assertEqual(config.dtype, 'bfloat16')
        self.assertEqual(config.temperature, 0)
        self.assertEqual(config.top_p, 0.95)
        self.assertEqual(config.max_new_tokens, 5000)

    def test_config_creation_with_custom_values(self):
        """Test creating config with custom parameters."""
        config = BaseInferenceModelConfig(
            model_name="custom-model",
            n_gpus=4,
            dtype="float16",
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=1000
        )
        
        self.assertEqual(config.model_name, "custom-model")
        self.assertEqual(config.n_gpus, 4)
        self.assertEqual(config.dtype, "float16")
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.max_new_tokens, 1000)

    def test_config_is_dataclass(self):
        """Test that config behaves as a dataclass."""
        config1 = BaseInferenceModelConfig(model_name="test")
        config2 = BaseInferenceModelConfig(model_name="test")
        config3 = BaseInferenceModelConfig(model_name="different")
        
        # Same parameters should be equal
        self.assertEqual(config1, config2)
        # Different parameters should not be equal
        self.assertNotEqual(config1, config3)


class TestBaseInferenceModel(unittest.TestCase):
    """Test suite for BaseInferenceModel abstract class."""

    def setUp(self):
        """Set up test environment."""
        # Create a concrete implementation for testing
        class ConcreteInferenceModel(BaseInferenceModel):
            def convert_chat_messages_to_custom_format(self, chat_messages):
                return f"Converted: {chat_messages}"
            
            def generate(self, inputs):
                return "Generated response"
        
        self.ConcreteModel = ConcreteInferenceModel

    def test_base_model_is_abstract(self):
        """Test that BaseInferenceModel is abstract and cannot be instantiated."""
        config = BaseInferenceModelConfig(model_name="test")
        
        with self.assertRaises(TypeError):
            BaseInferenceModel(config)

    def test_concrete_model_initialization(self):
        """Test initialization of concrete model."""
        config = BaseInferenceModelConfig(model_name="test-model")
        model = self.ConcreteModel(config)
        
        self.assertEqual(model.config, config)
        self.assertEqual(model.config.model_name, "test-model")

    def test_init_model_class_method(self):
        """Test the init_model class method."""
        model = self.ConcreteModel.init_model(
            model_name="test-model",
            n_gpus=2,
            dtype="float32"
        )
        
        self.assertIsInstance(model, self.ConcreteModel)
        self.assertEqual(model.config.model_name, "test-model")
        self.assertEqual(model.config.n_gpus, 2)
        self.assertEqual(model.config.dtype, "float32")

    def test_init_model_with_defaults(self):
        """Test init_model with default parameters."""
        model = self.ConcreteModel.init_model(model_name="test-model")
        
        self.assertEqual(model.config.model_name, "test-model")
        self.assertEqual(model.config.n_gpus, 1)  # default
        self.assertEqual(model.config.dtype, 'bfloat16')  # default

    def test_append_with_chat_template_empty_history(self):
        """Test appending message to empty chat history."""
        config = BaseInferenceModelConfig(model_name="test")
        model = self.ConcreteModel(config)
        
        result = model.append_with_chat_template("Hello world")
        
        expected = [{'role': 'user', 'content': 'Hello world'}]
        self.assertEqual(result, expected)

    def test_append_with_chat_template_with_role(self):
        """Test appending message with specific role."""
        config = BaseInferenceModelConfig(model_name="test")
        model = self.ConcreteModel(config)
        
        result = model.append_with_chat_template(
            "System message", 
            role='system'
        )
        
        expected = [{'role': 'system', 'content': 'System message'}]
        self.assertEqual(result, expected)

    def test_append_with_chat_template_existing_history(self):
        """Test appending message to existing chat history."""
        config = BaseInferenceModelConfig(model_name="test")
        model = self.ConcreteModel(config)
        
        existing_history = [
            {'role': 'system', 'content': 'System message'},
            {'role': 'user', 'content': 'First user message'}
        ]
        
        result = model.append_with_chat_template(
            "Second user message",
            role='user',
            chat_history=existing_history
        )
        
        expected = existing_history + [{'role': 'user', 'content': 'Second user message'}]
        self.assertEqual(result, expected)

    def test_append_with_chat_template_assistant_role(self):
        """Test appending assistant message."""
        config = BaseInferenceModelConfig(model_name="test")
        model = self.ConcreteModel(config)
        
        chat_history = [{'role': 'user', 'content': 'Question?'}]
        
        result = model.append_with_chat_template(
            "Assistant response",
            role='assistant',
            chat_history=chat_history
        )
        
        expected = [
            {'role': 'user', 'content': 'Question?'},
            {'role': 'assistant', 'content': 'Assistant response'}
        ]
        self.assertEqual(result, expected)

    def test_append_with_chat_template_validates_history(self):
        """Test that invalid chat history raises assertion error."""
        config = BaseInferenceModelConfig(model_name="test")
        model = self.ConcreteModel(config)
        
        # Missing 'role' key
        invalid_history = [{'content': 'message without role'}]
        
        with self.assertRaises(AssertionError):
            model.append_with_chat_template(
                "New message",
                chat_history=invalid_history
            )

    def test_append_with_chat_template_validates_history_missing_content(self):
        """Test validation with missing 'content' key."""
        config = BaseInferenceModelConfig(model_name="test")
        model = self.ConcreteModel(config)
        
        # Missing 'content' key
        invalid_history = [{'role': 'user'}]
        
        with self.assertRaises(AssertionError):
            model.append_with_chat_template(
                "New message",
                chat_history=invalid_history
            )

    def test_append_with_chat_template_empty_list_is_valid(self):
        """Test that empty chat history list is valid."""
        config = BaseInferenceModelConfig(model_name="test")
        model = self.ConcreteModel(config)
        
        # Should not raise assertion error
        result = model.append_with_chat_template(
            "First message",
            chat_history=[]
        )
        
        expected = [{'role': 'user', 'content': 'First message'}]
        self.assertEqual(result, expected)

    def test_convert_chat_messages_to_custom_format_is_abstract(self):
        """Test that convert_chat_messages_to_custom_format must be implemented."""
        class IncompleteModel(BaseInferenceModel):
            pass  # Missing implementation
        
        config = BaseInferenceModelConfig(model_name="test")
        
        with self.assertRaises(TypeError):
            IncompleteModel(config)

    def test_concrete_implementation_of_convert_method(self):
        """Test the concrete implementation of convert_chat_messages_to_custom_format."""
        config = BaseInferenceModelConfig(model_name="test")
        model = self.ConcreteModel(config)
        
        messages = [{'role': 'user', 'content': 'Hello'}]
        result = model.convert_chat_messages_to_custom_format(messages)
        
        self.assertEqual(result, f"Converted: {messages}")

    def test_role_validation_types(self):
        """Test that role parameter accepts valid literal types."""
        config = BaseInferenceModelConfig(model_name="test")
        model = self.ConcreteModel(config)
        
        # These should all work without error
        user_result = model.append_with_chat_template("Message", role='user')
        assistant_result = model.append_with_chat_template("Message", role='assistant')
        system_result = model.append_with_chat_template("Message", role='system')
        
        self.assertEqual(user_result[0]['role'], 'user')
        self.assertEqual(assistant_result[0]['role'], 'assistant')
        self.assertEqual(system_result[0]['role'], 'system')

    def test_chat_history_immutability(self):
        """Test that original chat history is not modified."""
        config = BaseInferenceModelConfig(model_name="test")
        model = self.ConcreteModel(config)
        
        original_history = [{'role': 'user', 'content': 'Original message'}]
        original_copy = original_history.copy()
        
        result = model.append_with_chat_template(
            "New message",
            chat_history=original_history
        )
        
        # Original history should be unchanged
        self.assertEqual(original_history, original_copy)
        # Result should have the new message
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1]['content'], 'New message')

    def test_multiple_append_operations(self):
        """Test multiple sequential append operations."""
        config = BaseInferenceModelConfig(model_name="test")
        model = self.ConcreteModel(config)
        
        # Build conversation step by step
        history = []
        history = model.append_with_chat_template(
            "Hello", role='user', chat_history=history
        )
        history = model.append_with_chat_template(
            "Hi there!", role='assistant', chat_history=history
        )
        history = model.append_with_chat_template(
            "How are you?", role='user', chat_history=history
        )
        
        expected = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'},
            {'role': 'user', 'content': 'How are you?'}
        ]
        
        self.assertEqual(history, expected)


class TestAbstractMethodEnforcement(unittest.TestCase):
    """Test that abstract methods are properly enforced."""

    def test_missing_convert_method_prevents_instantiation(self):
        """Test that missing convert method prevents instantiation."""
        class PartialModel(BaseInferenceModel):
            # Missing convert_chat_messages_to_custom_format implementation
            def some_other_method(self):
                pass
        
        config = BaseInferenceModelConfig(model_name="test")
        
        with self.assertRaises(TypeError) as context:
            PartialModel(config)
        
        # Check that the error mentions the missing abstract method
        error_msg = str(context.exception)
        self.assertIn("abstract", error_msg.lower())

    def test_all_methods_implemented_allows_instantiation(self):
        """Test that implementing all abstract methods allows instantiation."""
        class CompleteModel(BaseInferenceModel):
            def convert_chat_messages_to_custom_format(self, chat_messages):
                return "implemented"
        
        config = BaseInferenceModelConfig(model_name="test")
        
        # Should not raise any errors
        model = CompleteModel(config)
        self.assertIsInstance(model, CompleteModel)
        self.assertIsInstance(model, BaseInferenceModel)


if __name__ == "__main__":
    unittest.main()