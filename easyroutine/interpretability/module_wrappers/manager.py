from typing import Union, Type, Dict, List
import torch.nn as nn


from easyroutine.interpretability.module_wrappers.chameleon_attention import (
    ChameleonAttentionWrapper,
)
from easyroutine.interpretability.module_wrappers.llama_attention import (
    LlamaAttentionWrapper,
)
from easyroutine.interpretability.module_wrappers.mistral_attention import (
    MistralAttentionWrapper,
)
from easyroutine.interpretability.module_wrappers.T5_attention import T5AttentionWrapper
from easyroutine.interpretability.module_wrappers.gemma3_attention import (
    Gemma3AttentionWrapper,
)
from easyroutine.interpretability.module_wrappers.qwen2_attention import (
    Qwen2AttentionWrapper,
)
from easyroutine.logger import logger

from easyroutine.interpretability.utils import parse_module_path, find_all_modules


class AttentionWrapperFactory:
    """
    Maps a given model name to the correct attention wrapper class.
    """

    AVAILABLE_MODULE_WRAPPERS: dict = {
        ChameleonAttentionWrapper.original_name(): ChameleonAttentionWrapper,
        LlamaAttentionWrapper.original_name(): LlamaAttentionWrapper,
        T5AttentionWrapper.original_name(): T5AttentionWrapper,
        MistralAttentionWrapper.original_name(): MistralAttentionWrapper,
        Gemma3AttentionWrapper.original_name(): Gemma3AttentionWrapper,
        Qwen2AttentionWrapper.original_name(): Qwen2AttentionWrapper,
        
    }

    # MODEL_NAME_TO_WRAPPER = {
    #     "facebook/chameleon-7b": ChameleonAttentionWrapper,
    #     "facebook/chameleon-30b": ChameleonAttentionWrapper,
    #     "mistral-community/pixtral-12b": LlamaAttentionWrapper,
    #     "llava-hf/llava-v1.6-mistral-7b-hf": LlamaAttentionWrapper,
    #     "hf-internal-testing/tiny-random-LlamaForCausalLM": LlamaAttentionWrapper,
    #     "ChoereForAI/aya-101": T5AttentionWrapper,
    # }

    @staticmethod
    def get_wrapper_classes(
        model: nn.Module,
    ) -> Dict[
        str,
        Union[
            Type[ChameleonAttentionWrapper],
            Type[LlamaAttentionWrapper],
            Type[T5AttentionWrapper],
            Type[MistralAttentionWrapper],
            Type[Gemma3AttentionWrapper],
            Type[Qwen2AttentionWrapper]
        ],
    ]:
        """
        Returns a dictionary mapping module names to their corresponding wrapper classes
        for all supported modules found in the model.

        Args:
            model (nn.Module): The model to analyze

        Returns:
            Dict[str, Type]: Dictionary mapping original module names to wrapper classes
        """
        all_modules = find_all_modules(model, return_only_names=True)
        found_wrappers = {}

        for (
            candidate_name,
            candidate_wrapper,
        ) in AttentionWrapperFactory.AVAILABLE_MODULE_WRAPPERS.items():
            if candidate_name in all_modules:
                logger.info(f"Found a wrapper for {candidate_name}")
                found_wrappers[candidate_name] = candidate_wrapper

        if not found_wrappers:
            logger.warning(f"No wrappers found for any module in {model}")

        return found_wrappers


class ModuleWrapperManager:
    """
    Handles the logic of replacing original modules within a given model
    with custom wrappers. Supports multiple module types per model.
    Also allows restoring the original modules if needed.
    """

    def __init__(self, model: nn.Module, log_level: str = "INFO"):
        """
        Initializes the manager with a given model.

        Args:
            model (nn.Module): The model whose modules will be wrapped
            log_level (str): Logging level
        """
        # Get all available wrappers for modules in this model
        self.module_wrappers = AttentionWrapperFactory.get_wrapper_classes(model)

        # Dictionary to store {module_path: {module_type: original_module}}
        self.original_modules = {}

        # Store target module names for quick lookup
        self.target_module_names = list(self.module_wrappers.keys())

    def __contains__(self, module_name: str) -> bool:
        """
        Check if a module name is supported by this manager.

        Args:
            module_name (str): The name of the module to check

        Returns:
            bool: True if the module is supported, False otherwise
        """
        return module_name in self.target_module_names

    def substitute_attention_module(self, model: nn.Module) -> None:
        """
        Public method that performs the substitution of all supported modules in the model.
        Logs each replacement.

        Args:
            model (nn.Module): The model whose modules will be wrapped
        """
        self._traverse_and_modify(model, parent_path="", mode="substitute")

    def restore_original_attention_module(self, model: nn.Module) -> None:
        """
        Public method that restores the original modules in the model.
        Logs each restoration.

        Args:
            model (nn.Module): The model whose modules will be restored
        """
        self._traverse_and_modify(model, parent_path="", mode="restore")

    def _traverse_and_modify(
        self, module: nn.Module, parent_path: str, mode: str
    ) -> None:
        """
        Recursively traverses `module` and either substitutes or restores each matching
        submodule, depending on `mode`.

        - mode="substitute": Replaces original modules with wrappers
        - mode="restore": Replaces wrapper modules with their originals

        Args:
            module (nn.Module): The current module to inspect
            parent_path (str): A string that tracks the 'path' of this submodule in the overall model hierarchy
            mode (str): Either "substitute" or "restore"
        """
        for name, child in list(module.named_children()):
            # Identify the submodule path (e.g. "encoder.layer.0.attention")
            submodule_path = f"{parent_path}.{name}" if parent_path else name

            if mode == "substitute":
                # Look for any matching original module class names
                child_class_name = child.__class__.__name__
                if child_class_name in self.target_module_names:
                    # Get the appropriate wrapper for this module type
                    wrapper_class = self.module_wrappers[child_class_name]

                    # Store the original module
                    if submodule_path not in self.original_modules:
                        self.original_modules[submodule_path] = {}
                    self.original_modules[submodule_path][child_class_name] = child

                    # Wrap it
                    wrapped_module = wrapper_class(child)
                    setattr(module, name, wrapped_module)

                    logger.debug(
                        f"Substituted '{submodule_path}' with wrapper for {child_class_name}."
                    )
                else:
                    # Recurse
                    self._traverse_and_modify(child, submodule_path, mode="substitute")

            elif mode == "restore":
                # Check if this is any kind of wrapper we know about
                child_class_name = child.__class__.__name__
                for orig_name, wrapper_class in self.module_wrappers.items():
                    if child_class_name == wrapper_class.__name__:
                        # Found a wrapper, check if we have the original
                        if (
                            submodule_path in self.original_modules
                            and orig_name in self.original_modules[submodule_path]
                        ):
                            # Restore the original module
                            original_module = self.original_modules[submodule_path][
                                orig_name
                            ]
                            setattr(module, name, original_module)
                            logger.info(
                                f"Restored '{submodule_path}' to original {orig_name}."
                            )
                            break
                else:
                    # If not a wrapper or no original found, recurse
                    self._traverse_and_modify(child, submodule_path, mode="restore")
