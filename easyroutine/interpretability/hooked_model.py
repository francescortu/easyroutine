import torch
from transformers import GenerationConfig
from typing import Union, Literal, Optional, List, Dict, Callable, Any, Tuple

from easyroutine.interpretability.models import (
    ModelFactory,
    TokenizerFactory,
    InputHandler,
)
from easyroutine.interpretability.token_index import TokenIndex
from easyroutine.interpretability.activation_cache import ActivationCache
from easyroutine.interpretability.interventions import Intervention, InterventionManager
from easyroutine.interpretability.utils import get_attribute_by_name
from easyroutine.interpretability.module_wrappers.manager import ModuleWrapperManager
from easyroutine.logger import logger
from easyroutine.console import progress
from dataclasses import dataclass
# from easyroutine.interpretability.ablation import AblationManager

# from src.model.emu3.
from easyroutine.interpretability.utils import (
    map_token_to_pos,
    preprocess_patching_queries,
    logit_diff,
    get_attribute_from_name,
    kl_divergence_diff,
    conditional_no_grad,
)
from easyroutine.interpretability.hooks import (
    embed_hook,
    save_resid_hook,
    projected_value_vectors_head,
    avg_attention_pattern_head,
    attention_pattern_head,
    get_module_by_path,
    process_args_kwargs_output,
    query_key_value_hook,
    head_out_hook,
    layernom_hook,
    input_embedding_hook,
)

from functools import partial
import pandas as pd

import importlib.resources
import yaml



def load_config() -> dict:
    with importlib.resources.open_text(
        "easyroutine.interpretability.config", "config.yaml"
    ) as file:
        return yaml.safe_load(file)


yaml_config = load_config()

# to avoid running out of shared memory
# torch.multiprocessing.set_sharing_strategy("file_system")


@dataclass
class HookedModelConfig:
    """
    Configuration class for HookedModel initialization and behavior.

    This dataclass contains all the configuration parameters needed to initialize
    a HookedModel instance. It provides sensible defaults while allowing
    customization of model loading parameters, device configuration, and
    processing settings.

    Attributes:
        model_name (str): The identifier of the model to load. Can be:
            - A Hugging Face model repository name (e.g., "gpt2", "mistral-7b")
            - A local path to a model directory
            - Any model supported by transformers library
            
        device_map (Literal["balanced", "cuda", "cpu", "auto"], optional): 
            Device placement strategy for the model. Options:
            - "balanced": Distribute model across available GPUs evenly
            - "cuda": Place entire model on the first available GPU
            - "cpu": Place model on CPU (slower but uses less GPU memory)
            - "auto": Let transformers decide optimal placement
            Defaults to "balanced".
            
        torch_dtype (torch.dtype, optional): The data type for model parameters.
            Common options include torch.float16, torch.bfloat16, torch.float32.
            bfloat16 provides good balance of speed and stability.
            Defaults to torch.bfloat16.
            
        attn_implementation (Literal["eager", "custom_eager"], optional):
            The attention mechanism implementation to use:
            - "eager": Standard PyTorch attention implementation
            - "custom_eager": Enhanced implementation with better hook support
            The custom implementation is recommended for interpretability work
            as it provides more comprehensive hook coverage.
            Defaults to "custom_eager".
            
        batch_size (int, optional): The batch size for model inference.
            Currently, only batch size 1 is fully supported and tested.
            Using larger batch sizes may lead to unexpected behavior.
            Defaults to 1.

    Example:
        >>> config = HookedModelConfig(
        ...     model_name="gpt2",
        ...     device_map="auto",
        ...     torch_dtype=torch.float16
        ... )
        >>> model = HookedModel(config)
        
        >>> # Or use the convenience method
        >>> model = HookedModel.from_pretrained(
        ...     "gpt2", 
        ...     device_map="cuda",
        ...     torch_dtype=torch.bfloat16
        ... )

    Warning:
        Batch sizes greater than 1 are experimental and may not work correctly
        with all interpretability features. Use at your own risk.
    """

    model_name: str
    device_map: Literal["balanced", "cuda", "cpu", "auto"] = "balanced"
    torch_dtype: torch.dtype = torch.bfloat16
    attn_implementation: Literal["eager", "custom_eager"] = (
        "custom_eager"  # TODO: add flash_attention_2 in custom module to support it
    )
    batch_size: int = 1


@dataclass
class ExtractionConfig:
    """
    Configuration class for specifying which model activations to extract.

    This comprehensive configuration class allows fine-grained control over
    what internal activations and computations should be extracted from the
    model during inference. It supports extraction from various model components
    including residual streams, attention mechanisms, MLP layers, and more.

    The configuration follows a boolean flag pattern where each attribute
    specifies whether to extract a particular type of activation. This allows
    for flexible composition of extraction requirements based on research needs.

    Residual Stream Activations:
        extract_resid_in (bool): Extract activations flowing into residual connections.
            These are the inputs to each transformer layer before processing.
            Defaults to False.
            
        extract_resid_mid (bool): Extract intermediate activations within layers.
            These represent computational states between attention and MLP processing.
            Defaults to False.
            
        extract_resid_out (bool): Extract activations flowing out of residual connections.
            These are the final outputs of each transformer layer.
            Defaults to False.
            
        extract_resid_in_post_layernorm (bool): Extract residual inputs after layer
            normalization. Useful for studying the effect of normalization.
            Defaults to False.

    Attention Mechanism Activations:
        extract_attn_pattern (bool): Extract attention weight matrices showing
            which tokens attend to which other tokens. Essential for attention
            analysis and visualization. Defaults to False.
            
        extract_attn_out (bool): Extract attention layer outputs before residual
            connection. Shows the contribution of attention to each position.
            Defaults to False.
            
        extract_attn_in (bool): Extract attention layer inputs. Useful for
            studying how different inputs affect attention computations.
            Defaults to False.

    Attention Head Components:
        extract_head_values_projected (bool): Extract value vectors after
            projection in multi-head attention. Shows what information each
            head is passing forward. Defaults to False.
            
        extract_head_keys_projected (bool): Extract key vectors after projection.
            Combined with queries, determines attention patterns.
            Defaults to False.
            
        extract_head_queries_projected (bool): Extract query vectors after
            projection. Used with keys to compute attention weights.
            Defaults to False.
            
        extract_head_keys (bool): Extract raw key vectors before projection.
            Lower-level view of attention computation. Defaults to False.
            
        extract_head_values (bool): Extract raw value vectors before projection.
            Shows pre-projection value representations. Defaults to False.
            
        extract_head_queries (bool): Extract raw query vectors before projection.
            Shows pre-projection query representations. Defaults to False.
            
        extract_head_out (bool): [DEPRECATED] Extract head outputs.
            Use extract_attn_out instead. Defaults to False.

    Other Layer Components:
        extract_mlp_out (bool): Extract MLP (feed-forward) layer outputs.
            Shows the contribution of position-wise processing. Defaults to False.
            
        extract_embed (bool): Extract embedding layer outputs. Shows initial
            token representations before transformer processing. Defaults to False.
            
        extract_last_layernorm (bool): Extract final layer normalization outputs.
            Shows normalized representations before final predictions.
            Defaults to False.

    Metadata and Processing Options:
        save_input_ids (bool): Include input token IDs in the activation cache.
            Useful for mapping activations back to specific tokens.
            Defaults to False.
            
        save_logits (bool): Include model output logits in the cache.
            Essential for studying model predictions. Defaults to True.
            
        keep_gradient (bool): Preserve gradient information in extracted
            activations. Required for gradient-based analysis methods.
            Defaults to False.

    Aggregation Options:
        avg (bool): Compute average activations over specified target positions.
            Reduces memory usage when only summary statistics are needed.
            Defaults to False.
            
        avg_over_example (bool): Compute running average over multiple examples.
            Requires external cache management for accumulation. Defaults to False.

    Attention Analysis Options:
        attn_heads (Union[list[dict], Literal["all"]]): Specifies which attention
            heads to extract patterns from. Can be:
            - "all": Extract from all heads in all layers
            - List of dicts: Specific heads, e.g., [{"layer": 0, "head": 5}]
            Defaults to "all".
            
        attn_pattern_avg (Literal["mean", "sum", "baseline_ratio", "none"]):
            How to aggregate attention patterns. Options:
            - "mean": Average across specified dimensions
            - "sum": Sum across specified dimensions
            - "baseline_ratio": Ratio relative to baseline pattern
            - "none": No aggregation, return full patterns
            Defaults to "none".
            
        attn_pattern_row_positions (Optional[Union[List[int], List[Tuple], List[str], List[Union[int, Tuple, str]]]]):
            Specific row positions in attention patterns to extract.
            Can specify token positions, ranges, or special position names.
            If None, extracts full attention patterns. Defaults to None.

    Example:
        >>> # Basic residual stream extraction
        >>> config = ExtractionConfig(
        ...     extract_resid_out=True,
        ...     save_input_ids=True
        ... )
        
        >>> # Comprehensive attention analysis
        >>> config = ExtractionConfig(
        ...     extract_attn_pattern=True,
        ...     extract_head_values_projected=True,
        ...     attn_heads=[{"layer": 0, "head": 0}, {"layer": 1, "head": 3}],
        ...     attn_pattern_avg="mean"
        ... )
        
        >>> # Full model analysis
        >>> config = ExtractionConfig(
        ...     extract_resid_in=True,
        ...     extract_resid_out=True,
        ...     extract_attn_pattern=True,
        ...     extract_mlp_out=True,
        ...     save_logits=True,
        ...     save_input_ids=True
        ... )

    Note:
        Extracting many activation types can significantly increase memory usage
        and computation time. Enable only the activations needed for your analysis.
        The is_not_empty() method can be used to verify that at least one
        extraction option is enabled.
    """

    extract_embed: bool = False
    extract_resid_in: bool = False
    extract_resid_mid: bool = False
    extract_resid_out: bool = False
    extract_resid_in_post_layernorm: bool = False
    extract_attn_pattern: bool = False
    extract_head_values_projected: bool = False
    extract_head_keys_projected: bool = False
    extract_head_queries_projected: bool = False
    extract_head_keys: bool = False
    extract_head_values: bool = False
    extract_head_queries: bool = False
    extract_head_out: bool = False
    extract_attn_out: bool = False
    extract_attn_in: bool = False
    extract_mlp_out: bool = False
    extract_last_layernorm: bool = False
    save_input_ids: bool = False
    avg: bool = False
    avg_over_example: bool = False
    attn_heads: Union[list[dict], Literal["all"]] = "all"
    attn_pattern_avg: Literal["mean", "sum", "baseline_ratio", "none"] = "none"
    attn_pattern_row_positions: Optional[
        Union[List[int], List[Tuple], List[str], List[Union[int, Tuple, str]]]
    ] = None
    save_logits: bool = True
    keep_gradient: bool = False  # New flag

    def is_not_empty(self):
        """
        Check if any extraction options are enabled in this configuration.
        
        This method validates that at least one activation extraction flag is set
        to True, ensuring that the configuration will actually extract some data
        when used with a HookedModel. This is useful for validation before
        running expensive extraction operations.
        
        Returns:
            bool: True if at least one extraction option is enabled (any attribute
                is True), False if all extraction options are disabled.
        
        Example:
            >>> config = ExtractionConfig()  # All defaults (mostly False)
            >>> config.is_not_empty()
            True  # save_logits is True by default
            
            >>> config_empty = ExtractionConfig(save_logits=False)
            >>> config_empty.is_not_empty()
            False
            
            >>> config_active = ExtractionConfig(extract_resid_out=True)
            >>> config_active.is_not_empty()
            True
        
        Note:
            This method checks all boolean extraction flags. It's recommended
            to call this before expensive extraction operations to avoid
            unnecessary computation when no activations would be extracted.
        """
        return any(
            [
                self.extract_resid_in,
                self.extract_resid_mid,
                self.extract_resid_out,
                self.extract_attn_pattern,
                self.extract_head_values_projected,
                self.extract_head_keys_projected,
                self.extract_head_queries_projected,
                self.extract_head_keys,
                self.extract_head_values,
                self.extract_head_queries,
                self.extract_head_out,
                self.extract_attn_out,
                self.extract_attn_in,
                self.extract_mlp_out,
                self.save_input_ids,
                self.avg,
                self.avg_over_example,
            ]
        )

    def to_dict(self):
        return self.__dict__


class HookedModel:
    """
    A comprehensive wrapper around Hugging Face transformer models for mechanistic interpretability.
    
    This class provides advanced functionality for extracting internal activations from transformer
    models and performing mechanistic interpretability methods such as ablation studies, activation
    patching, and intervention analysis. It supports both language models and vision-language models
    with automatic module detection and custom attention implementations.
    
    The HookedModel class serves as the primary interface for interpretability research, offering:
    - Automatic model loading and configuration
    - Activation extraction from any model component
    - Support for intervention and ablation studies
    - Custom attention implementations for better hook support
    - Batch processing capabilities
    - Multi-device support
    
    Key Features:
        - Extract activations from residual streams, attention layers, MLP layers
        - Support for attention pattern analysis and head-specific extractions
        - Intervention capabilities for causal analysis
        - Automatic tokenizer and processor handling
        - Support for vision-language models with image processing
        - Custom eager attention implementation for comprehensive hook support
    
    Attributes:
        config (HookedModelConfig): Configuration object containing model settings
        hf_model: The underlying Hugging Face model
        hf_language_model: The language model component (for vision-language models)
        model_config: Internal model configuration for hook management
        hf_tokenizer: The model's tokenizer
        processor: Optional processor for vision-language models
        text_tokenizer: Text tokenizer component
        input_handler: Handles input preprocessing based on model type
        module_wrapper_manager: Manages custom module wrappers
        intervention_manager: Handles intervention operations
        
    Example:
        >>> # Basic model loading
        >>> model = HookedModel.from_pretrained("gpt2")
        >>> 
        >>> # Extract activations
        >>> cache = model.extract_cache(
        ...     inputs,
        ...     target_token_positions=["last"],
        ...     extraction_config=ExtractionConfig(extract_resid_out=True)
        ... )
        >>> 
        >>> # Perform interventions
        >>> result = model.run_with_interventions(inputs, interventions)
    
    Note:
        The model uses custom eager attention implementation by default to ensure
        comprehensive hook support. This can be disabled by setting 
        attn_implementation="eager" in the configuration.
    """

    def __init__(self, config: HookedModelConfig, log_file_path: Optional[str] = None):
        self.config = config
        self.hf_model, self.hf_language_model, self.model_config = (
            ModelFactory.load_model(
                model_name=config.model_name,
                device_map=config.device_map,
                torch_dtype=config.torch_dtype,
                attn_implementation="eager"
                if config.attn_implementation == "custom_eager"
                else config.attn_implementation,
            )
        )
        self.hf_model.eval()
        self.base_model = None
        self.module_wrapper_manager = ModuleWrapperManager(model=self.hf_model)
        self.intervention_manager = InterventionManager(model_config=self.model_config)

        tokenizer, processor = TokenizerFactory.load_tokenizer(
            model_name=config.model_name,
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
        )
        self.hf_tokenizer = tokenizer
        self.input_handler = InputHandler(model_name=config.model_name)
        if processor is True:
            self.processor = tokenizer
            self.text_tokenizer = self.processor.tokenizer  # type: ignore
        else:
            self.processor = None
            self.text_tokenizer = tokenizer

        self.first_device = next(self.hf_model.parameters()).device
        device_num = torch.cuda.device_count()
        logger.info(
            f"HookedModel: Model loaded in {device_num} devices. First device: {self.first_device}"
        )
        self.act_type_to_hook_name = {
            "resid_in": self.model_config.residual_stream_input_hook_name,
            "resid_out": self.model_config.residual_stream_hook_name,
            "resid_mid": self.model_config.intermediate_stream_hook_name,
            "attn_out": self.model_config.attn_out_hook_name,
            "attn_in": self.model_config.attn_in_hook_name,
            "values": self.model_config.head_value_hook_name,
            # Add other act_types if needed
        }
        self.additional_hooks = []
        self.additional_interventions = []
        self.assert_all_modules_exist()

        self.image_placeholder = yaml_config["tokenizer_placeholder"][config.model_name]

        if self.config.attn_implementation == "custom_eager":
            logger.info(
                """ HookedModel:
                            The model is using the custom eager attention implementation that support attention matrix hooks because I get config.attn_impelemntation == 'custom_eager'. If you don't want this, you can call HookedModel.restore_original_modules. 
                            However, we reccomend using this implementation since the base one do not contains attention matrix hook resulting in unexpected behaviours. 
                            """,
            )
            self.set_custom_modules()

    def __repr__(self):
        return f"""HookedModel(model_name={self.config.model_name}):
        {self.hf_model.__repr__()}
    """

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        """
        Create a HookedModel instance from a pretrained model.
        
        This convenience class method provides a simple interface for loading
        pretrained models without explicitly creating a HookedModelConfig object.
        It automatically constructs the configuration with the provided parameters
        and initializes the HookedModel.
        
        Args:
            model_name (str): The identifier of the pretrained model to load.
                Can be:
                - A Hugging Face model repository name (e.g., "gpt2", "mistral-7b")
                - A local path to a model directory
                - Any model supported by the transformers library
            **kwargs: Additional keyword arguments to pass to HookedModelConfig.
                Common options include:
                - device_map: Device placement strategy ("auto", "cuda", "cpu", "balanced")
                - torch_dtype: Data type for model parameters (torch.float16, torch.bfloat16, etc.)
                - attn_implementation: Attention implementation ("eager", "custom_eager")
                - batch_size: Batch size for inference (default: 1)
        
        Returns:
            HookedModel: A fully initialized HookedModel instance ready for
                interpretability analysis.
        
        Example:
            >>> # Basic model loading
            >>> model = HookedModel.from_pretrained("gpt2")
            
            >>> # With custom configuration
            >>> model = HookedModel.from_pretrained(
            ...     "mistral-7b",
            ...     device_map="auto",
            ...     torch_dtype=torch.bfloat16,
            ...     attn_implementation="custom_eager"
            ... )
            
            >>> # Local model loading
            >>> model = HookedModel.from_pretrained(
            ...     "/path/to/local/model",
            ...     device_map="cuda"
            ... )
        
        Note:
            This method is equivalent to:
            ```python
            config = HookedModelConfig(model_name=model_name, **kwargs)
            model = HookedModel(config)
            ```
        """

    def assert_module_exists(self, component: str):
        # Remove '.input' or '.output' from the component
        component = component.replace(".input", "").replace(".output", "")

        # Check if '{}' is in the component, indicating layer indexing
        if "{}" in component:
            for i in range(0, self.model_config.num_hidden_layers):
                attr_name = component.format(i)

                try:
                    get_attribute_by_name(self.hf_model, attr_name)
                except AttributeError:
                    try:
                        if attr_name in self.module_wrapper_manager:
                            self.set_custom_modules()
                            get_attribute_by_name(self.hf_model, attr_name)
                            self.restore_original_modules()
                    except AttributeError:
                        raise ValueError(
                            f"Component '{attr_name}' does not exist in the model. Please check the model configuration."
                        )
        else:
            try:
                get_attribute_by_name(self.hf_model, component)
            except AttributeError:
                raise ValueError(
                    f"Component '{component}' does not exist in the model. Please check the model configuration."
                )

    def assert_all_modules_exist(self):
        # get the list of all attributes of model_config
        all_attributes = [attr_name for attr_name in self.model_config.__dict__.keys()]
        # save just the attributes that have "hook" in the name
        hook_attributes = [
            attr_name for attr_name in all_attributes if "hook" in attr_name
        ]
        for hook_attribute in hook_attributes:
            self.assert_module_exists(getattr(self.model_config, hook_attribute))

    def set_custom_modules(self):
        """
        Apply the wrap of the custom modules. for now just the attention module
        """
        logger.info("HookedModel: Setting custom modules.")
        self.module_wrapper_manager.substitute_attention_module(self.hf_model)

    def restore_original_modules(self):
        """
        Restore the original modules of the model unloading the custom modules.
        """
        logger.info("HookedModel: Restoring original modules.")
        self.module_wrapper_manager.restore_original_attention_module(self.hf_model)

    def is_multimodal(self) -> bool:
        """
        Get if the model is multimodal or not
        """
        if self.processor is not None:
            return True
        return False

    def use_full_model(self):
        if self.processor is not None:
            logger.debug("HookedModel: Using full model capabilities")
            if self.base_model is not None:
                self.hf_model = self.base_model
                self.model_config.restore_full_model()
                self.base_model = None
        else:
            if self.base_model is not None:
                self.hf_model = self.base_model
            logger.debug("HookedModel: Using full text only model capabilities")

    def use_language_model_only(self):
        if self.hf_language_model is None:
            logger.warning(
                "HookedModel: The model does not have a separate language model that can be used",
            )
        else:
            # check if we are already using the language model
            if self.hf_model == self.hf_language_model:
                return
            self.base_model = self.hf_model
            self.hf_model = self.hf_language_model
            self.model_config.use_language_model()
            logger.debug("HookedModel: Using only language model capabilities")

    def get_tokenizer(self):
        """
        Get the primary tokenizer associated with this model.
        
        Returns the tokenizer that was loaded during model initialization.
        For vision-language models, this may be a processor that includes
        both text tokenization and image processing capabilities.
        
        Returns:
            Union[transformers.PreTrainedTokenizer, transformers.ProcessorMixin]:
                The tokenizer or processor associated with the model.
        
        Example:
            >>> model = HookedModel.from_pretrained("gpt2")
            >>> tokenizer = model.get_tokenizer()
            >>> tokens = tokenizer("Hello world", return_tensors="pt")
        
        Note:
            For text-only models, this returns a standard tokenizer.
            For vision-language models, this may return a processor that
            handles both text and image inputs. Use get_text_tokenizer()
            if you specifically need the text tokenization component.
        """

    def get_text_tokenizer(self):
        """
        Get the text tokenization component of the model's tokenizer.
        
        For vision-language models that use a processor (which combines text
        tokenization and image processing), this method extracts and returns
        just the text tokenizer component. For text-only models, this returns
        the same tokenizer as get_tokenizer().
        
        Returns:
            transformers.PreTrainedTokenizer: The text tokenizer component.
        
        Raises:
            ValueError: If the model uses a processor that doesn't have a
                tokenizer attribute.
        
        Example:
            >>> # For a vision-language model
            >>> model = HookedModel.from_pretrained("llava-v1.6-mistral-7b-hf")
            >>> text_tokenizer = model.get_text_tokenizer()
            >>> tokens = text_tokenizer("Hello world", return_tensors="pt")
            
            >>> # For a text-only model (same as get_tokenizer())
            >>> model = HookedModel.from_pretrained("gpt2")
            >>> text_tokenizer = model.get_text_tokenizer()
        
        Note:
            This method is particularly useful when you need to perform
            text-specific operations on vision-language models where the
            primary tokenizer is actually a multimodal processor.
        """
        if self.processor is not None:
            if not hasattr(self.processor, "tokenizer"):
                raise ValueError("The processor does not have a tokenizer")
            return self.processor.tokenizer  # type: ignore
        return self.hf_tokenizer

    def get_processor(self):
        r"""
        Return the processor of the model (None if the model does not have a processor, i.e. text only model)

        Args:
            None

        Returns:
            processor: the processor of the model
        """
        if self.processor is None:
            raise ValueError("The model does not have a processor")
        return self.processor

    def get_lm_head(self):
        return get_attribute_by_name(self.hf_model, self.model_config.unembed_matrix)

    def get_last_layernorm(self):
        return get_attribute_by_name(self.hf_model, self.model_config.last_layernorm)

    def get_image_placeholder(self) -> str:
        return self.image_placeholder

    def eval(self):
        r"""
        Set the model in evaluation mode
        """
        self.hf_model.eval()

    def device(self):
        r"""
        Return the device of the model. If the model is in multiple devices, it will return the first device

        Args:
            None

        Returns:
            device: the device of the model
        """
        return self.first_device

    def register_forward_hook(self, component: str, hook_function: Callable):
        r"""
        Add a new hook to the model. The hook will be called in the forward pass of the model.

        Args:
            component (str): the component of the model where the hook will be added.
            hook_function (Callable): the function that will be called in the forward pass of the model. The function must have the following signature:
                def hook_function(module, input, output):
                    pass

        Returns:
            None

        Examples:
            >>> def hook_function(module, input, output):
            >>>     # your code here
            >>>     pass
            >>> model.register_forward_hook("model.layers[0].self_attn", hook_function)
        """
        self.additional_hooks.append(
            {
                "component": component,
                "intervention": hook_function,
            }
        )

    def to_string_tokens(
        self,
        tokens: Union[list, torch.Tensor],
    ):
        r"""
        Transform a list or a tensor of tokens in a list of string tokens.

        Args:
            tokens (Union[list, torch.Tensor]): the tokens to transform in string tokens

        Returns:
            string_tokens (list): the list of string tokens

        Examples:
            >>> tokens = [101, 1234, 1235, 102]
            >>> model.to_string_tokens(tokens)
            ['[CLS]', 'hello', 'world', '[SEP]']
        """
        if isinstance(tokens, torch.Tensor):
            if tokens.dim() == 1:
                tokens = tokens.tolist()
            else:
                tokens = tokens.squeeze().tolist()
        string_tokens = []
        for tok in tokens:
            string_tokens.append(self.hf_tokenizer.decode(tok))  # type: ignore
        return string_tokens

    def register_interventions(self, interventions: List[Intervention]):
        self.additional_interventions = interventions
        logger.debug(f"HookedModel: Registered {len(interventions)} interventions")

    def clean_interventions(self):
        self.additional_interventions = []
        logger.debug(
            f"HookedModel: Removed {len(self.additional_interventions)} interventions"
        )

    def create_hooks(
        self,
        inputs,
        cache: ActivationCache,
        token_indexes: List,
        token_dict: Dict,
        # string_tokens: List[str],
        extraction_config: ExtractionConfig = ExtractionConfig(),
        interventions: Optional[List[Intervention]] = None,
        batch_idx: Optional[int] = None,
        external_cache: Optional[ActivationCache] = None,
    ):
        r"""
        Create the hooks to extract the activations of the model. The hooks will be added to the model and will be called in the forward pass of the model.

        Arguments:
            inputs (dict): dictionary with the inputs of the model (input_ids, attention_mask, pixel_values ...)
            cache (ActivationCache): dictionary where the activations of the model will be saved
            token_indexes (list[str]): list of tokens to extract the activations from (["last", "end-image", "start-image", "first"])
            token_dict (Dict): dictionary with the token indexes
            extraction_config (ExtractionConfig): configuration of the extraction of the activations of the model (default = ExtractionConfig())
            interventions (Optional[List[Intervention]]): list of interventions to perform during forward pass
            batch_idx (Optional[int]): index of the batch in the dataloader
            external_cache (Optional[ActivationCache]): external cache to use in the forward pass

        Returns:
            hooks (list[dict]): list of dictionaries with the component and the intervention to perform in the forward pass of the model
        """
        hooks = []

        # compute layer and head indexes
        if (
            isinstance(extraction_config.attn_heads, str)
            and extraction_config.attn_heads == "all"
        ):
            layer_indexes = [i for i in range(0, self.model_config.num_hidden_layers)]
            head_indexes = ["all"] * len(layer_indexes)
        elif isinstance(extraction_config.attn_heads, list):
            layer_head_indexes = [
                (el["layer"], el["head"]) for el in extraction_config.attn_heads
            ]
            layer_indexes = [el[0] for el in layer_head_indexes]
            head_indexes = [el[1] for el in layer_head_indexes]
        else:
            raise ValueError(
                "attn_heads must be 'all' or a list of dictionaries as [{'layer': 0, 'head': 0}]"
            )

        if extraction_config.extract_resid_out:
            # assert that the component exists in the model
            hooks += [
                {
                    "component": self.model_config.residual_stream_hook_name.format(i),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"resid_out_{i}",
                        token_indexes=token_indexes,
                        avg=extraction_config.avg,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]
        if extraction_config.extract_resid_in:
            # assert that the component exists in the model
            hooks += [
                {
                    "component": self.model_config.residual_stream_input_hook_name.format(
                        i
                    ),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"resid_in_{i}",
                        token_indexes=token_indexes,
                        avg=extraction_config.avg,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]

        if extraction_config.extract_resid_in_post_layernorm:
            hooks += [
                {
                    "component": self.model_config.residual_stream_input_post_layernorm_hook_name.format(
                        i
                    ),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"resid_in_post_layernorm_{i}",
                        token_indexes=token_indexes,
                        avg=extraction_config.avg,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]

        if extraction_config.save_input_ids:
            hooks += [
                {
                    "component": self.model_config.embed_tokens,
                    "intervention": partial(
                        embed_hook,
                        token_indexes=token_indexes,
                        cache=cache,
                        cache_key="input_ids",
                    ),
                }
            ]

        if extraction_config.extract_embed:  # New block
            hooks += [
                {
                    "component": self.model_config.embed_tokens,  # Use the embedding module name directly
                    "intervention": partial(
                        input_embedding_hook,
                        cache=cache,
                        cache_key="input_embeddings",
                        token_indexes=token_indexes,
                        keep_gradient=extraction_config.keep_gradient,
                        avg=extraction_config.avg,
                    ),
                }
            ]

        if extraction_config.extract_head_queries:
            hooks += [
                {
                    "component": self.model_config.head_query_hook_name.format(i),
                    "intervention": partial(
                        query_key_value_hook,
                        cache=cache,
                        cache_key="queries_",
                        token_indexes=token_indexes,
                        head_dim=self.model_config.head_dim,
                        avg=extraction_config.avg,
                        layer=i,
                        head=head,
                        num_key_value_groups=self.model_config.num_key_value_groups,
                    ),
                }
                for i, head in zip(layer_indexes, head_indexes)
            ]

        if extraction_config.extract_head_values:
            hooks += [
                {
                    "component": self.model_config.head_value_hook_name.format(i),
                    "intervention": partial(
                        query_key_value_hook,
                        cache=cache,
                        cache_key="values_",
                        token_indexes=token_indexes,
                        head_dim=self.model_config.head_dim,
                        avg=extraction_config.avg,
                        layer=i,
                        head=head,
                        num_key_value_groups=self.model_config.num_key_value_groups,
                    ),
                }
                for i, head in zip(layer_indexes, head_indexes)
            ]

        if extraction_config.extract_head_keys:
            hooks += [
                {
                    "component": self.model_config.head_key_hook_name.format(i),
                    "intervention": partial(
                        query_key_value_hook,
                        cache=cache,
                        cache_key="keys_",
                        token_indexes=token_indexes,
                        head_dim=self.model_config.head_dim,
                        avg=extraction_config.avg,
                        layer=i,
                        head=head,
                        num_key_value_groups=self.model_config.num_key_value_groups,
                    ),
                }
                for i, head in zip(layer_indexes, head_indexes)
            ]

        if extraction_config.extract_head_out:
            hooks += [
                {
                    "component": self.model_config.attn_o_proj_input_hook_name.format(
                        i
                    ),
                    "intervention": partial(
                        head_out_hook,
                        cache=cache,
                        cache_key="head_out_",
                        token_indexes=token_indexes,
                        avg=extraction_config.avg,
                        layer=i,
                        head=head,
                        num_heads=self.model_config.num_attention_heads,
                        head_dim=self.model_config.head_dim,
                        o_proj_weight=get_attribute_from_name(
                            self.hf_model,
                            self.model_config.attn_out_proj_weight.format(i),
                        ),
                        o_proj_bias=get_attribute_from_name(
                            self.hf_model,
                            self.model_config.attn_out_proj_bias.format(i),
                        ),
                    ),
                }
                for i, head in zip(layer_indexes, head_indexes)
            ]

        if extraction_config.extract_attn_in:
            hooks += [
                {
                    "component": self.model_config.attn_in_hook_name.format(i),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"attn_in_{i}",
                        token_indexes=token_indexes,
                        avg=extraction_config.avg,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]

        if extraction_config.extract_attn_out:
            hooks += [
                {
                    "component": self.model_config.attn_out_hook_name.format(i),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"attn_out_{i}",
                        token_indexes=token_indexes,
                        avg=extraction_config.avg,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]

        # if extraction_config.extract_avg:
        #     # Define a hook that saves the activations of the residual stream
        #     raise NotImplementedError(
        #         "The hook for the average is not working with token_index as a list"
        #     )

        #     # hooks.extend(
        #     #     [
        #     #         {
        #     #             "component": self.model_config.residual_stream_hook_name.format(
        #     #                 i
        #     #             ),
        #     #             "intervention": partial(
        #     #                 avg_hook,
        #     #                 cache=cache,
        #     #                 cache_key="resid_avg_{}".format(i),
        #     #                 last_image_idx=last_image_idxs, #type
        #     #                 end_image_idx=end_image_idxs,
        #     #             ),
        #     #         }
        #     #         for i in range(0, self.model_config.num_hidden_layers)
        #     #     ]
        #     # )
        if extraction_config.extract_resid_mid:
            hooks += [
                {
                    "component": self.model_config.intermediate_stream_hook_name.format(
                        i
                    ),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"resid_mid_{i}",
                        token_indexes=token_indexes,
                        avg=extraction_config.avg,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]

            # if we want to extract the output of the heads
        if extraction_config.extract_mlp_out:
            hooks += [
                {
                    "component": self.model_config.mlp_out_hook_name.format(i),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"mlp_out_{i}",
                        token_indexes=token_indexes,
                        avg=extraction_config.avg,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]

        if extraction_config.extract_last_layernorm:
            hooks += [
                {
                    "component": self.model_config.last_layernorm_hook_name,
                    "intervention": partial(
                        layernom_hook,
                        cache=cache,
                        cache_key="last_layernorm",
                        token_indexes=token_indexes,
                        avg=extraction_config.avg,
                    ),
                }
            ]
        # ABLATION AND PATCHING
        if interventions is not None:
            hooks += self.intervention_manager.create_intervention_hooks(
                interventions=interventions, token_dict=token_dict
            )
        if self.additional_interventions is not None:
            hooks += self.intervention_manager.create_intervention_hooks(
                interventions=self.additional_interventions, token_dict=token_dict
            )
        if extraction_config.extract_head_values_projected:
            hooks += [
                {
                    "component": self.model_config.head_value_hook_name.format(i),
                    "intervention": partial(
                        projected_value_vectors_head,
                        cache=cache,
                        token_indexes=token_indexes,
                        layer=i,
                        num_attention_heads=self.model_config.num_attention_heads,
                        num_key_value_heads=self.model_config.num_key_value_heads,
                        hidden_size=self.model_config.hidden_size,
                        d_head=self.model_config.head_dim,
                        out_proj_weight=get_attribute_from_name(
                            self.hf_model,
                            f"{self.model_config.attn_out_proj_weight.format(i)}",
                        ),
                        out_proj_bias=get_attribute_from_name(
                            self.hf_model,
                            f"{self.model_config.attn_out_proj_bias.format(i)}",
                        ),
                        head=head,
                        avg=extraction_config.avg,
                    ),
                }
                for i, head in zip(layer_indexes, head_indexes)
            ]

        if extraction_config.extract_head_keys_projected:
            hooks += [
                {
                    "component": self.model_config.head_key_hook_name.format(i),
                    "intervention": partial(
                        projected_key_vectors_head,
                        cache=cache,
                        token_indexes=token_indexes,
                        layer=i,
                        num_attention_heads=self.model_config.num_attention_heads,
                        num_key_value_heads=self.model_config.num_key_value_heads,
                        hidden_size=self.model_config.hidden_size,
                        d_head=self.model_config.head_dim,
                        out_proj_weight=get_attribute_from_name(
                            self.hf_model,
                            f"{self.model_config.attn_out_proj_weight.format(i)}",
                        ),
                        out_proj_bias=get_attribute_from_name(
                            self.hf_model,
                            f"{self.model_config.attn_out_proj_bias.format(i)}",
                        ),
                        head=head,
                        avg=extraction_config.avg,
                    ),
                }
                for i, head in zip(layer_indexes, head_indexes)
            ]

        if extraction_config.extract_head_queries_projected:
            hooks += [
                {
                    "component": self.model_config.head_query_hook_name.format(i),
                    "intervention": partial(
                        projected_query_vectors_head,
                        cache=cache,
                        token_indexes=token_indexes,
                        layer=i,
                        num_attention_heads=self.model_config.num_attention_heads,
                        num_key_value_heads=self.model_config.num_key_value_heads,
                        hidden_size=self.model_config.hidden_size,
                        d_head=self.model_config.head_dim,
                        out_proj_weight=get_attribute_from_name(
                            self.hf_model,
                            f"{self.model_config.attn_out_proj_weight.format(i)}",
                        ),
                        out_proj_bias=get_attribute_from_name(
                            self.hf_model,
                            f"{self.model_config.attn_out_proj_bias.format(i)}",
                        ),
                        head=head,
                        avg=extraction_config.avg,
                    ),
                }
                for i, head in zip(layer_indexes, head_indexes)
            ]

        if extraction_config.extract_attn_pattern:
            if extraction_config.avg_over_example:
                if external_cache is None:
                    logger.warning(
                        """The external_cache is None. The average could not be computed since missing an external cache where store the iterations.
                        """
                    )
                elif batch_idx is None:
                    logger.warning(
                        """The batch_idx is None. The average could not be computed since missing the batch index.
                       
                        """
                    )
                else:
                    # move the cache to the same device of the model
                    external_cache.to(self.first_device)
                    hooks += [
                        {
                            "component": self.model_config.attn_matrix_hook_name.format(
                                i
                            ),
                            "intervention": partial(
                                avg_attention_pattern_head,
                                token_indexes=token_indexes,
                                layer=i,
                                attn_pattern_current_avg=external_cache,
                                batch_idx=batch_idx,
                                cache=cache,
                                # avg=extraction_config.avg,
                                extract_avg_value=extraction_config.extract_head_values_projected,
                            ),
                        }
                        for i in range(0, self.model_config.num_hidden_layers)
                    ]
            else:
                hooks += [
                    {
                        "component": self.model_config.attn_matrix_hook_name.format(i),
                        "intervention": partial(
                            attention_pattern_head,
                            token_indexes=token_indexes,
                            cache=cache,
                            layer=i,
                            head=head,
                            attn_pattern_avg=extraction_config.attn_pattern_avg,
                            attn_pattern_row_partition=None
                            if extraction_config.attn_pattern_row_positions is None
                            else tuple(token_dict["attn_pattern_row_positions"]),
                        ),
                    }
                    for i, head in zip(layer_indexes, head_indexes)
                ]

            # if additional hooks are not empty, add them to the hooks list
        if self.additional_hooks:
            for hook in self.additional_hooks:
                hook["intervention"] = partial(
                    hook["intervention"],
                    cache=cache,
                    token_indexes=token_indexes,
                    token_dict=token_dict,
                    **hook["intervention"],
                )
                hooks.append(hook)
        return hooks

    @conditional_no_grad()
    # @torch.no_grad()
    def forward(
        self,
        inputs,
        target_token_positions: Union[
            List[Union[str, int, Tuple[int, int]]],
            List[str],
            List[int],
            List[Tuple[int, int]],
        ] = ["all"],
        pivot_positions: Optional[List[int]] = None,
        extraction_config: ExtractionConfig = ExtractionConfig(),
        interventions: Optional[List[Intervention]] = None,
        external_cache: Optional[ActivationCache] = None,
        # attn_heads: Union[list[dict], Literal["all"]] = "all",
        batch_idx: Optional[int] = None,
        move_to_cpu: bool = False,
        vocabulary_index: Optional[int] = None,
        **kwargs,
    ) -> ActivationCache:
        r"""
        Forward pass of the model. It will extract the activations of the model and save them in the cache. It will also perform ablation and patching if needed.

        Args:
            inputs (dict): dictionary with the inputs of the model (input_ids, attention_mask, pixel_values ...)
            target_token_positions (Union[Union[str, int, Tuple[int, int]], List[Union[str, int, Tuple[int, int]]]]): tokens to extract the activations from (["last", "end-image", "start-image", "first", -1, (2,10)]). See TokenIndex.get_token_index for more details
            pivot_positions (Optional[list[int]]): list of split positions of the tokens
            extraction_config (ExtractionConfig): configuration of the extraction of the activations of the model
            ablation_queries (Optional[pd.DataFrame | None]): dataframe with the ablation queries to perform during forward pass
            patching_queries (Optional[pd.DataFrame | None]): dataframe with the patching queries to perform during forward pass
            external_cache (Optional[ActivationCache]): external cache to use in the forward pass
            attn_heads (Union[list[dict], Literal["all"]]): list of dictionaries with the layer and head to extract the attention pattern or 'all' to
            batch_idx (Optional[int]): index of the batch in the dataloader
            move_to_cpu (bool): if True, move the activations to the cpu

        Returns:
            cache (ActivationCache): dictionary with the activations of the model

        Examples:
            >>> inputs = {"input_ids": torch.tensor([[101, 1234, 1235, 102]]), "attention_mask": torch.tensor([[1, 1, 1, 1]])}
            >>> model.forward(inputs, target_token_positions=["last"], extract_resid_out=True)
            {'resid_out_0': tensor([[[0.1, 0.2, 0.3, 0.4]]], grad_fn=<CopyBackwards>), 'input_ids': tensor([[101, 1234, 1235, 102]]), 'mapping_index': {'last': [0]}}
        """

        if target_token_positions is None and extraction_config.is_not_empty():
            raise ValueError(
                "target_token_positions must be passed if we want to extract the activations of the model"
            )

        cache = ActivationCache()
        string_tokens = self.to_string_tokens(
            self.input_handler.get_input_ids(inputs).squeeze()
        )
        token_index_finder = TokenIndex(
            self.config.model_name, pivot_positions=pivot_positions
        )
        token_indexes, token_dict = token_index_finder.get_token_index(
            tokens=target_token_positions,
            string_tokens=string_tokens,
            return_type="all",
        )
        if extraction_config.attn_pattern_row_positions is not None:
            token_row_indexes, _ = token_index_finder.get_token_index(
                tokens=extraction_config.attn_pattern_row_positions,
                string_tokens=string_tokens,
                return_type="all",
            )
            token_dict["attn_pattern_row_positions"] = token_row_indexes

        assert isinstance(token_indexes, list), "Token index must be a list"
        assert isinstance(token_dict, dict), "Token dict must be a dict"

        hooks = self.create_hooks(  # TODO: add **kwargs
            inputs=inputs,
            token_dict=token_dict,
            token_indexes=token_indexes,
            cache=cache,
            extraction_config=extraction_config,
            interventions=interventions,
            batch_idx=batch_idx,
            external_cache=external_cache,
        )

        hook_handlers = self.set_hooks(hooks)
        inputs = self.input_handler.prepare_inputs(
            inputs, self.first_device, self.config.torch_dtype
        )
        # forward pass
        output = self.hf_model(
            **inputs,
            # output_original_output=True,
            # output_attentions=extract_attn_pattern,
        )

        # save the logit of the target_token_positions
        flatten_target_token_positions = [
            item for sublist in token_indexes for item in sublist
        ]
        if extraction_config.save_logits:
            cache["logits"] = output.logits[:, flatten_target_token_positions, :]
        # since attention_patterns are returned in the output, we need to adapt to the cache structure
        if move_to_cpu:
            cache.cpu()
            if external_cache is not None:
                external_cache.cpu()

        stored_token_dict = {}
        mapping_index = {}
        current_index = 0

        for token in target_token_positions:
            mapping_index[token] = []
            if isinstance(token_dict, int):
                mapping_index[token].append(current_index)
                stored_token_dict[token] = token_dict
                current_index += 1
            elif isinstance(token_dict, dict):
                stored_token_dict[token] = token_dict[token]
                for idx in range(len(token_dict[token])):
                    mapping_index[token].append(current_index)
                    current_index += 1
            elif isinstance(token_dict, list):
                stored_token_dict[token] = token_dict
                for idx in range(len(token_dict)):
                    mapping_index[token].append(current_index)
                    current_index += 1
            else:
                raise ValueError("Token dict must be an int, a dict or a list")
        # update the mapping index in the cache if avg
        if extraction_config.avg:
            for i, token in enumerate(target_token_positions):
                mapping_index[token] = [i]
            mapping_index["info"] = "avg"
        cache["mapping_index"] = mapping_index
        cache["token_dict"] = stored_token_dict
        self.remove_hooks(hook_handlers)

        if extraction_config.keep_gradient:
            assert vocabulary_index is not None, (
                "dict_token_index must be provided if extract_input_embeddings_for_grad is True"
            )
            self._compute_input_gradients(cache, output.logits, vocabulary_index)

        return cache

    def __call__(self, *args, **kwds) -> ActivationCache:
        r"""
        Call the forward method of the model
        """
        return self.forward(*args, **kwds)

    def predict(self, k=10, strip: bool = True, **kwargs):
        out = self.forward(**kwargs)
        logits = out["logits"][:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        probs = probs.squeeze()
        topk = torch.topk(probs, k)
        # return a dictionary with the topk tokens and their probabilities
        string_tokens = self.to_string_tokens(topk.indices)
        token_probs = {}
        for token, prob in zip(string_tokens, topk.values):
            if strip:
                token = token.strip()
            if token not in token_probs:
                token_probs[token] = prob.item()
        return token_probs
        # return {
        #     token: prob.item() for token, prob in zip(string_tokens, topk.values)
        # }

    def get_module_from_string(self, component: str):
        r"""
        Return a module from the model given the string of the module.

        Args:
            component (str): the string of the module

        Returns:
            module (torch.nn.Module): the module of the model

        Examples:
            >>> model.get_module_from_string("model.layers[0].self_attn")
            BertAttention(...)
        """
        return self.hf_model.retrieve_modules_from_names(component)

    def set_hooks(self, hooks: List[Dict[str, Any]]):
        r"""
        Set the hooks in the model

        Args:
            hooks (list[dict]): list of dictionaries with the component and the intervention to perform in the forward pass of the model

        Returns:
            hook_handlers (list): list of hook handlers
        """

        if len(hooks) == 0:
            return []

        hook_handlers = []
        for hook in hooks:
            component = hook["component"]
            hook_function = hook["intervention"]

            # get the last module string (.input or .output) and remove it from the component string
            last_module = component.split(".")[-1]
            # now remove the last module from the component string
            component = component[: -len(last_module) - 1]
            # check if the component exists in the model
            try:
                self.assert_module_exists(component)
            except ValueError as e:
                logger.error(
                    f"Error: {e}. Probably the module {component} do not exists in the model. If the module is the attention_matrix_hook, try callig HookedModel.set_custom_hooks() or setting attn_implementation == 'custom_eager'.  Now we will skip the hook for the component {component}"
                )
                continue
            if last_module == "input":
                hook_handlers.append(
                    get_module_by_path(
                        self.hf_model, component
                    ).register_forward_pre_hook(
                        partial(hook_function, output=None), with_kwargs=True
                    )
                )
            elif last_module == "output":
                hook_handlers.append(
                    get_module_by_path(self.hf_model, component).register_forward_hook(
                        hook_function, with_kwargs=True
                    )
                )
            else:
                logger.warning(
                    f"Warning: the last module of the component {component} is not 'input' or 'output'. We will skip this hook"
                )

        return hook_handlers

    def remove_hooks(self, hook_handlers):
        """
        Remove all the hooks from the model
        """
        for hook_handler in hook_handlers:
            hook_handler.remove()

    @torch.no_grad()
    def generate(
        self,
        inputs,
        generation_config: Optional[GenerationConfig] = None,
        target_token_positions: Optional[List[str]] = None,
        return_text: bool = False,
        **kwargs,
    ) -> ActivationCache:
        r"""
        __WARNING__: This method could be buggy in the return dict of the output. Pay attention!

        Generate new tokens using the model and the inputs passed as argument
        Args:
            inputs (dict): dictionary with the inputs of the model {"input_ids": ..., "attention_mask": ..., "pixel_values": ...}
            generation_config (Optional[GenerationConfig]): original hf dataclass with the generation configuration
            **kwargs: additional arguments to control hooks generation (i.e. ablation_queries, patching_queries)
        Returns:
            output (ActivationCache): dictionary with the output of the model

        Examples:
            >>> inputs = {"input_ids": torch.tensor([[101, 1234, 1235, 102]]), "attention_mask": torch.tensor([[1, 1, 1, 1]])}
            >>> model.generate(inputs)
            {'sequences': tensor([[101, 1234, 1235, 102]])}
        """
        # Initialize cache for logits
        # raise NotImplementedError("This method is not working. It needs to be fixed")
        cache = ActivationCache()
        hook_handlers = None
        if target_token_positions is not None or self.additional_interventions is not None:
            string_tokens = self.to_string_tokens(
                self.input_handler.get_input_ids(inputs).squeeze()
            )
            token_indexes, token_dict = TokenIndex(
                self.config.model_name, pivot_positions=None
            ).get_token_index(tokens=[], string_tokens=string_tokens, return_type="all")
            assert isinstance(token_indexes, list), "Token index must be a list"
            assert isinstance(token_dict, dict), "Token dict must be a dict"
            hooks = self.create_hooks(
                inputs=inputs,
                token_dict=token_dict,
                token_indexes=token_indexes,
                cache=cache,
                **kwargs,
            )
            hook_handlers = self.set_hooks(hooks)

        inputs = self.input_handler.prepare_inputs(inputs, self.first_device)
        # print(inputs.keys())
        output = self.hf_model.generate(
            **inputs,  # type: ignore
            generation_config=generation_config,
            # output_scores=False,  # type: ignore
        )
        if hook_handlers:
            self.remove_hooks(hook_handlers)
        if return_text:
            return self.hf_tokenizer.decode(output[0], skip_special_tokens=True)  # type: ignore
        if not cache.is_empty():
            # if the cache is not empty, we will return the cache
            output = {"generation_output": output, "cache": cache}
        return output  # type: ignore

    def extract_cache(
        self,
        dataloader,
        target_token_positions: Union[
            List[Union[str, int, Tuple[int, int]]],
            List[str],
            List[int],
            List[Tuple[int, int]],
        ],
        extraction_config: ExtractionConfig = ExtractionConfig(),
        interventions: Optional[List[Intervention]] = None,
        batch_saver: Callable = lambda x: None,
        move_to_cpu_after_forward: bool = True,
        # save_other_batch_elements: bool = False,
        **kwargs,
    ):
        """
        Extract internal model activations from a dataset using forward passes.
        
        This is the primary method for extracting activations from the model for
        interpretability analysis. It processes each batch in the dataloader,
        performs forward passes while capturing specified activations, and
        aggregates results into a comprehensive activation cache.
        
        The method supports flexible extraction configurations, allowing users to
        specify exactly which model components to monitor and which token positions
        to extract from. It can handle both text-only and multimodal inputs.
        
        Args:
            dataloader (Iterable[Dict]): An iterable containing batches of model inputs.
                Each element must be a dictionary containing the inputs that the model
                expects (e.g., input_ids, attention_mask, pixel_values for VLMs).
                Common format: {"input_ids": tensor, "attention_mask": tensor, ...}
                
            target_token_positions (Union[List[Union[str, int, Tuple[int, int]]], ...]):
                Specification of which token positions to extract activations from.
                Supports multiple formats:
                - Strings: "last", "first", "end-image", "start-image", "all", etc.
                - Integers: Specific token indices (e.g., -1 for last token)
                - Tuples: Token ranges (e.g., (2, 10) for positions 2 through 10)
                - Mixed lists: Combinations of the above types
                See TokenIndex.get_token_index for complete specification.
                
            extraction_config (ExtractionConfig, optional): Configuration object
                specifying which activations to extract from the model.
                Controls extraction from residual streams, attention mechanisms,
                MLP layers, etc. Defaults to ExtractionConfig() (basic config).
                
            interventions (List[Intervention], optional): List of intervention
                objects to apply during forward passes. Enables ablation studies,
                activation patching, and other causal analysis methods.
                Defaults to None (no interventions).
                
            batch_saver (Callable, optional): Function to extract and save additional
                information from each batch element (e.g., labels, metadata).
                Should take a batch element and return a dictionary of items to save.
                Defaults to lambda x: None (no additional saving).
                
            move_to_cpu_after_forward (bool, optional): Whether to move extracted
                activations to CPU immediately after each forward pass. Helps
                manage GPU memory usage for large datasets. Defaults to True.
                
            **kwargs: Additional keyword arguments passed to the forward method.
                Can include ablation_queries, patching_queries, and other parameters
                for controlling hook generation and intervention behavior.
        
        Returns:
            ActivationCache: A comprehensive cache object containing all extracted
                activations organized by activation type and layer. The cache includes:
                - Activation tensors keyed by component names (e.g., 'resid_out_0')
                - Token position mappings indicating which positions were extracted
                - Additional batch elements saved via batch_saver
                - Metadata about the extraction process
        
        Example:
            >>> # Basic usage: extract residual stream outputs
            >>> dataloader = [
            ...     {"input_ids": torch.tensor([[101, 1234, 1235, 102]]),
            ...      "attention_mask": torch.tensor([[1, 1, 1, 1]])},
            ...     # ... more batches
            ... ]
            >>> 
            >>> config = ExtractionConfig(extract_resid_out=True, save_input_ids=True)
            >>> cache = model.extract_cache(
            ...     dataloader,
            ...     target_token_positions=["last"],
            ...     extraction_config=config
            ... )
            >>> print(cache.keys())  # ['resid_out_0', 'resid_out_1', ..., 'input_ids']
            
            >>> # Advanced usage: extract attention patterns with interventions
            >>> config = ExtractionConfig(
            ...     extract_attn_pattern=True,
            ...     extract_resid_out=True,
            ...     attn_heads=[{"layer": 0, "head": 5}]
            ... )
            >>> interventions = [some_intervention_object]
            >>> cache = model.extract_cache(
            ...     dataloader,
            ...     target_token_positions=["last", (5, 10)],
            ...     extraction_config=config,
            ...     interventions=interventions,
            ...     batch_saver=lambda x: {"labels": x.get("labels", None)}
            ... )
        
        Note:
            - Large datasets may require careful memory management via move_to_cpu_after_forward
            - The extraction_config must have is_not_empty() == True to extract anything
            - Token position specifications are flexible and support various analysis needs
            - Interventions enable causal analysis but may slow down extraction
        
        See Also:
            - ExtractionConfig: For configuring which activations to extract
            - TokenIndex.get_token_index: For token position specification details
            - ActivationCache: For working with the returned activation data
        """

        logger.info("HookedModel: Extracting cache")
        all_cache = ActivationCache()  # a list of dictoionaries, each dictionary contains the activations of the model for a batch (so a dict of tensors)
        attn_pattern = (
            ActivationCache()
        )  # Initialize the dictionary to hold running averages

        # if register_agregation is in the kwargs, we will register the aggregation of the attention pattern
        if "register_aggregation" in kwargs:
            all_cache.register_aggregation(
                kwargs["register_aggregation"][0], kwargs["register_aggregation"][1]
            )
            attn_pattern.register_aggregation(
                kwargs["register_aggregation"][0], kwargs["register_aggregation"][1]
            )

        # example_dict = {}
        n_batches = 0  # Initialize batch counter

        for batch in progress(dataloader, desc="Extracting cache", total=len(dataloader)):

            # log_memory_usage("Extract cache - Before batch")
            # tokens, others = batch
            # inputs = {k: v.to(self.first_device) for k, v in tokens.items()}

            # get input_ids, attention_mask, and if available, pixel_values from batch (that is a dictionary)
            # then move them to the first device

            inputs = self.input_handler.prepare_inputs(
                batch, self.first_device
            )  # require_grads is False, gradients handled by hook if needed
            others = {k: v for k, v in batch.items() if k not in inputs}

            cache = self.forward(
                inputs,
                target_token_positions=target_token_positions,
                pivot_positions=batch.get("pivot_positions", None),
                external_cache=attn_pattern,
                batch_idx=n_batches,
                extraction_config=extraction_config,
                interventions=interventions,
                vocabulary_index=batch.get("vocabulary_index", None),
                **kwargs,
            )

            # Compute input gradients if requested

            # possible memory leak from here -___--------------->
            additional_dict = batch_saver(
                others
            )  # TODO: Maybe keep the batch_saver in a different cache
            if additional_dict is not None:
                # cache = {**cache, **additional_dict}if a
                cache.update(additional_dict)

            if move_to_cpu_after_forward:
                cache.cpu()

            n_batches += 1  # Increment batch counter# Process and remove "pattern_" keys from cache
            all_cache.cat(cache)

            del cache

            # Use the new cleanup_tensors method from InputHandler to free memory
            self.input_handler.cleanup_tensors(inputs, others)

            torch.cuda.empty_cache()

        logger.debug("Forward pass finished - started to aggregate different batch")
        all_cache.update(attn_pattern)
        # all_cache["example_dict"] = example_dict
        # logger.info("HookedModel: Aggregation finished")

        torch.cuda.empty_cache()

        # add a metadata field to the cache
        all_cache.add_metadata(
            target_token_positions=target_token_positions,
            model_name=self.config.model_name,
            extraction_config=extraction_config.to_dict(),
            interventions=interventions,
        )

        return all_cache

    def compute_patching(
        self,
        target_token_positions: List[Union[str, int, Tuple[int, int]]],
        # counterfactual_dataset,
        base_dataloader,
        target_dataloader,
        patching_query=[
            {
                "patching_elem": "@end-image",
                "layers_to_patch": [1, 2, 3, 4],
                "activation_type": "resid_in_{}",
            }
        ],
        base_dictonary_idxs: Optional[List[List[int]]] = None,
        target_dictonary_idxs: Optional[List[List[int]]] = None,
        return_logit_diff: bool = False,
        batch_saver: Callable = lambda x: None,
        **kwargs,
    ) -> ActivationCache:
        r"""
        Method for activation patching. This substitutes the activations of the model
        with the activations of the counterfactual dataset.

        It performs three forward passes:
        1. Forward pass on the base dataset to extract the activations of the model (cat).
        2. Forward pass on the target dataset to extract clean logits (dog)
        [to compare against the patched logits].
        3. Forward pass on the target dataset to patch (cat) into (dog)
        and extract the patched logits.

        Arguments:
            - target_token_positions (Union[Union[str, int, Tuple[int, int]], List[Union[str, int, Tuple[int, int]]]]): List of tokens to extract the activations from. See TokenIndex.get_token_index for more details
            - base_dataloader (torch.utils.data.DataLoader): Dataloader with the base dataset. (dataset where we sample the activations from)
            - target_dataloader (torch.utils.data.DataLoader): Dataloader with the target dataset. (dataset where we patch the activations)
            - patching_query (list[dict]): List of dictionaries with the patching queries. Each dictionary must have the keys "patching_elem", "layers_to_patch" and "activation_type". The "patching_elem" is the token to patch, the "layers_to_patch" is the list of layers to patch and the "activation_type" is the type of the activation to patch. The activation type must be one of the following: "resid_in_{}", "resid_out_{}", "resid_mid_{}", "attn_in_{}", "attn_out_{}", "values_{}". The "{}" will be replaced with the layer index.
            - base_dictonary_idxs (list[list[int]]): List of list of integers with the indexes of the tokens in the dictonary that we are interested in. It's useful to extract the logit difference between the clean logits and the patched logits.
            - target_dictonary_idxs (list[list[int]]): List of list of integers with the indexes of the tokens in the dictonary that we are interested in. It's useful to extract the logit difference between the clean logits and the patched logits.
            - return_logit_diff (bool): If True, it will return the logit difference between the clean logits and the patched logits.


        Returns:
            final_cache (ActivationCache): dictionary with the activations of the model. The keys are the names of the activations and the values are the activations themselve

        Examples:
            >>> model.compute_patching(
            >>>     target_token_positions=["end-image", " last"],
            >>>     base_dataloader=base_dataloader,
            >>>     target_dataloader=target_dataloader,
            >>>     base_dictonary_idxs=base_dictonary_idxs,
            >>>     target_dictonary_idxs=target_dictonary_idxs,
            >>>     patching_query=[
            >>>         {
            >>>             "patching_elem": "@end-image",
            >>>             "layers_to_patch": [1, 2, 3, 4],
            >>>             "activation_type": "resid_in_{}",
            >>>         }
            >>>     ],
            >>>     return_logit_diff=False,
            >>>     batch_saver=lambda x: None,
            >>> )
            >>> print(final_cache)
            {
                "resid_out_0": tensor of shape [batch, seq_len, hidden_size] with the activations of the residual stream of layer 0
                "resid_mid_0": tensor of shape [batch, seq_len, hidden_size] with the activations of the residual stream of layer 0
                ....
                "logit_diff_variation": tensor of shape [batch] with the logit difference variation
                "logit_diff_in_clean": tensor of shape [batch] with the logit difference in the clean logits
                "logit_diff_in_patched": tensor of shape [batch] with the logit difference in the patched logits
            }
        """
        logger.debug("HookedModel: Computing patching")

        logger.debug("HookedModel: Forward pass started")
        logger.info(
            f"HookedModel: Patching elements: {[q['patching_elem'] for q in patching_query]} at {[query['activation_type'][:-3] for query in patching_query]}"
        )

        # if target_token_positions is not a list, convert it to a list
        if not isinstance(target_token_positions, list):
            target_token_positions = [target_token_positions]

        # get a random number in the range of the dataset to save a random batch
        all_cache = ActivationCache()
        # for each batch in the dataset
        for index, (base_batch, target_batch) in progress(
            enumerate(zip(base_dataloader, target_dataloader)),
            desc="Computing patching on the dataset:",
            total=len(base_dataloader),
        ):
            torch.cuda.empty_cache()
            inputs = self.input_handler.prepare_inputs(base_batch, self.first_device)

            # set the right arguments for extract the patching activations
            activ_type = [query["activation_type"][:-3] for query in patching_query]

            args = {
                "extract_resid_out": True,
                "extract_resid_in": False,
                "extract_resid_mid": False,
                "extract_attn_in": False,
                "extract_attn_out": False,
                "extract_head_values": False,
                "extract_head_out": False,
                "extract_avg_attn_pattern": False,
                "extract_avg_values_vectors_projected": False,
                "extract_head_values_projected": False,
                "extract_avg": False,
                "ablation_queries": None,
                "patching_queries": None,
                "external_cache": None,
                "attn_heads": "all",
                "batch_idx": None,
                "move_to_cpu": False,
            }

            if "resid_in" in activ_type:
                args["extract_resid_in"] = True
            if "resid_out" in activ_type:
                args["extract_resid_out"] = True
            if "resid_mid" in activ_type:
                args["extract_intermediate_states"] = True
            if "attn_in" in activ_type:
                args["extract_attn_in"] = True
            if "attn_out" in activ_type:
                args["extract_attn_out"] = True
            if "values" in activ_type:
                args["extract_head_values"] = True
            # other cases

            # first forward pass to extract the base activations
            base_cache = self.forward(
                inputs=inputs,
                target_token_positions=target_token_positions,
                pivot_positions=base_batch.get("pivot_positions", None),
                external_cache=args["external_cache"],
                batch_idx=args["batch_idx"],
                extraction_config=ExtractionConfig(**args),
                interventions=args["interventions"],
                move_to_cpu=args["move_to_cpu"],
            )

            # extract the target activations
            target_inputs = self.input_handler.prepare_inputs(
                target_batch, self.first_device
            )

            requested_position_to_extract = []
            interventions = []
            for query in patching_query:
                if (
                    query["patching_elem"].split("@")[1]
                    not in requested_position_to_extract
                ):
                    requested_position_to_extract.append(
                        query["patching_elem"].split("@")[1]
                    )
                interventions.extend(
                    [
                        Intervention(
                            type="full",
                            activation=query["activation_type"].format(layer),
                            token_positions=[query["patching_elem"].split("@")[1]],
                            patching_values=base_cache[
                                query["activation_type"].format(layer)
                            ]
                            .detach()
                            .clone(),
                        )
                        for layer in query["layers_to_patch"]
                    ]
                )

                # query["patching_activations"] = base_cache
                #     )
                # query["base_activation_index"] = base_cache["mapping_index"][
                #     query["patching_elem"].split("@")[1]
                # ]

            # second forward pass to extract the clean logits
            target_clean_cache = self.forward(
                target_inputs,
                target_token_positions=requested_position_to_extract,
                pivot_positions=target_batch.get("pivot_positions", None),
                # move_to_cpu=True,
            )

            # merge requested_position_to_extract with extracted_token_positio
            # third forward pass to patch the activations
            target_patched_cache = self.forward(
                target_inputs,
                target_token_positions=list(
                    set(target_token_positions + requested_position_to_extract)
                ),
                pivot_positions=target_batch.get("pivot_positions", None),
                patching_queries=patching_query,
                **kwargs,
            )

            if return_logit_diff:
                if base_dictonary_idxs is None or target_dictonary_idxs is None:
                    raise ValueError(
                        "To compute the logit difference, you need to pass the base_dictonary_idxs and the target_dictonary_idxs"
                    )
                logger.info("HookedModel: Computing logit difference")
                # get the target tokens (" cat" and " dog")
                base_targets = base_dictonary_idxs[index]
                target_targets = target_dictonary_idxs[index]

                # compute the logit difference
                result_diff = logit_diff(
                    base_label_tokens=[s for s in base_targets],
                    target_label_tokens=[c for c in target_targets],
                    target_clean_logits=target_clean_cache["logits"],
                    target_patched_logits=target_patched_cache["logits"],
                )
                target_patched_cache["logit_diff_variation"] = result_diff[
                    "diff_variation"
                ]
                target_patched_cache["logit_diff_in_clean"] = result_diff[
                    "diff_in_clean"
                ]
                target_patched_cache["logit_diff_in_patched"] = result_diff[
                    "diff_in_patched"
                ]

            # compute the KL divergence
            result_kl = kl_divergence_diff(
                base_logits=base_cache["logits"],
                target_clean_logits=target_clean_cache["logits"],
                target_patched_logits=target_patched_cache["logits"],
            )
            for key, value in result_kl.items():
                target_patched_cache[key] = value

            target_patched_cache["base_logits"] = base_cache["logits"]
            target_patched_cache["target_clean_logits"] = target_clean_cache["logits"]
            # rename logits to target_patched_logits
            target_patched_cache["target_patched_logits"] = target_patched_cache[
                "logits"
            ]
            del target_patched_cache["logits"]

            target_patched_cache.cpu()

            # all_cache.append(target_patched_cache)
            all_cache.cat(target_patched_cache)

        logger.debug(
            "HookedModel: Forward pass finished - started to aggregate different batch"
        )
        # final_cache = aggregate_cache_efficient(all_cache)

        logger.debug("HookedModel: Aggregation finished")
        return all_cache

    def _compute_input_gradients(self, cache, logits, vocabulary_index):
        """
        Private method to compute gradients of logits with respect to input embeddings.

        Args:
            cache (ActivationCache): Cache containing logits and input_embeddings
            logits (torch.Tensor): Model output logits
            vocabulary_index (int): Index in the vocabulary for which to compute gradients

        Returns:
            bool: True if gradients were successfully computed, False otherwise
        """

        supported_keys = ["input_embeddings"]

        if any(key not in cache for key in supported_keys):
            logger.warning(
                f"Cannot compute gradients: {supported_keys} not found in cache. "
                "Ensure extraction_config.extract_embed is True."
            )
            return False

        input_embeds = cache["input_embeddings"]

        if not input_embeds.requires_grad:
            logger.warning(
                "Cannot compute gradients: input embeddings do not require gradients."
            )
            return False

        # Select the specific logit for the target token
        target_logits = logits[0, -1, vocabulary_index]

        # Zero out existing gradients if any
        if input_embeds.grad is not None:
            input_embeds.grad.zero_()

        # try:
        # Backward pass - use retain_graph=False to free memory after each backward pass
        target_logits.backward(retain_graph=False)

        # Store the computed gradients before they're cleared
        for key in supported_keys:
            if key in cache and input_embeds.grad is not None:
                cache[key + "_gradients"] = input_embeds.grad.detach().clone()

        # Process token slicing
        tupled_indexes = tuple(cache["token_dict"].values())
        flatten_indexes = [item for sublist in tupled_indexes for item in sublist]
        for key in supported_keys:
            cache[key] = cache[key][..., flatten_indexes, :].detach()
            if key + "_gradients" in cache:
                cache[key + "_gradients"] = cache[key + "_gradients"][
                    ..., flatten_indexes, :
                ].detach()

        # Explicitly free memory
        torch.cuda.empty_cache()
        return True

        # except RuntimeError as e:
        #     logger.error(f"Error computing gradients: {e}")
        #     # Ensure memory is freed even in case of error
        #     torch.cuda.empty_cache()
        #     return False
