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
    Configuration of the HookedModel

    Arguments:
        model_name (str): the name of the model to load
        device_map (Literal["balanced", "cuda", "cpu", "auto"]): the device to use for the model
        torch_dtype (torch.dtype): the dtype of the model
        attn_implementation (Literal["eager", "flash_attention_2"]): the implementation of the attention
        batch_size (int): the batch size of the model. FOR NOW, ONLY BATCH SIZE 1 IS SUPPORTED. USE AT YOUR OWN RISK
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
    Configuration of the extraction of the activations of the model. It store what activations you want to extract from the model.

    Arguments:
        extract_resid_in (bool): if True, extract the input of the residual stream
        extract_resid_mid (bool): if True, extract the output of the intermediate stream
        extract_resid_out (bool): if True, extract the output of the residual stream
        extract_resid_in_post_layernorm(bool): if True, extract the input of the residual stream after the layernorm
        extract_attn_pattern (bool): if True, extract the attention pattern of the attn
        extract_head_values_projected (bool): if True, extract the values vectors projected of the model
        extract_head_keys_projected (bool): if True, extract the key vectors projected of the model
        extract_head_queries_projected (bool): if True, extract the query vectors projected of the model
        extract_head_keys (bool): if True, extract the keys of the attention
        extract_head_values (bool): if True, extract the values of the attention
        extract_head_queries (bool): if True, extract the queries of the attention
        extract_values (bool): if True, extract the values. This do not reshape the values to the attention heads as extract_head_values does
        extract_keys (bool): if True, extract the keys. This do not reshape the keys to the attention heads as extract_head_keys does
        extract_queries (bool): if True, extract the queries. This do not reshape the queries to the attention heads as extract_head_queries does
        extract_last_layernorm (bool): if True, extract the last layernorm of the model
        extract_head_out (bool): if True, extract the output of the heads [DEPRECATED]
        extract_attn_out (bool): if True, extract the output of the attention of the attn_heads passed
        extract_attn_in (bool): if True, extract the input of the attention of the attn_heads passed
        extract_mlp_out (bool): if True, extract the output of the mlp of the attn
        save_input_ids (bool): if True, save the input_ids in the cache
        avg (bool): if True, extract the average of the activations over the target positions
        avg_over_example (bool): if True, extract the average of the activations over the examples (it required a external cache to save the running avg)
        attn_heads (Union[list[dict], Literal["all"]]): list of dictionaries with the layer and head to extract the attention pattern or 'all' to
        attn_pattern_avg (Literal["mean", "sum", "baseline_ratio", "none"]): the type of average to perform over the attention pattern. See hook.py attention_pattern_head for more details
        attn_pattern_row_positions (Optional[Union[List[int], List[Tuple], List[str], List[Union[int, Tuple, str]]]): the row positions of the attention pattern to extract. See hook.py attention_pattern_head for more details
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
    extract_values: bool = False
    extract_keys: bool = False
    extract_queries: bool = False
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
        Return True if at least one extraction option is enabled in the config, False otherwise.
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
        """
        Return the configuration as a dictionary.
        """
        return self.__dict__


class HookedModel:
    """
    Wrapper around a HuggingFace model for extracting activations and supporting mechanistic interpretability methods.
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
        return cls(HookedModelConfig(model_name=model_name, **kwargs))

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
        Substitute custom modules (e.g., attention) into the model for advanced interpretability.
        """
        logger.info("HookedModel: Setting custom modules.")
        self.module_wrapper_manager.substitute_attention_module(self.hf_model)

    def restore_original_modules(self):
        """
        Restore the original modules of the model, removing any custom substitutions.
        """
        logger.info("HookedModel: Restoring original modules.")
        self.module_wrapper_manager.restore_original_attention_module(self.hf_model)

    def is_multimodal(self) -> bool:
        """
        Return True if the model supports multimodal inputs (e.g., images), False otherwise.
        """
        if self.processor is not None:
            return True
        return False

    def use_full_model(self):
        """
        Switch to the full model (including multimodal components if available).
        """
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
        """
        Switch to using only the language model component (text-only mode).
        """
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
        Return the tokenizer associated with the model.
        """
        return self.hf_tokenizer

    def get_text_tokenizer(self):
        r"""
        If the tokenizer is a processor, return just the tokenizer. If the tokenizer is a tokenizer, return the tokenizer

        Args:
            None

        Returns:
            tokenizer: the tokenizer of the model
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
        """
        Return the language modeling head (output projection layer) of the model.
        """
        return get_attribute_by_name(self.hf_model, self.model_config.unembed_matrix)

    def get_last_layernorm(self):
        """
        Return the last layer normalization module of the model.
        """
        return get_attribute_by_name(self.hf_model, self.model_config.last_layernorm)

    def get_image_placeholder(self) -> str:
        """
        Return the image placeholder string used by the tokenizer for multimodal models.
        """
        return self.image_placeholder

    def eval(self):
        """
        Set the model to evaluation mode.
        """
        self.hf_model.eval()

    def device(self):
        """
        Return the device (e.g., 'cuda', 'cpu') where the model is located.
        """
        return self.first_device

    def register_forward_hook(self, component: str, hook_function: Callable):
        r"""
        Register a forward hook on a model component.

        Args:
            component (str): Name of the model component.
            hook_function (Callable): Function to call during forward pass.

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
        """
        Register a list of interventions to be applied during forward passes.
        """
        self.additional_interventions = interventions
        logger.debug(f"HookedModel: Registered {len(interventions)} interventions")

    def clean_interventions(self):
        """
        Remove all registered interventions.
        """
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
        # register the intervention hooks as first thing to do
        if self.additional_interventions is not None:
            hooks += self.intervention_manager.create_intervention_hooks(
                interventions=self.additional_interventions, token_dict=token_dict
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
                        cache_key="head_queries_",
                        token_indexes=token_indexes,
                        head_dim=self.model_config.head_dim,
                        avg=extraction_config.avg,
                        layer=i,
                        head=head,
                        num_key_value_groups=self.model_config.num_key_value_groups,
                        num_attention_heads=self.model_config.num_attention_heads,
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
                        cache_key="head_values_",
                        token_indexes=token_indexes,
                        head_dim=self.model_config.head_dim,
                        avg=extraction_config.avg,
                        layer=i,
                        head=head,
                        num_key_value_groups=self.model_config.num_key_value_groups,
                        num_attention_heads=self.model_config.num_attention_heads,
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
                        cache_key="head_keys_",
                        token_indexes=token_indexes,
                        head_dim=self.model_config.head_dim,
                        avg=extraction_config.avg,
                        layer=i,
                        head=head,
                        num_key_value_groups=self.model_config.num_key_value_groups,
                        num_attention_heads=self.model_config.num_attention_heads,
                    ),
                }
                for i, head in zip(layer_indexes, head_indexes)
            ]

        if extraction_config.extract_values:
            hooks += [
                {
                    "component": self.model_config.head_value_hook_name.format(i),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"values_L{i}",
                        token_indexes=token_indexes,
                        avg=extraction_config.avg,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]

        if extraction_config.extract_keys:
            hooks += [
                {
                    "component": self.model_config.head_key_hook_name.format(i),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"keys_L{i}",
                        token_indexes=token_indexes,
                        avg=extraction_config.avg,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]

        if extraction_config.extract_queries:
            hooks += [
                {
                    "component": self.model_config.head_query_hook_name.format(i),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"queries_L{i}",
                        token_indexes=token_indexes,
                        avg=extraction_config.avg,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
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
        Remove all hooks from the model using the provided handlers.
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
        if (
            target_token_positions is not None
            or self.additional_interventions is not None
        ):
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
        r"""
        Method to extract the activations of the model from a specific dataset. Compute a forward pass for each batch of the dataloader and save the activations in the cache.

        Arguments:
            - dataloader (iterable): dataloader with the dataset. Each element of the dataloader must be a dictionary that contains the inputs that the model expects (input_ids, attention_mask, pixel_values ...)
            - extracted_token_position (Union[Union[str, int, Tuple[int, int]], List[Union[str, int, Tuple[int, int]]]]): list of tokens to extract the activations from (["last", "end-image", "start-image", "first", -1, (2,10)]). See TokenIndex.get_token_index for more details
            - batch_saver (Callable): function to save in the cache the additional element from each elemtn of the batch (For example, the labels of the dataset)
            - move_to_cpu_after_forward (bool): if True, move the activations to the cpu right after the any forward pass of the model
            - dict_token_index (Optional[torch.Tensor]): If provided, specifies the index in the vocabulary for which to compute gradients of logits with respect to input embeddings. Requires extraction_config.extract_input_embeddings_for_grad to be True.
            - **kwargs: additional arguments to control hooks generation, basically accept any argument handled by the `.forward` method (i.e. ablation_queries, patching_queries, extract_resid_in)

        Returns:
            final_cache: dictionary with the activations of the model. The keys are the names of the activations and the values are the activations themselve

        Examples:
            >>> dataloader = [{"input_ids": torch.tensor([[101, 1234, 1235, 102]]), "attention_mask": torch.tensor([[1, 1, 1, 1]]), "labels": torch.tensor([1])}, ...]
            >>> model.extract_cache(dataloader, extracted_token_position=["last"], batch_saver=lambda x: {"labels": x["labels"]})
            {'resid_out_0': tensor([[[0.1, 0.2, 0.3, 0.4]]], grad_fn=<CopyBackwards>), 'labels': tensor([1]), 'mapping_index': {'last': [0]}}
        """

        logger.info("HookedModel: Extracting cache")

        # get the function to save in the cache the additional element from the batch sime

        logger.info("HookedModel: Forward pass started")
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

        for batch in progress(
            dataloader, desc="Extracting cache", total=len(dataloader)
        ):
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
