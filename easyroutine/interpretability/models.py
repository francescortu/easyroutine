from dataclasses import dataclass
from easyroutine.logger import logger
from transformers import (
    ChameleonProcessor,
    ChameleonForConditionalGeneration,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    PixtralProcessor,
    LlamaForCausalLM,
    LlamaTokenizer,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    LlamaTokenizerFast,
    LlavaNextProcessor,
    Gemma3ForConditionalGeneration,
    Gemma3Processor,
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionProcessor,
)
import random
from typing import List, Literal, Union, Dict, Optional, Tuple, Any
import torch
import yaml


@dataclass
class ModelConfig:
    r"""
    Configuration class for storing model specific parameters.

    Attributes:
        residual_stream_input_hook_name (str): Name of the residual stream torch module where attach the hook
        residual_stream_hook_name (str): Name of the residual stram torch module where attach the hook
        intermediate_stream_hook_name (str): Name of the intermediate stream torch module where attach the hook
        residual_stream_input_post_layernorm_hook_name (str): Name of the residual stream input post layer norm
        head_key_hook_name (str): Name of the attention key torch module where attach the hook
        head_value_hook_name (str): Name of the attention value torch module where attach the hook
        head_query_hook_name (str): Name of the attention key torch module where attach the hook
        attn_in_hook_name (str): Name of the attention input torch module where attach the hook
        attn_out_hook_name (str): Name of the attention output torch module where attach the hook
        attn_matrix_hook_name (str): Name of the attention matrix torch module where attach the hook
        mlp_out_hook_name (str): Name of the mlp output torch module where attach the hook
        attn_out_proj_weight (str): Name of the attention output projection weight
        attn_out_proj_bias (str): Name of the attention output projection bias
        embed_tokens (str): Name of the embedding tokens torch module where attach the hook
        num_hidden_layers (int): Number of hidden layers
        num_attention_heads (int): Number of attention heads
        hidden_size (int): Hidden size of the transformer model
        num_key_value_heads (int): Number of key value heads
        num_key_value_groups (int): Number of key value groups
        head_dim (int): Dimension of the attention head

    """

    residual_stream_input_hook_name: str
    residual_stream_hook_name: str
    intermediate_stream_hook_name: str
    residual_stream_input_post_layernorm_hook_name: str
    head_key_hook_name: str
    head_value_hook_name: str
    head_query_hook_name: str
    # head_out_hook_name: str
    attn_in_hook_name: str
    attn_in_hook_name: str
    attn_out_hook_name: str
    attn_o_proj_input_hook_name: str
    attn_matrix_hook_name: str
    attn_matrix_pre_softmax_hook_name: str
    mlp_out_hook_name: str
    last_layernorm_hook_name: str
    

    attn_out_proj_weight: str
    attn_out_proj_bias: str
    embed_tokens: str
    unembed_matrix: str
    last_layernorm: str

    num_hidden_layers: int
    num_attention_heads: int
    hidden_size: int
    num_key_value_heads: int
    num_key_value_groups: int
    head_dim: int
    layernorm_type: Literal["RMS", "LayerNorm"]
    
    def use_language_model(self):
        """
        for the hook, remove the "language_model" prefix: "language_model.model.layers[{}]" -> "model.layers[{}]" . Remove from the data ckass
        """
        # iterate over the dataclass fields
        for field in self.__dataclass_fields__.keys():
            # remove the "language_model." prefix
            if "hook" in field:
                setattr(self, field, getattr(self, field).replace("language_model.", ""))
            
    def restore_full_model(self):
        for field in self.__dataclass_fields__.keys():
            if "hook" in field:
                setattr(self, field, getattr(self, field).replace("model.","language_model.model."))
    
    

# SPECIFIC MODEL CONFIGURATIONS


class ModelFactory:
    r"""
    This class is a factory to load the model and the processor. It supports the following models:

    Supported Models:
        The following models are supported by this factory:

        - **Chameleon-7b**: A 7-billion parameter model for general-purpose tasks.
        - **Chameleon-30b**: A larger version of the Chameleon series with 30 billion parameters.
        - **llava-hf/llava-v1.6-mistral-7b-hf**: A 7-billion parameter model for multimodal tasks.
        - **Pixtral-12b**: Optimized for image-to-text tasks.
        - **Emu3-Chat**: Fine-tuned for conversational AI.
        - **Emu3-Gen**: Specialized in text generation tasks.
        - **Emu3-Stage1**: Pretrained for multi-stage training pipelines.
        - **Llava-onevision**: A multimodal model for vision and language tasks.
        - **hf-internal-testing**: A tiny model for internal testing purposes.

    Adding a New Model:
        To add a new model:
        1. Implement its logic in the `load_model` method.
        2. Ensure it is correctly initialized and validated.
    """

    @staticmethod
    def load_model(
        model_name: str,
        attn_implementation: str,
        torch_dtype: torch.dtype,
        device_map: str,
    ) -> Tuple[torch.nn.Module, Optional[torch.nn.Module], ModelConfig]:
        r"""
        Load the model and its configuration based on the model name.

        Args:
            model_name (str): Name of the model to load.
            attn_implementation (str): Attention implementation type. (eager, flash-attn, sdp)
            torch_dtype (torch.dtype): Data type of the model.
            device_map (str): Device map for the model.

        Returns:
            model (HuggingFaceModel): Model instance.
            model_config (ModelConfig): Model configuration.
        """
        if attn_implementation != "eager":
            logger.warning(
                "ModelFactory: Using an attention type different from eager or custom eager could have unexpected behavior in some experiments!",
            )

        language_model = None
        if model_name in ["facebook/chameleon-7b", "facebook/chameleon-30b"]:
            model = ChameleonForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
            )
            model_config = ModelConfig(
                residual_stream_input_hook_name="model.layers[{}].input",
                residual_stream_hook_name="model.layers[{}].output",
                intermediate_stream_hook_name="model.layers[{}].post_attention_layernorm.output",
                residual_stream_input_post_layernorm_hook_name="model.layers[{}].self_attn.input",
                head_key_hook_name="model.layers[{}].self_attn.k_proj.output",
                head_value_hook_name="model.layers[{}].self_attn.v_proj.output",
                head_query_hook_name="model.layers[{}].self_attn.q_proj.output",
                attn_out_hook_name="model.layers[{}].self_attn.o_proj.output",
                attn_o_proj_input_hook_name="model.layers[{}].self_attn.o_proj.input",
                attn_in_hook_name="model.layers[{}].self_attn.input",
                mlp_out_hook_name="model.layers[{}].mlp.down_proj.output",
                attn_matrix_hook_name="model.layers[{}].self_attn.attention_matrix_hook.output",
                attn_matrix_pre_softmax_hook_name="model.layers[{}].self_attn.attention_matrix_pre_softmax_hook.output",
                last_layernorm_hook_name="model.norm.input",
                attn_out_proj_weight="model.layers[{}].self_attn.o_proj.weight",
                attn_out_proj_bias="model.layers[{}].self_attn.o_proj.bias",
                embed_tokens="model.layers[0].input",
                unembed_matrix="lm_head.weight",
                last_layernorm="model.norm",
                num_hidden_layers=model.config.num_hidden_layers,
                num_attention_heads=model.config.num_attention_heads,
                hidden_size=model.config.hidden_size,
                num_key_value_heads=model.config.num_key_value_heads,
                num_key_value_groups=model.config.num_attention_heads
                // model.config.num_key_value_heads,
                head_dim=model.config.hidden_size // model.config.num_attention_heads,
                layernorm_type="RMS",
            )

        elif model_name in [ # TODO: Refactor this, bad nested if
            "mistral-community/pixtral-12b",
            "llava-hf/llava-v1.6-mistral-7b-hf",
            "llava-hf/llava-onevision-qwen2-7b-ov-hf"
        ]:
            if model_name == "mistral-community/pixtral-12b":
                model = LlavaForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    attn_implementation=attn_implementation,
                )
                model_config = ModelConfig(
                    residual_stream_input_hook_name="language_model.model.layers[{}].input",
                    residual_stream_hook_name="language_model.model.layers[{}].output",
                    intermediate_stream_hook_name="language_model.model.layers[{}].post_attention_layernorm.output",
                    residual_stream_input_post_layernorm_hook_name="language_model.model.layers[{}].self_attn.input",
                    head_key_hook_name="language_model.model.layers[{}].self_attn.k_proj.output",
                    head_value_hook_name="language_model.model.layers[{}].self_attn.v_proj.output",
                    head_query_hook_name="language_model.model.layers[{}].self_attn.q_proj.output",
                    attn_out_hook_name="language_model.model.layers[{}].self_attn.o_proj.output",
                    attn_o_proj_input_hook_name="language_model.model.layers[{}].self_attn.o_proj.input",
                    attn_in_hook_name="language_model.model.layers[{}].self_attn.input",
                    attn_matrix_hook_name="language_model.model.layers[{}].self_attn.attention_matrix_hook.output",
                    attn_matrix_pre_softmax_hook_name="language_model.model.layers[{}].self_attn.attention_matrix_pre_softmax_hook.output",
                    mlp_out_hook_name="language_model.model.layers[{}].mlp.down_proj.output",
                    last_layernorm_hook_name="language_model.model.norm.input",
                    attn_out_proj_weight="language_model.model.layers[{}].self_attn.o_proj.weight",
                    attn_out_proj_bias="language_model.model.layers[{}].self_attn.o_proj.bias",
                    embed_tokens="language_model.model.layers[0].input",
                    unembed_matrix="language_model.lm_head.weight",
                    last_layernorm="language_model.model.norm",
                    num_hidden_layers=model.language_model.config.num_hidden_layers,
                    num_attention_heads=model.language_model.config.num_attention_heads,
                    hidden_size=model.language_model.config.hidden_size,
                    num_key_value_heads=model.language_model.config.num_key_value_heads,
                    num_key_value_groups=model.language_model.config.num_attention_heads // model.language_model.config.num_key_value_heads,
                    head_dim=model.language_model.config.head_dim,
                    layernorm_type="RMS",
                )
            elif model_name == "llava-hf/llava-v1.6-mistral-7b-hf":
            
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    attn_implementation=attn_implementation,
                )
                model_config = ModelConfig(
                    residual_stream_input_hook_name="language_model.model.layers[{}].input",
                    residual_stream_hook_name="language_model.model.layers[{}].output",
                    intermediate_stream_hook_name="language_model.model.layers[{}].post_attention_layernorm.output",
                    residual_stream_input_post_layernorm_hook_name="language_model.model.layers[{}].self_attn.input",
                    head_key_hook_name="language_model.model.layers[{}].self_attn.k_proj.output",
                    head_value_hook_name="language_model.model.layers[{}].self_attn.v_proj.output",
                    head_query_hook_name="language_model.model.layers[{}].self_attn.q_proj.output",
                    attn_out_hook_name="language_model.model.layers[{}].self_attn.o_proj.output",
                    attn_o_proj_input_hook_name="language_model.model.layers[{}].self_attn.o_proj.input",
                    attn_in_hook_name="language_model.model.layers[{}].self_attn.input",
                    attn_matrix_hook_name="language_model.model.layers[{}].self_attn.attention_matrix_hook.output",
                    attn_matrix_pre_softmax_hook_name="language_model.model.layers[{}].self_attn.attention_matrix_pre_softmax_hook.output",
                    mlp_out_hook_name="language_model.model.layers[{}].mlp.down_proj.output",
                    last_layernorm_hook_name="language_model.model.norm.input",
                    attn_out_proj_weight="language_model.model.layers[{}].self_attn.o_proj.weight",
                    attn_out_proj_bias="language_model.model.layers[{}].self_attn.o_proj.bias",
                    embed_tokens="language_model.model.layers[0].input",
                    unembed_matrix="language_model.lm_head.weight",
                    last_layernorm="language_model.model.norm",
                    num_hidden_layers=model.language_model.config.num_hidden_layers,
                    num_attention_heads=model.language_model.config.num_attention_heads,
                    hidden_size=model.language_model.config.hidden_size,
                    num_key_value_heads=model.language_model.config.num_key_value_heads,
                    num_key_value_groups=model.language_model.config.num_attention_heads // model.language_model.config.num_key_value_heads,
                    head_dim=model.language_model.config.head_dim,
                    layernorm_type="RMS",
                )
                    
            
            elif model_name =="llava-hf/llava-onevision-qwen2-7b-ov-hf":

                model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    attn_implementation=attn_implementation,
                )
                model_config = ModelConfig(
                    residual_stream_input_hook_name="language_model.model.layers[{}].input",
                    residual_stream_hook_name="language_model.model.layers[{}].output",
                    intermediate_stream_hook_name="language_model.model.layers[{}].post_attention_layernorm.output",
                    residual_stream_input_post_layernorm_hook_name="language_model.model.layers[{}].self_attn.input",
                    head_key_hook_name="language_model.model.layers[{}].self_attn.k_proj.output",
                    head_value_hook_name="language_model.model.layers[{}].self_attn.v_proj.output",
                    head_query_hook_name="language_model.model.layers[{}].self_attn.q_proj.output",
                    attn_out_hook_name="language_model.model.layers[{}].self_attn.o_proj.output",
                    attn_o_proj_input_hook_name="language_model.model.layers[{}].self_attn.o_proj.input",
                    attn_in_hook_name="language_model.model.layers[{}].self_attn.input",
                    attn_matrix_hook_name="language_model.model.layers[{}].self_attn.attention_matrix_hook.output",
                    attn_matrix_pre_softmax_hook_name="language_model.model.layers[{}].self_attn.attention_matrix_pre_softmax_hook.output",
                    mlp_out_hook_name="language_model.model.layers[{}].mlp.down_proj.output",
                    last_layernorm_hook_name="language_model.model.norm.input",
                    attn_out_proj_weight="language_model.model.layers[{}].self_attn.o_proj.weight",
                    attn_out_proj_bias="language_model.model.layers[{}].self_attn.o_proj.bias",
                    embed_tokens="language_model.model.layers[0].input",
                    unembed_matrix="language_model.lm_head.weight",
                    last_layernorm="language_model.model.norm",
                    num_hidden_layers=model.language_model.config.num_hidden_layers,
                    num_attention_heads=model.language_model.config.num_attention_heads,
                    hidden_size=model.language_model.config.hidden_size,
                    num_key_value_heads=model.language_model.config.num_key_value_heads,
                    num_key_value_groups=model.language_model.config.num_attention_heads // model.language_model.config.num_key_value_heads,
                    head_dim=model.language_model.config.hidden_size // model.language_model.config.num_attention_heads,
                    layernorm_type="RMS",
                )
            else:
                raise ValueError("Unsupported model_name")
            language_model = model.language_model

        elif model_name in ["Emu3-Chat", "Emu3-Gen", "Emu3-Stage1"]:
            raise NotImplementedError("Emu3 model not implemented yet")

        elif model_name in ["google/gemma-3-1b-it", "google/gemma-3-4b-it", "google/gemma-3-12b-it", "google/gemma-3-27b-it"]:
            model = Gemma3ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
            )
            model_config = ModelConfig(
                residual_stream_input_hook_name="language_model.model.layers[{}].input",
                residual_stream_hook_name="language_model.model.layers[{}].output",
                intermediate_stream_hook_name="language_model.model.layers[{}].post_attention_layernorm.output",
                residual_stream_input_post_layernorm_hook_name="language_model.model.layers[{}].self_attn.input",
                head_key_hook_name="language_model.model.layers[{}].self_attn.k_proj.output",
                head_value_hook_name="language_model.model.layers[{}].self_attn.v_proj.output",
                head_query_hook_name="language_model.model.layers[{}].self_attn.q_proj.output",
                attn_out_hook_name="language_model.model.layers[{}].self_attn.o_proj.output",
                attn_o_proj_input_hook_name="language_model.model.layers[{}].self_attn.o_proj.input",
                attn_in_hook_name="language_model.model.layers[{}].self_attn.input",
                attn_matrix_hook_name="language_model.model.layers[{}].self_attn.attention_matrix_hook.output",
                attn_matrix_pre_softmax_hook_name="language_model.model.layers[{}].self_attn.attention_matrix_pre_softmax_hook.output",
                mlp_out_hook_name="language_model.model.layers[{}].mlp.down_proj.output",
                last_layernorm_hook_name="language_model.model.norm.input",
                attn_out_proj_weight="language_model.model.layers[{}].self_attn.o_proj.weight",
                attn_out_proj_bias="language_model.model.layers[{}].self_attn.o_proj.bias",
                embed_tokens="language_model.model.layers[0].input",
                unembed_matrix="language_model.lm_head.weight",
                last_layernorm="language_model.model.norm",
                num_hidden_layers=model.language_model.config.num_hidden_layers,
                num_attention_heads=model.language_model.config.num_attention_heads,
                hidden_size=model.language_model.config.hidden_size,
                num_key_value_heads=model.language_model.config.num_key_value_heads,
                num_key_value_groups=model.language_model.config.num_attention_heads // model.language_model.config.num_key_value_heads,
                head_dim=model.language_model.config.head_dim,
                layernorm_type="RMS",
            )
            language_model = model.language_model


        elif model_name in ["hf-internal-testing/tiny-random-LlamaForCausalLM"]:
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
            )
            model_config = ModelConfig(
                residual_stream_input_hook_name="model.layers[{}].input",
                residual_stream_hook_name="model.layers[{}].output",
                intermediate_stream_hook_name="model.layers[{}].post_attention_layernorm.output",
                residual_stream_input_post_layernorm_hook_name="model.layers[{}].self_attn.input",
                head_query_hook_name="model.layers[{}].self_attn.q_proj.output",
                head_value_hook_name="model.layers[{}].self_attn.v_proj.output",
                head_key_hook_name="model.layers[{}].self_attn.k_proj.output",
                attn_out_hook_name="model.layers[{}].self_attn.o_proj.output",
                attn_o_proj_input_hook_name="model.layers[{}].self_attn.o_proj.input",
                attn_in_hook_name="model.layers[{}].self_attn.input",
                attn_matrix_hook_name="model.layers[{}].self_attn.attention_matrix_hook.output",
                attn_matrix_pre_softmax_hook_name="model.layers[{}].self_attn.attention_matrix_pre_softmax_hook.output",
                mlp_out_hook_name="model.layers[{}].mlp.down_proj.output",
                last_layernorm_hook_name="model.norm.input",
                attn_out_proj_weight="model.layers[{}].self_attn.o_proj.weight",
                attn_out_proj_bias="model.layers[{}].self_attn.o_proj.bias",
                embed_tokens="model.layers[0].input",
                unembed_matrix="lm_head.weight",
                last_layernorm="model.norm",
                num_hidden_layers=model.config.num_hidden_layers,
                num_attention_heads=model.config.num_attention_heads,
                hidden_size=model.config.hidden_size,
                num_key_value_heads=model.config.num_key_value_heads,
                num_key_value_groups=model.config.num_attention_heads
                // model.config.num_key_value_heads,
                head_dim=model.config.hidden_size // model.config.num_attention_heads,
                layernorm_type="RMS",
            )

        elif model_name in ["CohereForAI/aya-101"]:
            model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
            )
            language_model = None
            raise NotImplementedError("CohereForAI model not implemented yet")
            # model_config = ModelFactory._create_model_config(
            #     model.config, prefix="encoder."
            # )

        else:
            raise ValueError("Unsupported model_name")
        return model, language_model, model_config

    @staticmethod
    def _create_model_config(**kwargs):
        raise NotImplementedError("This method should be implemented in the if")

    # @staticmethod
    # def _create_model_config(model_config, prefix="model.",):
    #     return ModelConfig(
    #         residual_stream_input_hook_name=f"{prefix}layers[{{}}].input",
    #         residual_stream_hook_name=f"{prefix}layers[{{}}].output",
    #         intermediate_stream_hook_name=f"{prefix}layers[{{}}].post_attention_layernorm.output",
    #         residual_stream_input_post_layernorm_hook_name=f"{prefix}layers[{{}}].self_attn.input",
    #         head_value_hook_name=f"{prefix}layers[{{}}].self_attn.v_proj.output",
    #         attn_out_hook_name=f"{prefix}layers[{{}}].self_attn.o_proj.output",
    #         attn_in_hook_name=f"{prefix}layers[{{}}].self_attn.input",
    #         attn_matrix_hook_name=f"{prefix}layers[{{}}].self_attn.attention_matrix_hook.output",
    #         attn_out_proj_weight=f"{prefix}layers[{{}}].self_attn.o_proj.weight",
    #         attn_out_proj_bias=f"{prefix}layers[{{}}].self_attn.o_proj.bias",
    #         embed_tokens=f"{prefix}layers[0].input",
    #         unembed_matrix=f"{prefix}lm_head.weight",
    #         last_norm_module=
    #         num_hidden_layers=model_config.num_hidden_layers,
    #         num_attention_heads=model_config.num_attention_heads,
    #         hidden_size=model_config.hidden_size,
    #         num_key_value_heads=model_config.num_key_value_heads,
    #         num_key_value_groups=model_config.num_attention_heads // model_config.num_key_value_heads,
    #         head_dim=model_config.hidden_size // model_config.num_attention_heads,
    #     )


class TokenizerFactory:
    r"""
    This class return the right tokenizer for the model. If the model is multimodal return is_a_process == True
    """

    @staticmethod
    def load_tokenizer(model_name: str, torch_dtype: torch.dtype, device_map: str):
        r"""
        Load the tokenizer based on the model name.

        Args:
            model_name (str): Name of the model to load.
            torch_dtype (torch.dtype): Data type of the model.
            device_map (str): Device map for the model.

        Returns:
            processor (Tokenizer): Processor instance.
            is_a_processor (bool): True if the model is multimodal, False otherwise.
        """
        if model_name in ["facebook/chameleon-7b", "facebook/chameleon-30b"]:
            processor = ChameleonProcessor.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            is_a_processor = True
        elif model_name in ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"]:
            processor = LlamaTokenizerFast.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            is_a_processor = False
        elif model_name in ["google/gemma-3-1b-it", "google/gemma-3-4b-it", "google/gemma-3-12b-it", "google/gemma-3-27b-it"]:
            processor = Gemma3Processor.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            is_a_processor = True
        elif model_name in ["mistral-community/pixtral-12b"]:
            processor = PixtralProcessor.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            is_a_processor = True
        elif model_name in ["llava-hf/llava-v1.6-mistral-7b-hf"]:
            processor = LlavaNextProcessor.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            is_a_processor = True
        elif model_name in ["llava-hf/llava-onevision-qwen2-7b-ov-hf"]:
            processor = LlavaOnevisionProcessor.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            is_a_processor = True
        elif model_name in ["Emu3-Chat", "Emu3-Gen", "Emu3-Stage1"]:
            raise NotImplementedError("Emu3 model not implemented yet")
        elif model_name in ["hf-internal-testing/tiny-random-LlamaForCausalLM"]:
            processor = LlamaTokenizer.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            is_a_processor = False
        elif model_name in ["CohereForAI/aya-101"]:
            processor = T5TokenizerFast.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            is_a_processor = False

        else:
            raise ValueError("Unsupported model_name")

        return processor, is_a_processor


SUPPORTED_TOKENS = [
    "last",
    "last-2",
    "last-3",
    "last-image",
    "end-image",
    "all-image",
    "all-text",
    "all",
    "special",
    "random-text",
    "random-image",
    "random-image-10",
]


class InputHandler:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def prepare_inputs(
        self,
        batch_dict: Dict[str, torch.Tensor],
        device: Union[str, torch.device],
        torch_dtype: torch.dtype = torch.bfloat16,
        require_grads: bool = False,
    ):
        if self.model_name in [
            "facebook/chameleon-7b",
            "facebook/chameleon-30b",
        ]:
            if not "pixel_values" in batch_dict:
                input_dict = {
                    "input_ids": batch_dict["input_ids"],
                    "attention_mask": batch_dict["attention_mask"],
                }
            else:
                input_dict = {
                    "input_ids": batch_dict["input_ids"],
                    "attention_mask": batch_dict["attention_mask"],
                    "pixel_values": batch_dict["pixel_values"].to(torch_dtype),
                }
        elif self.model_name in ["mistral-community/pixtral-12b"]:
            if "pixel_values" not in batch_dict:
                input_dict = {
                    "input_ids": batch_dict["input_ids"],
                    "attention_mask": batch_dict["attention_mask"],
                }
            else:
                if (
                    isinstance(batch_dict["pixel_values"][0], list)
                    and len(batch_dict["pixel_values"]) == 1
                ):
                    # batch_dict["pixel_values"] = batch_dict["pixel_values"][0]
                    batch_dict["pixel_values"] = [
                        [
                            image.to(torch_dtype)
                            for image in batch_dict["pixel_values"][0]
                        ]
                    ]
                elif isinstance(batch_dict["pixel_values"], torch.Tensor):
                    batch_dict["pixel_values"] = batch_dict["pixel_values"].to(
                        torch_dtype
                    )
                elif isinstance(batch_dict["pixel_values"], list):
                    batch_dict["pixel_values"] = [
                        image.to(torch_dtype) for image in batch_dict["pixel_values"]
                    ]
                else:
                    raise ValueError("Pixel values not recognized. Please fix!")
                input_dict = {
                    "input_ids": batch_dict["input_ids"],
                    "attention_mask": batch_dict["attention_mask"],
                    "pixel_values": batch_dict["pixel_values"],
                    "image_sizes": batch_dict["image_sizes"]
                }

        elif self.model_name in ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"]:
            input_dict = {
                "input_ids": batch_dict["input_ids"],
                "attention_mask": batch_dict["attention_mask"],
            }
        elif self.model_name in ["Emu3-Chat", "Emu3-Gen", "Emu3-Stage1"]:
            raise NotImplementedError("Emu3 model not implemented yet")
        elif self.model_name in ["hf-internal-testing/tiny-random-LlamaForCausalLM"]:
            input_dict = {
                "input_ids": batch_dict["input_ids"],
                "attention_mask": batch_dict["attention_mask"],
            }
        elif self.model_name in ["llava-hf/llava-v1.6-mistral-7b-hf"]:
            if "pixel_values" not in batch_dict:
                input_dict = {
                    "input_ids": batch_dict["input_ids"],
                    "attention_mask": batch_dict["attention_mask"],
                }
            else:
                input_dict = {
                    "input_ids": batch_dict["input_ids"],
                    "attention_mask": batch_dict["attention_mask"],
                    "pixel_values": batch_dict["pixel_values"],
                    "image_sizes": batch_dict["image_sizes"],
                }
                
        elif self.model_name in ["llava-hf/llava-onevision-qwen2-7b-ov-hf"]:
            if "pixel_values" not in batch_dict:
                input_dict = {
                    "input_ids": batch_dict["input_ids"],
                    "attention_mask": batch_dict["attention_mask"],
                }
            else:
                input_dict = {
                    "input_ids": batch_dict["input_ids"],
                    "attention_mask": batch_dict["attention_mask"],
                    "pixel_values": batch_dict["pixel_values"],
                    "image_sizes": batch_dict["image_sizes"],
                }      
        elif self.model_name in ["google/gemma-3-1b-it", "google/gemma-3-4b-it", "google/gemma-3-12b-it", "google/gemma-3-27b-it"]:
            if "pixel_values" not in batch_dict:
                input_dict = {
                    "input_ids": batch_dict["input_ids"],
                    "attention_mask": batch_dict["attention_mask"],
                }
            else:
                input_dict = {
                    "input_ids": batch_dict["input_ids"],
                    "attention_mask": batch_dict["attention_mask"],
                    "pixel_values": batch_dict["pixel_values"],
                    "token_type_ids": batch_dict["token_type_ids"],
                }
        elif self.model_name in ["CohereForAI/aya-101"]:
            input_dict = {
                "input_ids": batch_dict["input_ids"],
                "decoder_input_ids": batch_dict["input_ids"],
                "attention_mask": batch_dict["attention_mask"],
            }
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}")

        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.to(device)
            elif isinstance(value, list):
                if isinstance(value[0], torch.Tensor):
                    value = [v.to(device) for v in value]
                elif isinstance(value[0], list):
                    value = [[v.to(device) for v in vv] for vv in value]
                else:
                    raise ValueError(
                        f"Problem while moving the input to the device. The input with key {key} is not a torch.Tensor, a list of torch.Tensor or a list of list of torch.Tensor."
                    )

            input_dict[key] = value
            
        if require_grads:
            input_dict["input_ids"].requires_grad = True
        return input_dict

    def get_input_ids(
        self,
        input_dict: Dict[str, torch.Tensor],
    ):
        return input_dict["input_ids"]

    def cleanup_tensors(self, inputs: Dict[str, Any], others: Dict[str, Any] = None):
        """
        Clean up tensors to free GPU memory by detaching, moving to CPU, and deleting.
        
        Args:
            inputs (Dict[str, Any]): Dictionary of input tensors (usually from prepare_inputs)
            others (Dict[str, Any], optional): Dictionary of other tensors not in inputs
            
        Returns:
            None
        """
        # Clean up inputs from the batch
        if inputs is not None:
            for k_in in list(inputs.keys()):  # Iterate over a copy of keys
                if isinstance(inputs[k_in], torch.Tensor):
                    inputs[k_in] = inputs[k_in].detach().cpu()
                del inputs[k_in]
            del inputs

        # Clean up others from the batch
        if others is not None:
            for k_oth in list(others.keys()):
                if isinstance(others[k_oth], torch.Tensor):
                    others[k_oth] = others[k_oth].detach().cpu()
                del others[k_oth]
            del others

        # Explicitly run garbage collection to free memory
        torch.cuda.empty_cache()
