from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Union, Optional, Tuple
import torch
from pydantic import BaseModel, Field
from enum import Enum
import re
from easyroutine.interpretability.models import ModelConfig
from easyroutine.interpretability.hooks import (
    intervention_attn_mat_hook,
    intervention_heads_hook,
    intervention_resid_hook,
    intervention_query_key_value_hook
)
from functools import partial
from dataclasses import dataclass


@dataclass
class Intervention:
    """
    User interface to define the intervention

    Arguments:
        - type: Literal["columns", "rows", "full", "block-img-txt", "block-img-img", "keep-self-attn"]: The type of intervention to be applied. "columns" will intervene on the columns of the attention matrix, "rows" will intervene on the rows of the attention matrix, "full" will intervene on the full attention matrix, "block-img-txt" will intervene on the block of image and text, "block-img-img" will intervene on the block of image and image, "keep-self-attn" will keep the self-attention of the model.
        - activation: str: The activation to be intervened. Should have the same format as returned from cache
        - token_positions: List[Union[str, int]]: The positions of the tokens that will be intervened or ablated
        - patching_values: Optional[Union[torch.Tensor, Literal["ablation"]]]: The values to be substituted during the intervention. If None or "ablation" the values will be set to zero.
    """

    type: Literal[
        "columns",
        "rows",
        "full",
        "block-img-txt",
        "block-img-img",
        "keep-self-attn",
        "grid",
        "columns_pre_softmax",
        "rows_pre_softmax",
        "grid_pre_softmax"
    ]
    activation: str
    token_positions: Union[List[Union[str, int]], Tuple[List[str], List[str]]]
    patching_values: Optional[Union[torch.Tensor, Literal["ablation"]]] = None
    multiplication_value: float = 0.0
    apply_softmax: bool = False

    def __getitem__(self, key):
        return getattr(self, key)


class InterventionConfig(BaseModel):
    """
    Essential information to apply the intervention

    Arguments:
        - hook_name: str: The name of the hook to be applied, should be the same as model_config
        - hook_func: Any: The function that will be applied to the hook
        - apply_intervention_func: Any: The function that is called to do the preliminary computation before attaching the hook to the model. This function is handled by the InterventionManager and is a pre-hook function. It useful since it will process the intervention data and return a clean hook_func that can be attached to the model.
    """

    hook_name: str
    hook_func: Any = None
    apply_intervention_func: Any = None
    head_dim: Optional[int] = None
    num_key_value_groups: Optional[int] = None
    num_attention_heads: Optional[int] = None

    def __getitem__(self, key):
        return getattr(self, key)


# from easyroutine.interpretability.ablation import (
#     columns_attn_mat,
#     intervention_attn_mat_full
# )


##
def columns_attn_mat(hook_name, intervention: Intervention, token_dict, intervention_config):
    """
    Pre-Hook function to compute the columns to be intervened in the attention matrix.
    """
    # compute the pre-hooks information and return the hook_func
    keys_intervention_token_position = []
    for token in intervention.token_positions:
        keys_intervention_token_position.extend(token_dict[token])

    queries_token_positions = [q for q in token_dict["all"]]
    try:
        layer = int(re.search(r"L(\d+)", intervention.activation).group(1))
        head = int(re.search(r"H(\d+)", intervention.activation).group(1))
    except AttributeError:
        raise ValueError(
            f"Activation {intervention['activation']} is not in the format pattern_L\d+H\d+"
        )

    return {
        "component": hook_name.format(layer),
        "intervention": partial(
            intervention_attn_mat_hook,
            q_positions=queries_token_positions,
            k_positions=keys_intervention_token_position,
            patching_values=intervention.patching_values,
            head=head,
            multiplication_value=intervention.multiplication_value,
            apply_softmax=intervention.apply_softmax,
        ),
    }


def rows_attn_mat(hook_name, intervention: Intervention, token_dict, intervention_config):
    """
    Pre-Hook function to compute the columns to be intervened in the attention matrix.
    """
    # compute the pre-hooks information and return the hook_func
    queries_token_positions = []
    for token in intervention.token_positions:
        queries_token_positions.extend(token_dict[token])

    keys_token_positions = [k for k in token_dict["all"]]
    try:
        layer = int(re.search(r"L(\d+)", intervention.activation).group(1))
        head = int(re.search(r"H(\d+)", intervention.activation).group(1))
    except AttributeError:
        raise ValueError(
            f"Activation {intervention['activation']} is not in the format pattern_L\d+H\d+"
        )

    return {
        "component": hook_name.format(layer),
        "intervention": partial(
            intervention_attn_mat_hook,
            q_positions=queries_token_positions,
            k_positions=keys_token_positions,
            patching_values=intervention.patching_values,
            head=head,
            multiplication_value=intervention.multiplication_value,
            apply_softmax=intervention.apply_softmax,
        ),
    }


def grid_attn_mat(hook_name, intervention, token_dict, intervention_config):
    try:
        assert (
            isinstance(intervention.token_positions, tuple)
            and len(intervention.token_positions) == 2
        )
        try:
            assert isinstance(intervention.token_positions[0], list) and isinstance(
                intervention.token_positions[1], list
            )
        except AssertionError:
            raise ValueError(
                f"Intervention token_positions should be a tuple of two lists if intervening on pattern with two grid, got {type(intervention.token_positions)}"
            )
    except AssertionError:
        raise ValueError(
            f"Intervention token_positions should be a tuple of two lists if intervening on pattern with two grid, got {type(intervention.token_positions)}"
        )

    queries_token_positions = []
    for token in intervention.token_positions[0]:
        if isinstance(token, str) and token in token_dict:
            queries_token_positions.extend(token_dict[token])
        elif isinstance(token, int):
            # If the token is an int, add it directly to the target positions
            queries_token_positions.append(token)
        else:
            raise ValueError(
                f"Token {token} is not in the token_dict and is not an int, got {type(token)}"
            )


    keys_token_positions = []
    for token in intervention.token_positions[1]:
        if isinstance(token, str) and token in token_dict:
            keys_token_positions.extend(token_dict[token])
        elif isinstance(token, int):
            # If the token is an int, add it directly to the target positions
            keys_token_positions.append(token)
        else:
            raise ValueError(
                f"Token {token} is not in the token_dict and is not an int, got {type(token)}"
            )
    try:
        layer = int(re.search(r"L(\d+)", intervention.activation).group(1))
        head = int(re.search(r"H(\d+)", intervention.activation).group(1))
    except AttributeError:
        raise ValueError(
            f"Activation {intervention['activation']} is not in the format pattern_L\d+H\d+"
        )

    return {
        "component": hook_name.format(layer),
        "intervention": partial(
            intervention_attn_mat_hook,
            q_positions=queries_token_positions,
            k_positions=keys_token_positions,
            patching_values=intervention.patching_values,
            head=head,
            multiplication_value=intervention.multiplication_value,
            apply_softmax=intervention.apply_softmax,
        ),
    }


def block_img_txt_attn_mat(hook_name, intervention, token_dict):
    # compute the pre-hooks information and return the hook_func
    pass


def intervention_resid_full(hook_name, intervention, token_dict, intervention_config):
    # compute the pre-hooks information and return the hook_func
    target_positions = []
    for token in intervention["token_positions"]:
        if isinstance(token, str) and token in token_dict:
            # If the token is a string, get the positions from the token_dict
            target_positions.extend(token_dict[token])
        elif isinstance(token, int):
            # If the token is an int, add it directly to the target positions
            target_positions.append(token)

    # get the integer layer number from teh activation string resid_out_L\d+ or resid_in_L\d+ or resid_mid_L\d+
    layer = int(
        re.search(r"(\d+)", intervention.activation).group(1)
    )
    
    return {
        "component": hook_name.format(layer),
        "intervention": partial(
            intervention_resid_hook,
            token_indexes=target_positions,
            patching_values=intervention["patching_values"],
        ),
    }

def intervention_query_key_value(hook_name, intervention, token_dict, intervention_config):
    """
    Pre-Hook function to compute the values to be intervened in the attention matrix.
    """
    # compute the pre-hooks information and return the hook_func
    target_positions = []
    for token in intervention["token_positions"]:
        if isinstance(token, str) and token in token_dict:
            # If the token is a string, get the positions from the token_dict
            target_positions.extend(token_dict[token])
        elif isinstance(token, int):
            # If the token is an int, add it directly to the target positions
            target_positions.append(token)

    # get the integer layer number from teh activation string resid_out_L\d+ or resid_in_L\d+ or resid_mid_L\d+
    layer = int(
        re.search(r"(\d+)", intervention.activation).group(1)
    )
    head = int(
        re.search(r"H(\d+)", intervention.activation).group(1)
    )
    
    return {
        "component": hook_name.format(layer),
        "intervention": partial(
            intervention_query_key_value_hook,
            token_indexes=target_positions,
            head=head,
            head_dim=intervention_config.head_dim,
            num_key_value_groups = intervention_config.num_key_value_groups,
            num_attention_heads = intervention_config.num_attention_heads,
            patching_values=intervention["patching_values"],
        ),
    }

class InterventionManager:
    """
    Class to manage the interventions (ablation, patching, etc) on the model

    User should define intervention object

    Arguments:
        - model_config: ModelConfig: The configuration of the model
    """

    def __init__(
        self,
        model_config: ModelConfig,
    ):
        self.model_config = model_config

    def create_intervention_hooks(
        self, interventions: List[Intervention], token_dict: dict
    ):
        """
        Function that given a list of interventions, returns a list of hooks to be applied to the model.

        Arguments:
            - interventions: List[Intervention]. The list of interventions to be applied to the model
            - token_dict: dict. The dictionary containing the token positions in the model

        Returns:
            - List[Dict[str, Any]]: The list of hooks to be applied to the model
        """
        self._register_interventions()  # Register the interventions. Here to support dynamical model_config changes

        hooks = []
        for intervention in interventions:
            type_str = intervention["activation"]
            intervention_type = intervention["type"]

            # Find the matching regex key from supported_interventions
            matched_config = None
            for pattern, config_dict in self.supported_interventions.items():
                if pattern.match(type_str):
                    matched_config = config_dict
                    break
            if matched_config is None:
                raise ValueError(
                    f"No supported intervention found for activation type {type_str}"
                )

            # Check if the intervention_type is supported for that regex key.
            if intervention_type not in matched_config:
                raise ValueError(
                    f"Intervention type {intervention_type} is not supported for activation {type_str}"
                )

            intervention_config = matched_config[intervention_type]
            hook = intervention_config.apply_intervention_func(
                hook_name=intervention_config.hook_name,
                intervention=intervention,
                token_dict=token_dict,
                intervention_config = intervention_config,
                
                # Pass additional parameters if needed
                # e.g., num_key_value_groups, num_attention_heads, etc.
                # head_dim=intervention_config.head_dim,
            )
            hooks.append(hook)
        return hooks

    def _register_interventions(self):
        """
        Function to register the interventions supported by the model.
        """
        self.supported_interventions = {
            re.compile(r"pattern_L\d+H\d+"): {
                "columns": InterventionConfig(
                    hook_name=self.model_config.attn_matrix_hook_name,
                    hook_func=intervention_attn_mat_hook,
                    apply_intervention_func=columns_attn_mat,
                ),
                "rows": InterventionConfig(
                    hook_name=self.model_config.attn_matrix_hook_name,
                    hook_func=intervention_attn_mat_hook,
                    apply_intervention_func=rows_attn_mat,
                ),
                "grid": InterventionConfig(
                    hook_name=self.model_config.attn_matrix_hook_name,
                    hook_func=intervention_attn_mat_hook,
                    apply_intervention_func=grid_attn_mat,
                ),
                "block_img_txt": InterventionConfig(
                    hook_name=self.model_config.attn_matrix_hook_name,
                    hook_func=intervention_attn_mat_hook,
                    apply_intervention_func=block_img_txt_attn_mat,
                ),
                "colums_pre_softmax": InterventionConfig(
                    hook_name=self.model_config.attn_matrix_pre_softmax_hook_name,
                    hook_func=intervention_attn_mat_hook,
                    apply_intervention_func=columns_attn_mat,
                ),
                "rows_pre_softmax": InterventionConfig(
                    hook_name=self.model_config.attn_matrix_pre_softmax_hook_name,
                    hook_func=intervention_attn_mat_hook,
                    apply_intervention_func=rows_attn_mat,
                ),
                "grid_pre_softmax": InterventionConfig(
                    hook_name=self.model_config.attn_matrix_pre_softmax_hook_name,
                    hook_func=intervention_attn_mat_hook,
                    apply_intervention_func=grid_attn_mat,
                ),
                "block_img_txt_pre_softmax": InterventionConfig(
                    hook_name=self.model_config.attn_matrix_pre_softmax_hook_name,
                    hook_func=intervention_attn_mat_hook,
                    apply_intervention_func=block_img_txt_attn_mat,
                ),
            },
            # Residual stream interventions
            re.compile(r"resid_out_\d+"): {
                "full": InterventionConfig(
                    hook_name=self.model_config.residual_stream_hook_name,
                    hook_func=intervention_resid_hook,
                    apply_intervention_func=intervention_resid_full,
                )
            },
            re.compile(r"resid_in_\d+"): {
                "full": InterventionConfig(
                    hook_name=self.model_config.residual_stream_input_hook_name,
                    hook_func=intervention_resid_hook,
                    apply_intervention_func=intervention_resid_full,
                )
            },
            re.compile(r"resid_mid_\d+"): {
                "full": InterventionConfig(
                    hook_name=self.model_config.intermediate_stream_hook_name,
                    hook_func=intervention_resid_hook,
                    apply_intervention_func=intervention_resid_full,
                )
            },
            # Attention input and output interventions
            re.compile(r"attn_in_\d+"): {
                "full": InterventionConfig(
                    hook_name=self.model_config.attn_in_hook_name,
                    hook_func=intervention_resid_hook,
                    apply_intervention_func=intervention_resid_full,
                )
            },
            re.compile(r"attn_out_\d+"): {
                "full": InterventionConfig(
                    hook_name=self.model_config.attn_out_hook_name,
                    hook_func=intervention_resid_hook,
                    apply_intervention_func=intervention_resid_full,
                )
            },
            # MLP output interventions
            re.compile(r"mlp_out_\d+"): {
                "full": InterventionConfig(
                    hook_name=self.model_config.mlp_out_hook_name,
                    hook_func=intervention_resid_hook,
                    apply_intervention_func=intervention_resid_full,
                )
            },
            # Head component interventions
            re.compile(r"values_L\d"): {
                "full": InterventionConfig(
                    hook_name=self.model_config.head_value_hook_name,
                    hook_func=intervention_resid_hook,
                    apply_intervention_func=intervention_resid_full,
                )
            },
            re.compile(r"keys_L\d"): {
                "full": InterventionConfig(
                    hook_name=self.model_config.head_key_hook_name,
                    hook_func=intervention_resid_hook,
                    apply_intervention_func=intervention_resid_full,
                )
            },
            re.compile(r"queries_L\d"): {
                "full": InterventionConfig(
                    hook_name=self.model_config.head_query_hook_name,
                    hook_func=intervention_resid_hook,
                    apply_intervention_func=intervention_resid_full,
                )
            },
            re.compile(r"head_values_L\d+H\d"): {
                "full": InterventionConfig(
                    hook_name=self.model_config.head_value_hook_name,
                    hook_func=intervention_query_key_value,
                    apply_intervention_func=intervention_query_key_value,
                    head_dim=self.model_config.head_dim,
                    num_key_value_groups=self.model_config.num_key_value_groups,
                    num_attention_heads=self.model_config.num_attention_heads,
                )
            },
            re.compile(r"head_keys_L\d+H\d+"): {
                "full": InterventionConfig(
                    hook_name=self.model_config.head_key_hook_name,
                    hook_func=intervention_query_key_value,
                    apply_intervention_func=intervention_query_key_value,
                    head_dim=self.model_config.head_dim,
                    num_key_value_groups=self.model_config.num_key_value_groups,
                    num_attention_heads=self.model_config.num_attention_heads,
                )
            },
            re.compile(r"head_queries_L\d+H\d+"): {
                "full": InterventionConfig(
                    hook_name=self.model_config.head_query_hook_name,
                    hook_func=intervention_query_key_value,
                    apply_intervention_func=intervention_query_key_value,
                    head_dim=self.model_config.head_dim,
                    num_key_value_groups=self.model_config.num_key_value_groups,
                    num_attention_heads=self.model_config.num_attention_heads,
                )
            },
            # Projected vectors interventions
            re.compile(r"projected_value_L\d+H\d+"): {
                "full": InterventionConfig(
                    hook_name=self.model_config.head_value_hook_name,
                    hook_func=intervention_resid_hook,
                    apply_intervention_func=intervention_resid_full,
                )
            },
            re.compile(r"projected_key_L\d+H\d+"): {
                "full": InterventionConfig(
                    hook_name=self.model_config.head_key_hook_name,
                    hook_func=intervention_resid_hook,
                    apply_intervention_func=intervention_resid_full,
                )
            },
            re.compile(r"projected_query_L\d+H\d+"): {
                "full": InterventionConfig(
                    hook_name=self.model_config.head_query_hook_name,
                    hook_func=intervention_resid_hook,
                    apply_intervention_func=intervention_resid_full,
                )
            },
            # Head output interventions
            re.compile(r"head_out_L\d+H\d+"): {
                "full": InterventionConfig(
                    hook_name=self.model_config.attn_o_proj_input_hook_name,
                    hook_func=intervention_resid_hook,
                    apply_intervention_func=intervention_resid_full,
                )
            },
        }
