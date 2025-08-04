# Description: This file contains the hooks used to extract the activations of the model
import torch
from copy import deepcopy
import pandas as pd
from typing import List, Callable, Union, Literal, Optional
from einops import rearrange, einsum
from easyroutine.interpretability.utils import repeat_kv, get_module_by_path
from easyroutine.interpretability.activation_cache import ActivationCache
from easyroutine.logger import logger
from functools import partial
import re
import torch.nn as nn
import einops
import itertools


def process_args_kwargs_output(args, kwargs, output):
    """
    Extract the main tensor from output, args, or kwargs.
    Prioritizes output (first element if tuple), then first arg, then kwargs['hidden_states'] if present.
    Args:
        args: Positional arguments from the hook.
        kwargs: Keyword arguments from the hook.
        output: Output from the hooked function.
    Returns:
        The main tensor to be processed by the hook.
    """
    if output is not None:
        if isinstance(output, tuple):
            b = output[0]
        else:
            b = output
    else:
        if len(args) > 0:
            b = args[0]
        else:
            candidate_keys = ["hidden_states"]
            for key in candidate_keys:
                if key in kwargs:
                    b = kwargs[key]
                    break
    return b  # type:ignore


def restore_same_args_kwargs_output(b, args, kwargs, output):
    """
    Restore the structure of output, args, and kwargs after modification.
    Args:
        b: The new tensor to insert.
        args: Original positional arguments.
        kwargs: Original keyword arguments.
        output: Original output from the hooked function.
    Returns:
        The updated output, or (args, kwargs) if output is None.
    """

    if output is not None:
        if isinstance(output, tuple):
            b = (b,) + output[1:]
    if output is None:
        if len(args) > 0:
            args = (b,) + args[1:]
        else:
            candidate_keys = ["hidden_states"]
            for key in candidate_keys:
                if key in kwargs:
                    kwargs[key] = b
                    break
        return args, kwargs
    return b  # type:ignore


# 2. Retrieving the module


def create_dynamic_hook(pyvene_hook: Callable, **kwargs):
    r"""
    DEPRECATED: pyvene is not used anymore.
    This function is used to create a dynamic hook. It is a wrapper around the pyvene_hook function.
    """
    partial_hook = partial(pyvene_hook, **kwargs)

    def wrap(*args, **kwargs):
        return partial_hook(*args, **kwargs)

    return wrap


# def embed_hook(module, input, output, cache, cache_key):
#     r"""
#     Hook function to extract the embeddings of the tokens. It will save the embeddings in the cache (a global variable out the scope of the function)
#     """
#     if output is None:
#         b = input[0]
#     else:
#         b = output
#     cache[cache_key] = b.data.detach().clone()
#     return b


def embed_hook(module, args, kwargs, output, token_indexes, cache, cache_key):
    r"""
    Hook function to extract the embeddings of the specified tokens and save them in the cache.
    Args:
        module: The module being hooked.
        args: Positional arguments.
        kwargs: Keyword arguments.
        output: Output from the module.
        token_indexes: List of token indexes to extract.
        cache: The cache object to store results.
        cache_key: The key under which to store the embeddings.
    """
    b = process_args_kwargs_output(args, kwargs, output)
    cache[cache_key] = []
    for token_index in token_indexes:
        cache[cache_key].append(b.data.detach().clone()[..., list(token_index)])
    cache[cache_key] = tuple(cache[cache_key])
    # cache[cache_key] = b.data.detach().clone()


def compute_statistics(tensor, dim=-1, keepdim=True, eps=1e-6):
    """
    Computes the mean, variance, and second moment of a given tensor along a specified dimension.

    Args:
        tensor (torch.Tensor): Input tensor.
        dim (int): Dimension along which to compute statistics (default: -1).
        keepdim (bool): Whether to keep the reduced dimension (default: True).
        eps (float): Small constant for numerical stability.

    Returns:
        tuple: (mean, variance, second_moment)
    """
    mean = tensor.mean(dim=dim, keepdim=keepdim)  # Compute mean
    second_moment = tensor.pow(2).mean(
        dim=dim, keepdim=keepdim
    )  # Compute second moment
    variance = second_moment - mean.pow(2)  # Compute variance using E[X²] - (E[X])²

    return mean.squeeze(-1), variance.squeeze(-1), second_moment.squeeze(-1)


def layernom_hook(
    module, args, kwargs, output, token_indexes, cache, cache_key, avg: bool = False
):
    """
    Compute and save mean, variance, and second moment for the specified token indexes.
    If avg is True, computes statistics for each token group; otherwise, flattens indexes.
    Args:
        module: The module being hooked.
        args: Positional arguments.
        kwargs: Keyword arguments.
        output: Output from the module.
        token_indexes: List of token index groups.
        cache: The cache object to store results.
        cache_key: The key under which to store the statistics.
        avg: Whether to average over each token group separately.
    """
    b = process_args_kwargs_output(args, kwargs, output)
    if avg:
        token_avgs = []
        for token_index in token_indexes:
            slice_ = b.data.detach().clone()[..., list(token_index), :]
            mean, variance, second_moment = compute_statistics(slice_)
            token_avgs.append(
                {"mean": mean, "variance": variance, "second_moment": second_moment}
            )
        cache[cache_key] = token_avgs
    flatten_indexes = [item for sublist in token_indexes for item in sublist]
    mean, variance, second_moment = compute_statistics(b[..., flatten_indexes, :])
    cache[cache_key] = {
        "mean": mean,
        "variance": variance,
        "second_moment": second_moment,
    }


# Define a hook that saves the activations of the residual stream
def save_resid_hook(
    module,
    args,
    kwargs,
    output,
    cache: ActivationCache,
    cache_key,
    token_indexes,
    avg: bool = False,
):
    r"""
    Save the activations of the residual stream for the specified token indexes in the cache.
    If avg is True, saves averaged activations for each token group.
    Args:
        module: The module being hooked.
        args: Positional arguments.
        kwargs: Keyword arguments.
        output: Output from the module.
        cache: The cache object to store results.
        cache_key: The key under which to store the activations.
        token_indexes: List of token index groups.
        avg: Whether to average over each token group separately.
    """
    b = process_args_kwargs_output(args, kwargs, output)

    # slice the tensor to get the activations of the token we want to extract
    if avg:
        token_avgs = []
        for token_index in token_indexes:
            slice_ = b.data.detach().clone()[..., list(token_index), :]
            token_avgs.append(torch.mean(slice_, dim=-2, keepdim=True))

        # cache[cache_key] = torch.cat(token_avgs, dim=-2)
        cache.add_with_info(
            cache_key,
            torch.cat(token_avgs, dim=-2),
            "Shape: batch avg_over_target_token_position, d_model",
        )

    else:
        flatten_indexes = [item for sublist in token_indexes for item in sublist]
        cache[cache_key] = b.data.detach().clone()[..., flatten_indexes, :]


def intervention_resid_hook(
    module,
    args,
    kwargs,
    output,
    token_indexes,
    patching_values: Optional[Union[str, torch.Tensor]] = None,
):
    r"""
    Hook function to ablate the tokens in the residual stream. It will set to 0 the value vector of the
    tokens to ablate
    """
    b = process_args_kwargs_output(args, kwargs, output)
    # detach b to avoid modifying the original tensor
    b = b.data.detach().clone()
    if patching_values is None or patching_values == "ablation":
        logger.debug(
            "No patching values provided, ablation will be performed on the residual stream"
        )
        b[..., token_indexes, :] = 0
    else:
        logger.debug(
            "Patching values provided, applying patching values to the residual stream"
        )
        assert b[..., list(token_indexes), :].shape == patching_values.shape, (
            f"Shape mismatch: activations is {b[..., list(token_indexes), :].shape} but patching values is {patching_values.shape}"
        )
        b[..., list(token_indexes), :] = patching_values
    return restore_same_args_kwargs_output(b, args, kwargs, output)


def query_key_value_hook(
    module,
    args,
    kwargs,
    output,
    cache: ActivationCache,
    cache_key,
    token_indexes,
    layer,
    head_dim,
    num_key_value_groups: int,
    num_attention_heads: int,
    head: Union[str, int] = "all",
    avg: bool = False,
):
    r"""
    Same as save_resid_hook but for the query, key and value vectors, it just have a reshape to have the head dimension.
    """
    b = process_args_kwargs_output(args, kwargs, output)
    input_shape = b.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    b = b.view(hidden_shape).transpose(1, 2)
    # cache[cache_key] = b.data.detach().clone()[..., token_index, :]
    if (
        num_key_value_groups > 1 and b.size(1) != num_attention_heads
    ):  # we are in kv group attention
        # we need to repeat the key and value states
        b = repeat_kv(b, num_attention_heads // b.size(1))

    # check if we are using kv group attention

    info_string = "Shape: batch seq_len d_head"

    heads = [idx for idx in range(b.size(1))] if head == "all" else [head]
    for head_idx in heads:
        # Compute the group index for keys/values if needed.
        group_idx = head_idx // (b.size(1) // num_key_value_groups)
        # Decide whether to use group_idx or head_idx based on cache_key.
        if "head_values_" in cache_key or "head_keys_" in cache_key:
            # Select the slice corresponding to the group index.
            tensor_slice = b.data.detach().clone()[:, group_idx, ...]
        else:
            # Use the head index directly.
            tensor_slice = b.data.detach().clone()[:, head_idx, ...]

        # Process the token indexes.
        if avg:
            # For each token tuple, average over the tokens.
            # Note: After slicing, the token dimension is the first dimension of tensor_slice,
            # i.e. tensor_slice has shape (batch, tokens, d_head) so we average along dim=1.
            tokens_avgs = []
            for token_tuple in token_indexes:
                # Slice tokens using the token_tuple.
                token_subslice = tensor_slice[:, list(token_tuple), :]
                # Average over the token dimension (dim=1) and keep that dimension.
                token_avg = torch.mean(token_subslice, dim=1, keepdim=True)
                tokens_avgs.append(token_avg)
            # Concatenate the averages along the token dimension (dim=1).
            processed_tokens = torch.cat(tokens_avgs, dim=1)
        else:
            # Flatten the token indexes from the list of tuples.
            flatten_indexes = [item for tup in token_indexes for item in tup]
            processed_tokens = tensor_slice[:, flatten_indexes, :]

        # Build a unique key for the cache by including layer and head information.
        key = f"{cache_key}L{layer}H{head_idx}"
        cache.add_with_info(key, processed_tokens, info_string)


def intervention_query_key_value_hook(
    module,
    args,
    kwargs,
    output,
    token_indexes,
    head,
    head_dim,
    num_key_value_groups: int,
    num_attention_heads: int,
    patching_values: Optional[Union[str, torch.Tensor]] = None,
):
    r"""
    Hook function to intervene on the query, key and value vectors. It first unpack the vectors from the output of the module and then apply the intervention and then repack the vectors.
    """
    b = process_args_kwargs_output(args, kwargs, output)
    # input_shape = b.shape[-1]
    # hidden_shape = (*input_shape, -1, head_dim)
    hidden_shape = b.shape
    # b = b.view(hidden_shape).transpose(1, 2)

    if (
        num_key_value_groups > 1 and b.size(-1) < num_attention_heads * head_dim
    ):  # we are in kv group attention
        b = einops.rearrange(
            b,
            "batch seq_len (num_attention_heads head_dim) -> batch num_attention_heads seq_len head_dim",
            num_attention_heads=num_attention_heads // num_key_value_groups,
            head_dim=head_dim,
        )
    # check if we are using kv group attention
    else:
        b = einops.rearrange(
            b,
            "batch seq_len (num_attention_heads head_dim) -> batch num_attention_heads seq_len head_dim",
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
        )

    # tensor_head = b.data.detach().clone()[:, head, ...]
    # Apply the intervention
    if patching_values is None or patching_values == "ablation":
        logger.debug(
            "No patching values provided, ablation will be performed on the query, key and value vectors"
        )
        b[:, head, token_indexes, :] = 0
    else:
        logger.debug(
            "Patching values provided, applying patching values to the query, key and value vectors"
        )
        b[:, head, list(token_indexes), :] = patching_values

    # Repack the vectors
    b = einops.rearrange(
        b,
        "batch num_attention_heads seq_len head_dim -> batch seq_len (num_attention_heads head_dim)",
    )

    # Restore the args and kwargs
    b = restore_same_args_kwargs_output(b, args, kwargs, output)
    return b


def avg_hook(
    module,
    args,
    kwargs,
    output,
    cache,
    cache_key,
    last_image_idx,
    end_image_idx,
):
    r"""
    It save the activations of the residual stream in the cache. It will save the activations in the cache (a global variable out the scope of the function)
    """
    b = process_args_kwargs_output(args, kwargs, output)

    img_avg = torch.mean(
        b.data.detach().clone()[:, 1 : last_image_idx + 1, :],
        dim=1,
    )
    text_avg = torch.mean(b.data.detach().clone()[:, end_image_idx:, :], dim=1)
    all_avg = torch.mean(b.data.detach().clone()[:, :, :], dim=1)

    cache[f"avg_{cache_key}"] = torch.cat(
        [img_avg.unsqueeze(1), text_avg.unsqueeze(1), all_avg.unsqueeze(1)], dim=1
    )


def head_out_hook(
    module,
    args,
    kwargs,
    output,
    cache,
    cache_key,
    token_indexes,
    layer,
    num_heads,
    head_dim,
    o_proj_weight,
    o_proj_bias,
    head: Union[str, int] = "all",
    avg: bool = False,
):
    b = process_args_kwargs_output(args, kwargs, output)

    bsz, seq_len, hidden_size = b.shape
    b = b.view(bsz, seq_len, num_heads, head_dim)
    # reshape the weights to have the head dimension
    o_proj_weight = einops.rearrange(
        o_proj_weight,
        "d_model (num_heads head_dim) -> num_heads head_dim d_model",
        num_heads=num_heads,
        head_dim=head_dim,
    )

    # apply the projection
    projected_values = einsum(
        b,
        o_proj_weight,
        "batch seq_len num_head d_head, num_head d_head d_model -> batch seq_len num_head d_model",
    )

    if o_proj_bias is not None:
        projected_values = projected_values + o_proj_bias // num_heads

    # Process token indexes from projected_values of shape [batch, tokens, num_heads, d_model]
    if avg:
        # For each tuple, slice out the corresponding tokens and average over them.
        token_avgs = []
        for token_tuple in token_indexes:
            # Slice out the tokens specified by the tuple.
            token_slice = projected_values[:, list(token_tuple), :, :]
            # Average over the token dimension (dim=1) and keep that dimension.
            token_avg = torch.mean(token_slice, dim=1, keepdim=True)
            token_avgs.append(token_avg)
        # Concatenate the per-tuple averages along the token dimension.
        projected_values = torch.cat(token_avgs, dim=1)
    else:
        # Flatten the list of tuples into a single list of token indexes.
        flatten_indexes = [item for tup in token_indexes for item in tup]
        projected_values = projected_values[:, flatten_indexes, :, :]

    # Determine the heads to process.
    heads = list(range(num_heads)) if head == "all" else [head]
    for head_idx in heads:
        cache.add_with_info(
            f"{cache_key}L{layer}H{head_idx}",
            # Select the corresponding head (axis 2).
            projected_values[:, :, int(head_idx), :],
            "Shape: batch selected_inputs_ids_len, d_model",
        )


def multiply_pattern(tensor, multiplication_value):
    r"""
    Set the attention values to zero
    """
    # return torch.zeros_like(tensor) + multiplication_value
    return tensor * multiplication_value


# b.copy_(attn_matrix)
def intervention_attn_mat_hook(
    module,
    args,
    kwargs,
    output,
    q_positions,
    k_positions,
    head,
    multiplication_value,
    patching_values: Optional[Union[str, torch.Tensor]] = None,
    apply_softmax: bool = False,
    # ablation_queries: pd.DataFrame,
):
    r"""
    Hook function to ablate the tokens in the attention
    mask. It will set to 0 the value vector of the
    tokens to ablate
    """
    # Get the shape of the attention matrix
    b = process_args_kwargs_output(args, kwargs, output)
    batch_size, num_heads, seq_len_q, seq_len_k = b.shape

    # Used during generation
    if seq_len_q < len(q_positions):
        q_positions = 0

    # Create boolean masks for queries and keys
    q_mask = torch.zeros(seq_len_q, dtype=torch.bool, device=b.device)
    q_mask[q_positions] = True  # Set positions to True

    k_mask = torch.zeros(seq_len_k, dtype=torch.bool, device=b.device)
    k_mask[k_positions] = True  # Set positions to TrueW

    # Create a 2D mask using outer product
    head_mask = torch.outer(q_mask, k_mask)  # Shape: (seq_len_q, seq_len_k)

    # Expand mask to match the dimensions of the attention matrix
    # Shape after expand: (batch_size, num_heads, seq_len_q, seq_len_k)
    # head_mask = (
    #     head_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
    # )

    # select the head
    # head_mask = head_mask[:, head, :, :]

    if patching_values is None or patching_values == "ablation":
        logger.debug("No patching values provided, ablation will be performed")
        # Apply the ablation function directly to the attention matrix
        b[:, head, head_mask] = multiply_pattern(
            b[:, head, head_mask], multiplication_value
        )

    else:
        # Apply the patching values to the attention matrix
        logger.debug("Patching values provided, applying patching values")
        logger.debug(
            "Patching values shape: %s. It is expected to have shape seq_len x seq_len",
            patching_values.shape,
        )

        b[:, head, head_mask] = patching_values[head_mask]
    if apply_softmax:
        b[:, head] = torch.nn.functional.softmax(b[:, head], dim=-1)
    return b


# def ablate_tokens_hook_flash_attn(
#     module,
#     args,
#     kwargs,
#     output,
#     ablation_queries: pd.DataFrame,
#     num_layers: int = 32,
# ):
#     r"""
#     same of ablate_tokens_hook but for flash attention. This apply the ablation on the values vectors instead of the attention mask
#     """
#     b = process_args_kwargs_output(args, kwargs, output)
#     batch_size, seq, d_model = b.shape
#     if seq == 1:
#         return b
#     values = b.clone().data
#     device = values.device

#     ablation_queries.reset_index(
#         drop=True, inplace=True
#     )  # Reset index to avoid problems with casting to tensor
#     head_indices = torch.tensor(
#         ablation_queries["head"], dtype=torch.long, device=device
#     )
#     pos_indices = torch.tensor(
#         ablation_queries["keys"], dtype=torch.long, device=device
#     )
#     # if num_layers != len(head_indices) or not torch.all(pos_indices == pos_indices[0]) :
#     #     raise ValueError("Flash attention ablation should be done on all heads at the same layer and at the same token position")
#     # if seq < pos_indices[0]:
#     #     # during generation the desired value vector has already been ablated
#     #     return b
#     pos_indices = pos_indices[0]
#     # Use advanced indexing to set the specified slices to zero
#     values[..., pos_indices, :] = 0

#     b.copy_(values)

#     #!!dirty fix

#     return b


def intervention_heads_hook(
    module,
    args,
    kwargs,
    output,
    ablation_queries: pd.DataFrame,
):
    r"""
    Hook function to ablate the heads in the attention
    mask. It will set to 0 the output of the heads to
    ablate
    """
    b = process_args_kwargs_output(args, kwargs, output)
    attention_matrix = b.clone().data

    for head in ablation_queries["head"]:
        attention_matrix[0, head, :, :] = 0

    b.copy_(attention_matrix)
    return b


def ablate_pos_keep_self_attn_hook(
    module,
    args,
    kwargs,
    output,
    ablation_queries: pd.DataFrame,
):
    r"""
    Hook function to ablate the tokens in the attention
    mask but keeping the self attn weigths.
    It will set to 0 the row of tokens to ablate except for
    the las position
    """
    b = process_args_kwargs_output(args, kwargs, output)
    Warning("This function is deprecated. Use ablate_attn_mat_hook instead")
    attn_matrix = b.data
    # initial_shape = attn_matrix.shape

    for head, pos in zip(
        ablation_queries["head"], ablation_queries["pos_token_to_ablate"]
    ):
        attn_matrix[0, head, pos, :-1] = 0

    b.copy_(attn_matrix)

    return b


def ablate_tokens_hook(*args, **kwargs):
    raise NotImplementedError(
        "This function will be discaderd keeping only for backward compatibility"
    )


def ablate_images_hook(*args, **kwargs):
    raise NotImplementedError(
        "This function will be discaderd keeping only for backward compatibility"
    )


def ablate_image_image_hook(*args, **kwargs):
    raise NotImplementedError(
        "This function will be discaderd keeping only for backward compatibility"
    )


def projected_value_vectors_head(
    module,
    args,
    kwargs,
    output,
    layer,
    cache,
    token_indexes,
    num_attention_heads: int,
    num_key_value_heads: int,
    hidden_size: int,
    d_head: int,
    out_proj_weight,
    out_proj_bias,
    head: Union[str, int] = "all",
    act_on_input=False,
    expand_head: bool = True,
    avg=False,
):
    r"""
    Hook function to extract the values vectors of the heads. It will extract the values vectors and then project them with the final W_O projection
    As the other hooks, it will save the activations in the cache (a global variable out the scope of the function)

    Args:
        b: the input of the hook function. It's the output of the values vectors of the heads
        s: the state of the hook function. It's the state of the model
        layer: the layer of the model
        head: the head of the model. If "all" is passed, it will extract all the heads of the layer
        expand_head: bool to expand the head dimension when extracting the values vectors and the attention pattern. If true, in the cache we will have a key for each head, like "value_L0H0", "value_L0H1", ...
                        while if False, we will have only one key for each layer, like "value_L0" and the dimension of the head will be taken into account in the tensor.

    """
    # first get the values vectors
    b = process_args_kwargs_output(args, kwargs, output)

    values = b.data.detach().clone()  # (batch, num_heads,seq_len, head_dim)

    # reshape the values vectors to have a separate dimension for the different heads
    values = rearrange(
        values,
        "batch seq_len (num_key_value_heads d_heads) -> batch num_key_value_heads seq_len d_heads",
        num_key_value_heads=num_key_value_heads,
        d_heads=d_head,
    )

    #        "batch seq_len (num_key_value_heads d_heads) -> batch seq_len num_key_value_heads d_heads",

    values = repeat_kv(values, num_attention_heads // num_key_value_heads)

    values = rearrange(
        values,
        "batch num_head seq_len d_model -> batch seq_len num_head d_model",
    )

    # reshape in order to get the blocks for each head
    out_proj_weight = out_proj_weight.t().view(
        num_attention_heads,
        d_head,
        hidden_size,
    )

    # apply bias if present (No in Chameleon)
    if out_proj_bias is not None:
        out_proj_bias = out_proj_bias.view(1, 1, 1, hidden_size)

    # apply the projection for each head
    projected_values = einsum(
        values,
        out_proj_weight,
        "batch seq_len num_head d_head, num_head d_head d_model -> batch seq_len num_head d_model",
    )
    if out_proj_bias is not None:
        projected_values = projected_values + out_proj_bias

    # rearrange the tensor to have dimension that we like more
    projected_values = rearrange(
        projected_values,
        "batch seq_len num_head d_model -> batch num_head seq_len d_model",
    )

    # slice for token index
    # Assume projected_values has shape [batch, num_heads, tokens, d_model]
    if avg:
        # For each tuple, slice the tokens along dimension -2 and average over that token slice.
        token_avgs = []
        for token_tuple in token_indexes:
            # Slice out the tokens for this tuple.
            # Using ellipsis ensures we index the last two dimensions correctly.
            token_slice = projected_values[..., list(token_tuple), :]
            # Average over the token dimension (which is -2) while keeping that dimension.
            token_avg = torch.mean(token_slice, dim=-2, keepdim=True)
            token_avgs.append(token_avg)
        # Concatenate the averaged slices along the token dimension (-2).
        projected_values = torch.cat(token_avgs, dim=-2)
    else:
        # Flatten the list of token tuples into a single list of token indices.
        flatten_indexes = [item for tup in token_indexes for item in tup]
        projected_values = projected_values[..., flatten_indexes, :]

    # Post-process the value vectors by selecting heads.
    if head == "all":
        for head_idx in range(num_attention_heads):
            cache[f"projected_value_L{layer}H{head_idx}"] = projected_values[
                :, head_idx
            ]
    else:
        cache[f"projected_value_L{layer}H{head}"] = projected_values[:, int(head)]


def projected_key_vectors_head(
    module,
    args,
    kwargs,
    output,
    layer,
    cache,
    token_indexes,
    num_attention_heads: int,
    num_key_value_heads: int,
    hidden_size: int,
    d_head: int,
    out_proj_weight,
    out_proj_bias,
    head: Union[str, int] = "all",
    act_on_input=False,
    expand_head: bool = True,
    avg=False,
):
    r"""
    Hook function to extract the key vectors of the heads and project them through the attention output matrix.
    This shows the contribution that keys in each position could have to the residual stream through the attention mechanism.

    Like other hooks, it saves the activations in the cache.

    Args:
        b: the input of the hook function (output of the key vectors)
        layer: the layer of the model
        head: the head of the model. If "all" is passed, it will extract all the heads of the layer
        expand_head: bool to expand the head dimension when extracting the keys vectors
    """
    # Get the key vectors
    b = process_args_kwargs_output(args, kwargs, output)

    keys = b.data.detach().clone()  # (batch, num_heads, seq_len, head_dim)

    # Reshape the key vectors to have a separate dimension for the different heads
    keys = rearrange(
        keys,
        "batch seq_len (num_key_value_heads d_heads) -> batch num_key_value_heads seq_len d_heads",
        num_key_value_heads=num_key_value_heads,
        d_heads=d_head,
    )

    # If needed, repeat KV heads to match attention heads (for grouped query attention)
    keys = repeat_kv(keys, num_attention_heads // num_key_value_heads)

    keys = rearrange(
        keys,
        "batch num_head seq_len d_model -> batch seq_len num_head d_model",
    )

    # Reshape out_proj_weight to get the blocks for each head
    out_proj_weight = out_proj_weight.t().view(
        num_attention_heads,
        d_head,
        hidden_size,
    )

    # Apply bias if present
    if out_proj_bias is not None:
        out_proj_bias = out_proj_bias.view(1, 1, 1, hidden_size)

    # Apply the projection for each head
    projected_keys = einsum(
        keys,
        out_proj_weight,
        "batch seq_len num_head d_head, num_head d_head d_model -> batch seq_len num_head d_model",
    )
    if out_proj_bias is not None:
        projected_keys = projected_keys + out_proj_bias

    # Rearrange the tensor to have dimensions that we prefer
    projected_keys = rearrange(
        projected_keys,
        "batch seq_len num_head d_model -> batch num_head seq_len d_model",
    )

    # Process token indices
    if avg:
        # For each tuple, slice out the tokens and average over them
        token_avgs = []
        for token_tuple in token_indexes:
            token_slice = projected_keys[..., list(token_tuple), :]
            token_avg = torch.mean(token_slice, dim=-2, keepdim=True)
            token_avgs.append(token_avg)
        projected_keys = torch.cat(token_avgs, dim=-2)
    else:
        flatten_indexes = [item for tup in token_indexes for item in tup]
        projected_keys = projected_keys[..., flatten_indexes, :]

    # Save to cache based on selected heads
    if head == "all":
        for head_idx in range(num_attention_heads):
            cache[f"projected_key_L{layer}H{head_idx}"] = projected_keys[:, head_idx]
    else:
        cache[f"projected_key_L{layer}H{head}"] = projected_keys[:, int(head)]


def projected_query_vectors_head(
    module,
    args,
    kwargs,
    output,
    layer,
    cache,
    token_indexes,
    num_attention_heads: int,
    num_key_value_heads: int,
    hidden_size: int,
    d_head: int,
    out_proj_weight,
    out_proj_bias,
    head: Union[str, int] = "all",
    act_on_input=False,
    expand_head: bool = True,
    avg=False,
):
    r"""
    Hook function to extract the query vectors of the heads and project them through the attention output matrix.
    This shows the contribution that queries in each position could have to the residual stream through the attention mechanism.

    Like other hooks, it saves the activations in the cache.

    Args:
        b: the input of the hook function (output of the query vectors)
        layer: the layer of the model
        head: the head of the model. If "all" is passed, it will extract all the heads of the layer
        expand_head: bool to expand the head dimension when extracting the query vectors
    """
    # Get the query vectors
    b = process_args_kwargs_output(args, kwargs, output)

    queries = b.data.detach().clone()  # (batch, seq_len, num_heads*d_head)

    # Reshape the query vectors to have a separate dimension for the heads
    queries = rearrange(
        queries,
        "batch seq_len (num_attention_heads d_heads) -> batch num_attention_heads seq_len d_heads",
        num_attention_heads=num_attention_heads,
        d_heads=d_head,
    )

    queries = rearrange(
        queries,
        "batch num_head seq_len d_model -> batch seq_len num_head d_model",
    )

    # Reshape out_proj_weight to get the blocks for each head
    out_proj_weight = out_proj_weight.t().view(
        num_attention_heads,
        d_head,
        hidden_size,
    )

    # Apply bias if present
    if out_proj_bias is not None:
        out_proj_bias = out_proj_bias.view(1, 1, 1, hidden_size)

    # Apply the projection for each head
    projected_queries = einsum(
        queries,
        out_proj_weight,
        "batch seq_len num_head d_head, num_head d_head d_model -> batch seq_len num_head d_model",
    )
    if out_proj_bias is not None:
        projected_queries = projected_queries + out_proj_bias

    # Rearrange the tensor to have dimensions that we prefer
    projected_queries = rearrange(
        projected_queries,
        "batch seq_len num_head d_model -> batch num_head seq_len d_model",
    )

    # Process token indices
    if avg:
        # For each tuple, slice out the tokens and average over them
        token_avgs = []
        for token_tuple in token_indexes:
            token_slice = projected_queries[..., list(token_tuple), :]
            token_avg = torch.mean(token_slice, dim=-2, keepdim=True)
            token_avgs.append(token_avg)
        projected_queries = torch.cat(token_avgs, dim=-2)
    else:
        flatten_indexes = [item for tup in token_indexes for item in tup]
        projected_queries = projected_queries[..., flatten_indexes, :]

    # Save to cache based on selected heads
    if head == "all":
        for head_idx in range(num_attention_heads):
            cache[f"projected_query_L{layer}H{head_idx}"] = projected_queries[
                :, head_idx
            ]
    else:
        cache[f"projected_query_L{layer}H{head}"] = projected_queries[:, int(head)]


def avg_attention_pattern_head(
    module,
    args,
    kwargs,
    output,
    token_indexes,
    layer,
    attn_pattern_current_avg,
    batch_idx,
    cache,
    avg: bool = False,
    extract_avg_value: bool = False,
    act_on_input=False,
):
    """
    Hook function to extract the average attention pattern of the heads. It will extract the attention pattern and then average it.
    As the other hooks, it will save the activations in the cache (a global variable out the scope of the function)

    Args:
        - b: the input of the hook function. It's the output of the attention pattern of the heads
        - s: the state of the hook function. It's the state of the model
        - layer: the layer of the model
        - head: the head of the model
        - attn_pattern_current_avg: the current average attention pattern
    """
    # first get the attention pattern
    b = process_args_kwargs_output(args, kwargs, output)

    attn_pattern = b.data.detach().clone()  # (batch, num_heads,seq_len, seq_len)
    # attn_pattern = attn_pattern.to(torch.float32)
    num_heads = attn_pattern.size(1)

    token_indexes = [item for sublist in token_indexes for item in sublist]

    for head in range(num_heads):
        key = f"avg_pattern_L{layer}H{head}"
        if key not in attn_pattern_current_avg:
            attn_pattern_current_avg[key] = attn_pattern[:, head, token_indexes][
                :, :, token_indexes
            ]
        else:
            attn_pattern_current_avg[key] += (
                attn_pattern[:, head, token_indexes][:, :, token_indexes]
                - attn_pattern_current_avg[key]
            ) / (batch_idx + 1)
        attn_pattern_current_avg[key] = attn_pattern_current_avg[key]
        # var_key = f"M2_pattern_L{layer}H{head}"
        # if var_key not in attn_pattern_current_avg:
        #     attn_pattern_current_avg[var_key] = torch.zeros_like(attn_pattern[:, head])
        # attn_pattern_current_avg[var_key] = attn_pattern_current_avg[var_key] + (attn_pattern[:, head] - attn_pattern_current_avg[key]) * (attn_pattern[:, head] - attn_pattern_current_avg[var_key])

        if extract_avg_value:
            value_key = f"projected_value_L{layer}H{head}"
            try:
                values = cache[value_key]
            except KeyError:
                print(f"Values not found for {value_key}")
                return
            # get the attention pattern for the values
            value_norm = torch.norm(values, dim=-1)

            norm_matrix = (
                value_norm.unsqueeze(1).expand_as(attn_pattern[:, head]).transpose(1, 2)
            )

            norm_matrix = norm_matrix * attn_pattern[:, head]

            if value_key not in attn_pattern_current_avg:
                attn_pattern_current_avg[value_key] = norm_matrix[..., token_indexes, :]
            else:
                attn_pattern_current_avg[value_key] += (
                    norm_matrix[..., token_indexes, :]
                    - attn_pattern_current_avg[value_key]
                ) / (batch_idx + 1)

            # remove values from cache
            del cache[value_key]


def attention_pattern_head(
    module,
    args,
    kwargs,
    output,
    token_indexes,
    layer,
    cache,
    head: Union[str, int] = "all",
    act_on_input=False,
    attn_pattern_avg: Literal["mean", "sum", "baseline_ratio", "none"] = "none",
    attn_pattern_row_partition=None,
):
    """
    Hook function to extract the attention pattern of the heads. It will extract the attention pattern.
    As the other hooks, it will save the activations in the cache (a global variable out the scope of the function)

    Arguments:
        - args: the input args of the hook function
        - kwargs: the input kwargs of the hook function
        - output: the output of the hook function
        - token_indexes (List[Tuple]) : the indexes of the tokens to extract the attention pattern
        - layer (int): the layer of the model
        - cache (ActivationCache): the cache where to save the activations
        - head (Union[str, int]): the head of the model. If "all" is passed, it will extract all the heads of the layer
        - attn_pattern_avg (Literal["mean", "sum", "baseline_ratio", "none"]): the method to average the attention pattern
        - attn_pattern_row_partition (List[int]): the indexes of the tokens to partition the attention pattern

    Avg strategies:
        If the attn_pattern_avg is not "none", the attention pattern is divided in blocks and the average value of each block is computed, using the method specified in attn_pattern_avg.
        The idea is to partition the attention pattern into groups of tokens, and then compute a single average value for each group. The pattern is divided into len(attn_pattern_row_partition) x len(token_indexes) blocks, where each block B_ij is defined to have the indeces of the rows in attn_pattern_row_partition[i] and the columns in token_indexes[j]. If attn_pattern_row_partition is None, then the rows are the same as token_indexes.

        0| a_00 0    0    0    0    0    0              token_indexes = [(1,3), (4)]
        1| a_10 a_11 0    0    0    0    0              attn_pattern_row_partition = [(0,1)]
        2| a_20 a_21 a_22 0    0    0    0
        3| a_30 a_31 a_32 a_33 0    0    0
        4| a_40 a_41 a_42 a_43 a_44 0    0
        5| a_50 a_51 a_52 a_53 a_54 a_55 0
        6| a_60 a_61 a_62 a_63 a_64 a_65 a_66
           ------------------------------
            0    1     2   3     4    5    6

        - Block B_00:
            - Rows: 0,1
            - Columns: 1,2,3
            - Block: [a_01, a_02, a_03, a_11, a_12, a_13]
        - Block B_01:
            - Rows: 0,1
            - Columns: 4
            - Block: [a_04, a_14]


        If attn_pattern_avg is "mean", the average value for each block is computed as the mean of the block, so the output will be: batch n_row_blocks n_col_blocks
        So in this case, the output will be: batch 1 2 where the first value is the average of the first block and the second value is the average of the second block.

        The method to compute a single value for each block is specified by the attn_pattern_avg parameter, and can be one of the following:
        - "mean": Compute the mean of the block.
        - "sum": Compute the sum of the block.
        - "baseline_ratio": Compute the ratio of the observed average attention to the expected average attention. The expected average attention is computed by assuming that attention is uniform across the block. So, for each row in attn_pattern_row_partition, we compute the fraction of allowed keys that belong to token_indexes. The expected average attention is the sum of these fractions divided by the number of rows. The final ratio is the observed average attention divided by the expected average attention.


    """
    # first get the attention pattern
    b = process_args_kwargs_output(args, kwargs, output)

    attn_pattern = b.data.detach().clone()  # (batch, num_heads,seq_len, seq_len)

    if head == "all":
        head_indices = range(attn_pattern.size(1))
    else:
        head_indices = [head]

    if attn_pattern_row_partition is not None:
        token_indexes_group1 = attn_pattern_row_partition
    else:
        token_indexes_group1 = token_indexes

    # For each token group (each tuple in token_indexes), compute a single average value.
    if attn_pattern_row_partition is not None:
        for h in head_indices:
            # For head h, pattern has shape [batch, seq_len, seq_len].
            group_avgs = []

            # Generate all combinations of groups
            for group1 in token_indexes_group1:
                for group2 in token_indexes:
                    # Extract the attention block for this combination.
                    attn_block = attn_pattern[:, h, list(group1), :][:, :, list(group2)]

                    # Depending on the selected averaging method, compute a metric.
                    if attn_pattern_avg == "mean":
                        # Simple mean over the block.
                        avg_val = torch.mean(attn_block, dim=(-2, -1))  # shape: [batch]

                    elif attn_pattern_avg == "sum":
                        # Simple sum over the block.
                        avg_val = torch.sum(attn_block, dim=(-2, -1))

                    elif attn_pattern_avg == "baseline_ratio":
                        # ---- Step 1. Compute the observed average attention in the block.
                        observed_val = torch.mean(
                            attn_block, dim=(-2, -1)
                        )  # shape: [batch]

                        # ---- Step 2. Compute the baseline expectation.
                        # For each row (i.e. token index) in group1, we calculate the fraction of allowed keys
                        # that belong to group2. Because the attention is lower-triangular,
                        # a row with index 'i' can only attend to tokens with indices <= i.
                        # Thus, for each row 'i' in group1, the expected fraction (if uniform) is:
                        #      (# of tokens in group2 with index <= i) / (i+1)
                        baseline_list = []
                        for i in group1:
                            # Count the number of tokens in group2 that are allowed for row i.
                            allowed_count = sum(1 for j in group2 if j <= i)
                            # Total keys available for row i (assuming indices start at 0).
                            total_allowed = i + 1
                            # Avoid division by zero (should not happen if i>=0).
                            baseline_ratio = (
                                allowed_count / total_allowed
                                if total_allowed > 0
                                else 0.0
                            )
                            baseline_list.append(baseline_ratio)

                        # Average the per-row baseline over all rows in group1.
                        # This represents the expected average attention to group2 if it were uniformly distributed.
                        baseline_val = sum(baseline_list) / len(baseline_list)

                        # ---- Step 3. Compute the final ratio.
                        # We compare the observed average attention to the baseline expectation.
                        # A value > 1 means that, on average, attention in this block is higher than expected.
                        # Expand baseline_val to match the batch shape for element-wise division.
                        baseline_tensor = torch.tensor(
                            baseline_val, device=observed_val.device
                        ).expand_as(observed_val)
                        avg_val = observed_val / baseline_tensor
                    else:
                        avg_val = attn_block

                    if attn_pattern_avg != "none":
                        # Append the computed metric for this block (keeping the batch dimension).
                        group_avgs.append(avg_val.unsqueeze(1))  # shape: [batch, 1]
                    else:
                        # If no averaging is requested, store the block directly.
                        group_avgs.append(attn_block)

            if attn_pattern_avg != "none":
                pattern_avg = einops.rearrange(
                    torch.cat(group_avgs, dim=1),
                    "batch (G1 G2) -> batch G1 G2",
                    G1=len(token_indexes_group1),
                    G2=len(token_indexes),
                )
            else:
                try:
                    pattern_avg = torch.cat(group_avgs, dim=1)
                except:
                    raise ValueError(
                        f"Error concatenating group_avgs with shapes {[x.shape for x in group_avgs]}"
                    )

            # Add the pattern to the cache
            cache[f"pattern_L{layer}H{h}"] = pattern_avg
    else:
        # Without averaging, flatten token_indexes into one list.
        flatten_indexes = [item for tup in token_indexes for item in tup]
        for h in head_indices:
            # Slice both token dimensions using the flattened indexes.
            pattern_slice = attn_pattern[:, h, flatten_indexes][:, :, flatten_indexes]
            cache[f"pattern_L{layer}H{h}"] = pattern_slice


def input_embedding_hook(
    module,
    args,
    kwargs,
    output,
    cache,
    cache_key,
    token_indexes,
    keep_gradient: bool = False,
    avg: bool = False,
):
    r"""
    Hook to capture the output of the embedding layer, enable gradients, and store it in the cache.
    """
    embeddings_tensor = process_args_kwargs_output(args, kwargs, output)

    if keep_gradient:
        # Enable gradient tracking for the embeddings tensor
        embeddings_tensor.requires_grad_(True).retain_grad()
        cache[cache_key] = embeddings_tensor  # we slice in the end if keep gradient
        return restore_same_args_kwargs_output(
            embeddings_tensor, args, kwargs, output
        )  # Return the original (potentially modified in-place) output structure
    if avg:
        token_avgs = []
        for token_tuple in token_indexes:
            # Slice out the tokens specified by the tuple.
            token_slice = embeddings_tensor[:, list(token_tuple), :]
            # Average over the token dimension (dim=1) and keep that dimension.
            token_avg = torch.mean(token_slice, dim=1, keepdim=True)
            token_avgs.append(token_avg)
        cache[cache_key] = torch.cat(
            token_avgs, dim=1
        )  # Store the tensor that's part of the graph
    else:
        flatten_indexes = [item for tup in token_indexes for item in tup]
        cache[cache_key] = embeddings_tensor[
            :, flatten_indexes, :
        ]  # Store the tensor that's part of the graph

    return (
        output  # Return the original (potentially modified in-place) output structure
    )
