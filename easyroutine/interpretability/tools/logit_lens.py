import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, Tuple, List, Literal
from easyroutine.interpretability.activation_cache import ActivationCache
from easyroutine.interpretability.hooked_model import HookedModel
from easyroutine.interpretability.models import ModelConfig
# from easyroutine.logger import logger
from easyroutine.console import progress
from copy import deepcopy
import re
from pathlib import Path
import json

class LogitLens:
    def __init__(self, unembedding_matrix, last_layer_norm: nn.Module, model_name: str, model_config:ModelConfig):
        self.unembed = deepcopy(unembedding_matrix)
        self.norm = deepcopy(last_layer_norm)
        self.model_name = model_name
        self.layernorm_type:Literal["RMS", "LayerNorm"] = model_config.layernorm_type
        self.num_attn_heads  = model_config.num_attention_heads
        self.tuned_lens = None
        
    def __repr__(self):
        return f"LogitLens({self.model_name})"
    @classmethod
    def from_model(cls, model: HookedModel) -> 'LogitLens':
        model.use_full_model()
        return cls(model.get_lm_head(), model.get_last_layernorm(), model.config.model_name, model.model_config)
    
    @classmethod
    def from_model_name(cls, model_name: str) -> 'LogitLens':
        model = HookedModel.from_model_name(model_name)
        cls = cls.from_model(model)
        del model
        torch.cuda.empty_cache()
        return cls
    
    @classmethod
    def from_tuned_lens(cls, tuned_lens_path: Path, model: HookedModel) -> 'LogitLens':
        cls = cls.from_model(model)
        tuned_lens_path = Path(tuned_lens_path)
        if not tuned_lens_path.exists():
            raise FileNotFoundError(f"Tuned lens path {tuned_lens_path} does not exist")
        if not tuned_lens_path.is_dir():
            raise ValueError(f"Tuned lens path {tuned_lens_path} is not a directory")
        
        cls.tuned_lens = torch.load(Path(tuned_lens_path,"params.pt"))
        tuned_lens_config = json.load(open(Path(tuned_lens_path,"config.json"), "r"))
        if tuned_lens_config["base_model_name_or_path"] != model.config.model_name:
            raise ValueError(f"Tuned lens model {tuned_lens_config['base_model_name_or_path']} does not match model {model.config.model_name}")
        
        return cls
        
    def to(self, device: Union[str, torch.device]):
        # move 
        self.unembed = self.unembed.to(device)
        self.norm = self.norm.to(device)
        
    def device(self) -> torch.device:
        if self.unembed.device != self.norm.weight.device:
            self.unembed = self.unembed.to(self.norm.weight.device)
        return self.unembed.device
    
    def get_vocab_size(self):
        return self.unembed.shape[1]
    
    # def get_keys(self, activations: ActivationCache, key: str):
    #     # check if is a format key ("resid_out_{i}")
    #     keys = []
    #     if "{i}" in key:
    #         # check if exists and how many starting from 0
    #         i = 0
    #         while activations.get(f"{key.format(i=i)}") is not None:
    #             keys.append(f"{key.format(i=i)}")
    #             i += 1
    #         if i == 0:
    #             raise KeyError(f"Key {key} not found in activations")
    #         else:
    #             self.logger.info(f"Key {key} found in activations with {i} elements")
    #             return keys
    #     else:
    #         if activations.get(key) is None:
    #             raise KeyError(f"Key {key} not found in activations")
    #         else:
    #             self.logger.info(f"Key {key} found in activations")
    #             return [key]


    def get_keys(self, activations, key_pattern: str):
        """
        Given an ActivationCache `activations` and a key pattern containing placeholders
        (e.g. "head_out_L{i}H{j}"), generate all keys present in the cache that match this pattern.
        
        This implementation converts the pattern into a regex by replacing each placeholder 
        with a regex group that matches one or more digits.
        """
        # If no placeholders are found, check for the literal key.
        if "{" not in key_pattern:
            if activations.get(key_pattern) is None:
                raise KeyError(f"Key {key_pattern} not found in activations")
            return [key_pattern]

        # Escape the literal parts of the key pattern.
        regex_pattern = re.escape(key_pattern)
        # Replace escaped placeholders (e.g. "\{i\}") with a regex group for digits.
        regex_pattern = re.sub(r'\\\{.*?\\\}', r'(\\d+)', regex_pattern)
        regex_compiled = re.compile(f"^{regex_pattern}$")

        # Collect all keys that match this regex.
        matching_keys = [k for k in activations.keys() if regex_compiled.fullmatch(k)]
        
        if not matching_keys:
            raise KeyError(f"No keys found for pattern {key_pattern}")

        # Sort keys numerically based on the numbers embedded in them.
        def sort_key(s):
            return [int(x) for x in re.findall(r'\d+', s)]
        
        return sorted(matching_keys, key=sort_key)
    
    def apply_layernorm(self, act, mean, variance, second_moment, key):
        """
        Apply RMSNorm or LayerNorm correctly.

        - RMSNorm: Uses `second_moment` for normalization.
        - LayerNorm: Uses `mean` and `variance`, applying it manually when cached parameters exist.

        Args:
            act (torch.Tensor): Activation tensor `[batch, token_indexes, hidden_dim]`
            mean (torch.Tensor): Cached mean `[batch, token_indexes]`
            variance (torch.Tensor): Cached variance `[batch, token_indexes]`
            second_moment (torch.Tensor): Cached second moment `[batch, token_indexes]`
            key (str): The activation key to decide normalization type.

        Returns:
            torch.Tensor: Normalized activation tensor.
        """
        input_dtype = act.dtype
        act = act.to(torch.float32)  # Improve numerical stability
        normalized_shape = (act.shape[-1],)  # Normalize across `hidden_dim`
        device = act.device

        if self.layernorm_type == "RMS":
            weight = self.norm.weight
            if hasattr(self.norm, "variance_epsilon"):
                variance_eps = self.norm.variance_epsilon
            elif hasattr(self.norm, "eps"):
                variance_eps = self.norm.eps
            else:
                raise ValueError("No variance epsilon found in RMSNorm")
            weight= weight.to(device)

            # Ensure broadcasting `[batch, token_indexes] -> [batch, token_indexes, hidden_dim]`
            second_moment = second_moment.unsqueeze(-1)

            # RMSNorm computation: x / sqrt(E[x²] + eps)
            act = act * torch.rsqrt(second_moment + variance_eps)
            return weight * act.to(input_dtype)

        elif self.layernorm_type == "LayerNorm":
            weight = self.norm.weight
            bias = self.norm.bias
            eps = self.norm.eps
            weight = weight.to(device)

            # Ensure broadcasting `[batch, token_indexes] -> [batch, token_indexes, hidden_dim]`
            mean = mean.unsqueeze(-1)
            variance = variance.unsqueeze(-1)

            # Apply LayerNorm manually
            act = (act - mean) / torch.sqrt(variance + eps)

            if "head" in key:
                return weight * act + (bias / self.num_attn_heads).unsqueeze(-1)
            else:
                return weight * act + bias.unsqueeze(-1)

    def compute(
        self,
        activations: ActivationCache,
        target_key: str,
        token_directions: Optional[Union[List[int], List[Tuple[int, int]]]] = None,
        apply_norm: bool = True,
        apply_softmax: bool = False,
        metric: Optional[Literal["logit_diff", "accuracy"]] = "logit_diff" 
    ) -> dict:
        """
        Compute the logit lens on the activations given at the target_key.

        Arguments:
            activations (ActivationCache): The activations store.
            target_key (str): The key where to apply the logit lens.
            apply_norm (bool): Whether to apply the last layer norm.
            apply_softmax (bool): Whether to apply softmax after unembedding.
            metric (Literal["logit_diff", "accuracy"]): The metric to compute. If used, token_directions must be provided. "logit_diff" computes the difference between the difference of the two tokens while "accuracy" computes the frequency of the first token being greater than the second token in the logits.

        Returns:
            dict: A dictionary with the logit lens results.
        """

        # Ensure last_layernorm parameters are available if applying normalization
        if apply_norm:
            assert "last_layernorm" in activations.keys(), "Last layer norm not found in activations"

        keys = self.get_keys(activations, target_key)
        logit_lens = {}
        last_layernorm_cached_parameters = activations.get("last_layernorm")

        with progress_bar as p:
            for key in p.track(keys, total=len(keys), description=f"Computing Logit Lens of {target_key}"):
                act = activations.get(key).to("cpu")  # Assuming computations on CPU

                if token_directions is not None:
                    assert len(token_directions) == act.shape[0], "Token directions must match batch size"
                    if metric == "logit_diff":
                        batch_size = act.shape[0]
                        seq_len = act.shape[1] if len(act.shape) > 2 else 1
                        logits = torch.zeros(batch_size, seq_len, device=act.device)

                        for i, direction in enumerate(token_directions):
                            if (isinstance(direction, tuple) or isinstance(direction, list)) and len(direction) == 2:
                                tok1, tok2 = direction
                                direction_vector = self.unembed[tok1] - self.unembed[tok2]
                            else:
                                direction_vector = self.unembed[direction]

                            # Select activations and corresponding norm parameters
                            curr_act = act[i] if len(act.shape) == 2 else act[i].reshape(-1, act.shape[-1])
                            mean = last_layernorm_cached_parameters["mean"][i, :]
                            variance = last_layernorm_cached_parameters["variance"][i, :]
                            second_moment = last_layernorm_cached_parameters["second_moment"][i, :]

                            # Apply LayerNorm
                            if apply_norm:
                                curr_act = self.apply_layernorm(curr_act, mean, variance, second_moment, key)
                            
                            if self.tuned_lens is not None:
                                curr_act = self.apply_tuned_lens_traslator(curr_act, key)

                            logits[i] = torch.matmul(curr_act.to(self.unembed.device), direction_vector)

                    elif metric == "accuracy":
                        batch_size = act.shape[0]
                        seq_len = act.shape[1] if len(act.shape) > 2 else 1
                        logits_tok1 = torch.zeros(batch_size, seq_len, device=act.device)
                        logits_tok2 = torch.zeros(batch_size, seq_len, device=act.device)
                        total_tok1_greater_tok2 = 0
                        for i, direction in enumerate(token_directions):
                            if (isinstance(direction, tuple) or isinstance(direction, list)) and len(direction) == 2:
                                tok1, tok2 = direction
                                tok1_unembed = self.unembed[tok1]
                                tok2_unembed = self.unembed[tok2]
                                
                                curr_act = act[i] if len(act.shape) == 2 else act[i].reshape(-1, act.shape[-1])
                                mean = last_layernorm_cached_parameters["mean"][i, :]
                                variance = last_layernorm_cached_parameters["variance"][i, :]
                                second_moment = last_layernorm_cached_parameters["second_moment"][i, :]
                                
                                if apply_norm:
                                    curr_act = self.apply_layernorm(curr_act, mean, variance, second_moment, key)
                                if self.tuned_lens is not None:
                                    curr_act = self.apply_tuned_lens_traslator(curr_act, key)
                                
                                logits_tok1 = torch.matmul(curr_act.to(self.unembed.device), tok1_unembed)
                                logits_tok2 = torch.matmul(curr_act.to(self.unembed.device), tok2_unembed)

                                    
                                if logits_tok1 > logits_tok2:
                                    total_tok1_greater_tok2 += 1
                        
                        logits = torch.tensor(total_tok1_greater_tok2 / len(token_directions))
                    else:
                        raise ValueError(f"Metric {metric} not supported")
                            
                            
                    
                    logit_lens[f"logit_lens_{key}"] = logits

                else:
                    if apply_norm:
                        mean = last_layernorm_cached_parameters["mean"]
                        variance = last_layernorm_cached_parameters["variance"]
                        second_moment = last_layernorm_cached_parameters["second_moment"]

                        act = self.apply_layernorm(act, mean, variance, second_moment, key)

                    logits = torch.matmul(act.to(self.unembed.device), self.unembed.T)
                    if apply_softmax:
                        logits = torch.softmax(logits, dim=-1)
                    logit_lens[f"logit_lens_{key}"] = logits

        return logit_lens

    def apply_tuned_lens_traslator(self, act, key):
        """
        Apply the tuned lens translator to the activations.

        Args:
            act (torch.Tensor): The activations to apply the tuned lens translator to.
            key (str): The key to use for the tuned lens translator.

        Returns:
            torch.Tensor: The translated activations.
        """
        # get the layer number from the key both for resid_out and head_out. The key is in the format "resid_out_{i}" or "head_out_L{i}H{j}"
        layer_number = int(re.search(r"\d+", key).group())
        
        if self.tuned_lens is None:
            raise ValueError("Tuned lens not found. Please load the tuned lens first.")
        
        weight = self.tuned_lens[f"{layer_number}.weight"]
        bias = self.tuned_lens[f"{layer_number}.bias"]
        # move to the same device 
        weight = weight.to(act.device)
        # print("transp")
        weight = weight.t()
        bias = bias.to(act.device)
        # Apply the tuned lens translator
        act = F.linear(act, weight, bias)
        return act
        
    
        
        