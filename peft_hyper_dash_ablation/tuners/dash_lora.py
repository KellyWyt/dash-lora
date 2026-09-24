# -*- coding: utf-8 -*-

# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import json
import math
import os
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import PeftConfig, PeftType, transpose

from .dash_ablation_importance import DashAblationAnalyzer, load_topology, save_topology


@dataclass
class LoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    lora_type: str = field(default=None, metadata={"help": "Lora type, moka or other lora"}) #NOTE qwen3

    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_nums: int = field(default=None, metadata={"help": "Numbers of Lora"})
    blc_alpha: int = field(default=None, metadata={"help": "Alpha of blcloss"})
    blc_weight: int = field(default=None, metadata={"help": "Weight of blcloss"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})

    ### eval
    reserved_modality: str = field(default=None, metadata={"help": "Alpha of blcloss"})
    loramethod: str = field(default=None, metadata={"help": "Alpha of blcloss"})



    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    hierarchical_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use hierarchical LoRA"}
    )
    safe_importance: bool = field(
        default=False,
        metadata={"help": "Use rank-0 FP32 importance analysis and broadcast its topology."},
    )
    top_k_layers: int = field(
        default=65, 
        metadata={"help": "Number of important layers"}
    )
    important_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of important module names"}
    )
    shared_modules: Optional[List[str]] = field(
        default=None, 
        metadata={"help": "List of shared module names"}
    )   
    # NOTE: Modified Config for Asymmetric Sharing
    text_important_modules: Optional[List[str]] = field(
        default=None, metadata={"help": "List of modules where Text (A0) is exclusive"}
    )
    others_important_modules: Optional[List[str]] = field(
        default=None, metadata={"help": "List of modules where Others (A1, A2) are exclusive"}
    )
    ratio:int = field(
        default=3, 
        metadata={"help": "ratio:1:1"}
    )
    dash_ablation_mode: str = field(
        default="energy",
        metadata={"help": "Ablation scoring mode: energy, random_topology, global_energy, reverse_energy, gradient."},
    )
    dash_ablation_seed: int = field(
        default=123,
        metadata={"help": "Seed used by random_topology."},
    )
    dash_ablation_topology_path: Optional[str] = field(
        default=None,
        metadata={"help": "JSON path used to save/load a fixed Dash-LoRA topology."},
    )
    dash_ablation_precomputed_topology: Optional[dict] = field(
        default=None,
        metadata={"help": "Already computed topology dictionary. Used by gradient ablation before LoRA is attached."},
    )


    def __post_init__(self):
        self.peft_type = PeftType.LORA


class LoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, config, model): # LoraConfig, CasualLM
        super().__init__()
        self.peft_config = config
        self.model = model

        # NOTE if hierarchical LoRA, analyze the importance first
        if getattr(config, 'hierarchical_lora', False):
            self._analyze_and_setup_hierarchical_lora()
        else:
            self._find_and_replace()

        mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward


    def _analyze_and_setup_hierarchical_lora(self): #NOTE
        """Analyze the model and set up hierarchical LoRA."""
        # print("Start the hierarchical LoRA importance analysis....")
        
        # first filter target modules
        target_module_keys = []
        key_list = [key for key, _ in self.model.named_modules()]

        target_modules = self.peft_config.target_modules
        
        llama_backbone_pattern = re.compile(r"(^|\.)model\.layers\.\d+\.")

        for key, module in self.model.named_modules():
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(target_modules, key)
            else:
                target_module_found = (
                    key.split(".")[-1] in target_modules
                    and llama_backbone_pattern.search(key) is not None
                )
            
            if target_module_found and isinstance(module, torch.nn.Linear):
                target_module_keys.append(key)
        
        # print(f"Found {len(target_module_keys)} target modules for LoRA")

        # Analysis of Model Importance
        
        safe_importance = getattr(self.peft_config, "safe_importance", False)
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if distributed else 0

        topology_path = getattr(self.peft_config, "dash_ablation_topology_path", None)
        precomputed_topology = getattr(self.peft_config, "dash_ablation_precomputed_topology", None)

        if precomputed_topology is not None:
            analysis_result = precomputed_topology
        elif topology_path and os.path.exists(topology_path):
            analysis_result = load_topology(topology_path)
        elif safe_importance:
            analysis_result = None
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            if rank == 0:
                analyzer = DashAblationAnalyzer(
                    mode=getattr(self.peft_config, "dash_ablation_mode", "energy"),
                    seed=getattr(self.peft_config, "dash_ablation_seed", 123),
                )
                if torch.cuda.is_available():
                    torch.cuda.set_device(local_rank)
                    analyzer.available_gpus = [local_rank]
                analysis_result = analyzer.analyze_single_gpu_safe(
                    self.model, target_module_keys,
                    top_k=self.peft_config.top_k_layers,
                    ratio=self.peft_config.ratio,
                )
                if topology_path:
                    save_topology(topology_path, analysis_result)
                del analyzer

            if distributed:
                payload = [analysis_result]
                if torch.cuda.is_available():
                    torch.cuda.set_device(local_rank)
                    broadcast_device = torch.device("cuda", local_rank)
                else:
                    broadcast_device = None
                torch.distributed.broadcast_object_list(payload, src=0, device=broadcast_device)
                analysis_result = payload[0]
        else:
            analyzer = DashAblationAnalyzer(
                mode=getattr(self.peft_config, "dash_ablation_mode", "energy"),
                seed=getattr(self.peft_config, "dash_ablation_seed", 123),
            )
            analysis_result = analyzer.analyze_model_only_target_modules(
                self.model, self.peft_config.target_modules, target_module_keys,
                top_k=self.peft_config.top_k_layers,
                ratio=self.peft_config.ratio,
            )
            if topology_path and rank == 0:
                save_topology(topology_path, analysis_result)
            del analyzer

        import gc
        gc.collect()
        torch.cuda.empty_cache() 
        
        # Guard against stale or externally supplied topology files that contain
        # visual/audio encoder modules. LoRA should be mounted only on LLaMA
        # backbone modules collected in target_module_keys.
        target_module_key_set = set(target_module_keys)
        for field in ("text_exclusive", "others_exclusive"):
            original = list(analysis_result.get(field, []))
            filtered = [name for name in original if name in target_module_key_set]
            removed = len(original) - len(filtered)
            if removed and rank == 0:
                print(
                    f"[DashAblation] filtered {removed} non-backbone modules from {field} "
                    f"before LoRA mounting."
                )
            analysis_result[field] = filtered
        
        # Save specific lists to config
        self.peft_config.text_important_modules = analysis_result['text_exclusive']
        self.peft_config.others_important_modules = analysis_result['others_exclusive']
        
        # Determine strict "shared modules" (those that are shared by AT LEAST one modality)
        # Actually, we just need to identify projection types for the matrix creation
        all_exclusive = set(self.peft_config.text_important_modules) | set(self.peft_config.others_important_modules)
        # We need shared matrices for any module that is NOT exclusive for a modality
        # Simply put: create shared matrices for ALL projection types found in target_modules.
        # The Linear layer will decide whether to use them.
        
        # Pre-create shared matrices (Independent of layers, created by proj_type)
        # We assume shared_modules config is now just a helper or we derive it
        self.peft_config.shared_modules = list(set(target_module_keys) - set(analysis_result['text_exclusive'])) # loosely defined
        self._create_shared_A_matrices(self.peft_config.target_modules) # Pass simple list like ['q_proj']

        # Replace
        self._find_and_replace_hierarchical(target_module_keys)


    def _create_shared_A_matrices(self,target_modules): #NOTE
        """Create shared A matrices for each projection type separately"""
        if not hasattr(self.peft_config, 'shared_modules') or not self.peft_config.shared_modules:
            return
        
        # Initialize shared matrices dictionary structure
        # Format: {proj_type: {'shared_A0': Linear, 'shared_A1': Linear, ...}}
        self.shared_A_params = {}
        
        # Group shared modules by projection type
        shared_modules_by_type = self._group_shared_modules_by_type(target_modules)
        
        # # Parse r into dimensions
        # r = self.peft_config.r
        # rr = []
        # if isinstance(r, int):
        #     r_str = str(r)
        #     rr = [int(digit) for digit in r_str]
        # else:
        #     rr = [int(char) for char in str(r)]   #NOTE moka

        # Parse r into dimensions
        r = self.peft_config.r
        tmp_len = 3  

        if isinstance(r, int):
            # [r, r, r]
            rr = [r] * tmp_len
        else:
            rr = [int(r)] * tmp_len #NOTE fix

        default_dimensions = {
                'q_proj': 4096, 'k_proj': 4096, 'v_proj': 4096, 'o_proj': 4096,
                'gate_proj': 4096, 'up_proj': 4096, 'down_proj': 11008
            }

        # Create shared matrices for each projection type
        for proj_type in target_modules:
            in_features = None
            
            if proj_type in shared_modules_by_type and shared_modules_by_type[proj_type]:
                try:
                    first_module_key = shared_modules_by_type[proj_type][0]
                    parent, target, target_name = self._get_submodules(first_module_key)
                    in_features = target.in_features
                except Exception as e:
                    print(f"Warning: Could not get input features for {proj_type}: {e}")
            
            if in_features is None and getattr(self.peft_config, "safe_importance", False):
                # A projection type can have zero shared modules when all of it is
                # selected as exclusive. Infer its real input width from any base
                # module instead of using a Llama-specific fallback dimension.
                llama_backbone_pattern = re.compile(r"(^|\.)model\.layers\.\d+\.")
                for module_name, module in self.model.named_modules():
                    if (
                        module_name.endswith(proj_type)
                        and llama_backbone_pattern.search(module_name) is not None
                        and hasattr(module, "in_features")
                    ):
                        in_features = module.in_features
                        break

            if in_features is None:
                in_features = default_dimensions.get(proj_type, 4096)
                print(f"Using fallback input features {in_features} for {proj_type}")
            
            type_params = {}
            for i in range(min(self.peft_config.lora_nums, len(rr))):
                param = nn.Parameter(torch.empty(rr[i],in_features)) #TODO
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                type_params[f'shared_A{i}'] = param
                
                # print(f"Created shared parameter for {proj_type}.shared_A{i}: "
                #     f"shape=({in_features}, {rr[i]})")
            
            self.shared_A_params[proj_type] = type_params
        
        self.shared_A_matrices = self.shared_A_params
        
        print("\nShared parameter creation summary:")
        for proj_type, params in self.shared_A_params.items():
            for key, param in params.items():
                print(f"  {proj_type}.{key}: shape={tuple(param.shape)}")

    def _group_shared_modules_by_type(self, target_modules): #NOTE
        """Group shared modules by their projection type"""
        shared_modules_by_type = {proj_type: [] for proj_type in target_modules}
        
        if not hasattr(self.peft_config, 'shared_modules'):
            return shared_modules_by_type
        
        def get_projection_type(key):
            """Extract projection type from module name"""
            for proj_type in target_modules:
                if f'.{proj_type}' in key or key.endswith(f'.{proj_type}'):
                    return proj_type
            return None
        
        # Group all shared modules
        for key in self.peft_config.shared_modules:
            proj_type = get_projection_type(key)
            if proj_type and proj_type in shared_modules_by_type:
                shared_modules_by_type[proj_type].append(key)
        
        # Print grouping statistics
        # print("\nShared modules grouped by projection type:")
        for proj_type, modules in shared_modules_by_type.items():
            if modules:
                print(f"  {proj_type:10s}: {len(modules):3d} modules")
            else:
                print(f"  {proj_type:10s}: 0 modules (no shared modules of this type)")
        
        return shared_modules_by_type

    def _find_and_replace_hierarchical(self,target_module_keys): #NOTE
        """Replace modules hierarchically with type-specific shared parameters"""
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        base_kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "lora_nums": self.peft_config.lora_nums,
            "blc_alpha": self.peft_config.blc_alpha,
            "blc_weight": self.peft_config.blc_weight,
            "reserved_modality": self.peft_config.reserved_modality,
            "loramethod": self.peft_config.loramethod,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
        }
        
        text_list = set(self.peft_config.text_important_modules)
        others_list = set(self.peft_config.others_important_modules)

        for key in target_module_keys:
            kwargs = base_kwargs.copy()
            
            # Determine independence per modality for THIS specific layer
            is_text_independent = key in text_list
            is_others_independent = key in others_list
            
            # Get shared params (if needed)
            proj_type = self._get_projection_type_from_key(key, self.peft_config.target_modules)
            shared_params = self.shared_A_params.get(proj_type, None)
            
            self._replace_single_module_asymmetric(
                key, kwargs, 
                is_text_independent, 
                is_others_independent, 
                shared_params
            )

    def _get_projection_type_from_key(self, key, target_modules):
        """Extract projection type from module key"""
        for proj_type in target_modules:
            if f'.{proj_type}' in key or key.endswith(f'.{proj_type}'):
                return proj_type
        return None
    
    def _replace_single_module_asymmetric(self, key, kwargs, is_text_independent, is_others_independent, shared_A_matrices):
        try:
            parent, target, target_name = self._get_submodules(key)
            bias = target.bias is not None
            
            if isinstance(target, torch.nn.Linear):
                new_module = Linear(
                    target.in_features, 
                    target.out_features, 
                    bias=bias, 
                    # New Flags
                    is_text_independent=is_text_independent,
                    is_others_independent=is_others_independent,
                    shared_A_matrices=shared_A_matrices,
                    **kwargs
                )
                self._replace_module(parent, target_name, new_module, target)
        except Exception as e:
            print(f"Error replacing {key}: {e}")

    def _find_and_replace(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit):
            raise ImportError(
                "To use Lora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "lora_nums": self.peft_config.lora_nums,
            "blc_alpha": self.peft_config.blc_alpha,
            "blc_weight": self.peft_config.blc_weight,
            "reserved_modality": self.peft_config.reserved_modality,
            "loramethod": self.peft_config.loramethod,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        llama_backbone_pattern = re.compile(r"(^|\.)model\.layers\.\d+\.")
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = (
                    any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
                    and llama_backbone_pattern.search(key) is not None
                )
            if target_module_found: # here
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None

                if isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)

                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):  
        # #NOTE moka
        # self.r=[]
        # ll=str(r)
        # for i in range(len(ll)):
        #     self.r.append(int(ll[i]))
        #NOTE fix
        tmp=3
        self.r=[]
        for i in range(tmp):
            self.r.append(r)
        # print(f"DEBUG: self.r[0]:{self.r[0]},self.r[1]:{self.r[1]},self.r[2]:{self.r[2]}") #fix

        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 2,
        blc_alpha: float = 0.0,
        blc_weight: float = 0.0,
        lora_dropout: float = 0.0,
        reserved_modality="text",
        loramethod="uni",
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        # hierarchical_type: str = "important",  # Added: Layer types
        shared_A_matrices: dict = None, # TODO
        is_text_independent: bool = False,   # A0 exclusive?
        is_others_independent: bool = False, # A1, A2 exclusive?
        ratio:int = 1,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.loramethod= loramethod

        self.lora_num = lora_nums
        self.blc_alpha = blc_alpha
        self.blc_weight = blc_weight

        self.reserved_modality=reserved_modality
        # self.hierarchical_type = hierarchical_type #NOTE
        self.shared_A_matrices = shared_A_matrices #NOTE

        self.is_text_independent = is_text_independent
        self.is_others_independent = is_others_independent
        
        self.fan_in_fan_out = fan_in_fan_out
        self.ratio = ratio

        # #NOTE moka
        # rr=[]
        # ll=str(r)
        # for i in range(len(ll)):
        #     rr.append(int(ll[i]))

        #NOTE fix
        tmp=3
        rr=[]
        # print(self.lora_num)
        # print(self.lora_nums)
        for i in range(tmp):
            rr.append(r)

        
        self.d_k=rr[0]
        # print(f"DEBUG INIT: r={r}, rr[0]={rr[0]}, rr[1]={rr[1]},rr[2]={rr[2]},d_k={self.d_k}") #NOTE  加上这一行


        if rr[0] > 0:
            # === A0: Text Modality ===
            if is_text_independent:
                self.lora_A0 = nn.Linear(in_features, rr[0], bias=False)
            else:
                if shared_A_matrices and 'shared_A0' in shared_A_matrices:
                    self.lora_A0 = nn.Linear(in_features, rr[0], bias=False)
                    del self.lora_A0.weight
                    self.lora_A0._parameters['weight'] = shared_A_matrices['shared_A0']
                else:
                    # Fallback
                    self.lora_A0 = nn.Linear(in_features, rr[0], bias=False)

            # === A1, A2...: Other Modalities ===
            for i in range(1, self.lora_num):
                if is_others_independent:
                    setattr(self, f"lora_A{i}", nn.Linear(in_features, rr[i], bias=False))
                else:
                    if shared_A_matrices and f'shared_A{i}' in shared_A_matrices:
                        l_a = nn.Linear(in_features, rr[i], bias=False)
                        del l_a.weight
                        l_a._parameters['weight'] = shared_A_matrices[f'shared_A{i}']
                        setattr(self, f"lora_A{i}", l_a)
                    else:
                        setattr(self, f"lora_A{i}", nn.Linear(in_features, rr[i], bias=False))


            for i in range(1):
                setattr(self, f"lora_B{i}", nn.Linear(rr[0], out_features, bias=False))
            

            self.scaling = []

            ## scaling text lora
            for i in range(1):
                self.scaling.append(self.lora_alpha / self.r[i]) #NOTE moka
            # ## scaling text lora
            # for i in range(1):
            #     self.scaling.append(self.lora_alpha / self.r) #NOTE fix
            
             # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

        
        if hasattr(self, "lora_A0"):
            if self.is_text_independent:
                nn.init.kaiming_uniform_(self.lora_A0.weight, a=math.sqrt(5))

            for i in range(1, self.lora_num):
                if hasattr(self, f"lora_A{i}"):
                    if self.is_others_independent:
                         nn.init.kaiming_uniform_(getattr(self, f"lora_A{i}").weight, a=math.sqrt(5))

            for i in range(1):
                if hasattr(self, f"lora_B{i}"):
                    nn.init.zeros_(getattr(self, f"lora_B{i}").weight)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)

        for i in range(self.lora_num):
            getattr(self, f"lora_A{i}").train(mode)
        for i in range(1):
            getattr(self, f"lora_B{i}").train(mode)

    def eval(self):
        nn.Linear.eval(self)
        for i in range(self.lora_num):
            getattr(self, f"lora_A{i}").eval()
        for i in range(1):
            getattr(self, f"lora_B{i}").eval()



    def forward(self, x: torch.Tensor, modality_mask:List[torch.Tensor]=None):

        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)        

            
        ## infer not first forward only text token
        if(('test' in self.loramethod) &(x.size(1)==1) ):
            # only text token
            output_a=getattr(self, f"lora_A0")(self.lora_dropout(x))*self.scaling[0]
            input_b=output_a
            output_b=getattr(self, f"lora_B0")(input_b)

            result=output_b+result

            return result


        ## infer first forward
        if(('test' in self.loramethod) &(x.size(1)!=1) ):

            text_mask=modality_mask[0]
            video_mask=modality_mask[1]
            audio_mask=modality_mask[2]
            question_mask=modality_mask[3]
            ## train process
        
            only_inputs=[x*text_mask,x*video_mask,x*audio_mask]
            # 0: text 
            # 1: video 
            # 2: audio
            # 3: question

            #### get question mask

            output_a=[]
            for i in range(self.lora_num):
                output_a.append(getattr(self, f"lora_A{i}")(self.lora_dropout(only_inputs[i]))*self.scaling[0])
            

            ### video_token: cross attention per sample
            video_token=output_a[1]
            question_token=output_a[0]*question_mask
            new_video=torch.zeros_like(video_token)

            for i in range(question_token.size(0)):
                query=video_token[i,:,:].unsqueeze(0)
                
                ## get question tokens
                indices = torch.where(question_mask[i,:,:] == 1)[0]
                key=question_token[i,indices[0]:indices[-1]+1,:].unsqueeze(0)
                value=question_token[i,indices[0]:indices[-1]+1,:].unsqueeze(0)


                score = torch.matmul(query, key.transpose(-2, -1))/ math.sqrt(self.d_k) 
                score = torch.softmax(score, dim=-1)
                output = torch.matmul(score, value)  # shape: (1, token_num, 4)
                attention_outputs=video_mask[i,:,:]*output
                new_video[i,:,:]=video_token[i,:,:]+attention_outputs*self.blc_weight

            ### audio_token: cross attention per sample
            audio_token=output_a[2]
            question_token=output_a[0]*question_mask
            new_audio=torch.zeros_like(audio_token)

            for i in range(question_token.size(0)):


                query=audio_token[i,:,:].unsqueeze(0)

                ## get question tokens
                indices = torch.where(question_mask[i,:,:] == 1)[0]
                key=question_token[i,indices[0]:indices[-1]+1,:].unsqueeze(0)
                value=question_token[i,indices[0]:indices[-1]+1,:].unsqueeze(0)


                score = torch.matmul(query, key.transpose(-2, -1))/ math.sqrt(self.d_k) 
                score = torch.softmax(score, dim=-1)
                output = torch.matmul(score, value)  # shape: (1, token_num, 4)
                attention_outputs=audio_mask[i,:,:]*output
                new_audio[i,:,:]=audio_token[i,:,:]+attention_outputs*self.blc_weight
            

            input_b=[output_a[0],new_video,new_audio]
            input_b=sum(input_b)


            output_b=getattr(self, f"lora_B0")(input_b)

            result=output_b+result
            
            return result

        ## train
        if('train' in self.loramethod):

            text_mask=modality_mask[0]
            video_mask=modality_mask[1]
            audio_mask=modality_mask[2]
            question_mask=modality_mask[3]
            ## train process
        
            only_inputs=[x*text_mask,x*video_mask,x*audio_mask]
            # 0: text 
            # 1: video 
            # 2: audio

            #### get question mask

            output_a=[]
            for i in range(self.lora_num):
                output_a.append(getattr(self, f"lora_A{i}")(self.lora_dropout(only_inputs[i]))*self.scaling[0])
            

            ### video_token: cross attention per sample
            video_token=output_a[1]
            question_token=output_a[0]*question_mask
            new_video=torch.zeros_like(video_token)

            for i in range(question_token.size(0)):
                query=video_token[i,:,:].unsqueeze(0)

                ## get question tokens
                indices = torch.where(question_mask[i,:,:] == 1)[0]
                key=question_token[i,indices[0]:indices[-1]+1,:].unsqueeze(0)
                value=question_token[i,indices[0]:indices[-1]+1,:].unsqueeze(0)
                # print(f"DEBUG: query shape: {query.shape}, key shape: {key.shape}")

                score = torch.matmul(query, key.transpose(-2, -1))/ math.sqrt(self.d_k) 
                score = torch.softmax(score, dim=-1)
                output = torch.matmul(score, value)  # shape: (1, token_num, 4)
                attention_outputs=video_mask[i,:,:]*output
                new_video[i,:,:]=video_token[i,:,:]+attention_outputs*self.blc_weight

            ### audio_token: cross attention per sample
            audio_token=output_a[2]
            question_token=output_a[0]*question_mask
            new_audio=torch.zeros_like(audio_token)

            for i in range(question_token.size(0)):


                query=audio_token[i,:,:].unsqueeze(0)

                ## get question tokens
                indices = torch.where(question_mask[i,:,:] == 1)[0]
                key=question_token[i,indices[0]:indices[-1]+1,:].unsqueeze(0)
                value=question_token[i,indices[0]:indices[-1]+1,:].unsqueeze(0)


                score = torch.matmul(query, key.transpose(-2, -1))/ math.sqrt(self.d_k)
                score = torch.softmax(score, dim=-1)
                output = torch.matmul(score, value)  # shape: (1, token_num, 4)
                attention_outputs=audio_mask[i,:,:]*output
                new_audio[i,:,:]=audio_token[i,:,:]+attention_outputs*self.blc_weight
            

            input_b=[output_a[0],new_video,new_audio]
            input_b=sum(input_b)


            output_b=getattr(self, f"lora_B0")(input_b)

            result=output_b+result
            
            return result
