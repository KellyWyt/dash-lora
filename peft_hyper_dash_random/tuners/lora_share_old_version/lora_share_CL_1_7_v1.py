# coding=utf-8
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
import math
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

# from ..utils import PeftConfig, PeftType, transpose

class PeftType(str, Enum):
    LORA = "LORA"

@dataclass
class PeftConfig:
    peft_type: str = field(default=None)
    base_model_name_or_path: str = field(default=None)
    task_type: str = field(default=None)
    inference_mode: bool = field(default=False)

def transpose(weight, fan_in_fan_out):
    return weight.t() if fan_in_fan_out else weight


# from .importance_compute import HierarchicalImportanceAnalyzer
from .importance_compute_multi_GPU_1_7 import HierarchicalImportanceAnalyzer


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

    # TODO Hierarchical LoRA Configuration Added
    hierarchical_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use hierarchical LoRA"}
    )
    top_k_layers: int = field(
        default=50, 
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

        self.shared_A_matrices = {} # TODO

        # TODO if hierarchical LoRA, analyze the importance first
        if getattr(config, 'hierarchical_lora', False):
            self._analyze_and_setup_hierarchical_lora()
        else:
            self._find_and_replace()

        mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward 

    # TODO
    def _analyze_and_setup_hierarchical_lora(self):
        """Analyze the model and set up hierarchical LoRA."""
        # print("Start the hierarchical LoRA importance analysis....")
        
        # first filter target modules
        target_module_keys = []
        key_list = [key for key, _ in self.model.named_modules()]

        target_modules = self.peft_config.target_modules
        
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in target_modules)
            
            if target_module_found:
                target_module_keys.append(key)
        
        # print(f"Found {len(target_module_keys)} target modules for LoRA")

        # Analysis of Model Importance
        
        analyzer = HierarchicalImportanceAnalyzer()
        top_k_layers, importance_scores = analyzer.analyze_model_only_target_modules(
            self.model,target_modules,target_module_keys,top_k=self.peft_config.top_k_layers
        )
        
        # print(f"Top-{self.peft_config.top_k_layers} important:")
        # for i, (layer_name, score) in enumerate(
        #     sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:self.peft_config.top_k_layers]
        # ):
            # print(f"{i+1}. {layer_name}: {score:.4f}")
        
        # Separate important layers and shared layers.
        target_count = len(target_module_keys)
        matched_count = 0

        important_modules = []
        shared_modules = []

        for key in target_module_keys:
            # Direct check if in top_k_layers
            if key in top_k_layers:
                important_modules.append(key)
                matched_count += 1
                # print(f"✓ IMPORTANT: {key}")
            else:
                shared_modules.append(key)
                # print(f"  SHARED: {key}")
        
        # print(f"len(importance LoRA module): {len(important_modules)} ")
        # print(f"len(shared LoRA module): {len(shared_modules)} ")
        
        # save to config
        self.peft_config.important_modules = important_modules
        self.peft_config.shared_modules = shared_modules
        
        # pre create shared-A matrice (by projection type)
        self._create_shared_A_matrices(target_modules)

        # replace
        self._find_and_replace_hierarchical(target_modules)

    def _create_shared_A_matrices1(self,target_modules):
        """create_shared_A_matrices"""
        if not hasattr(self.peft_config, 'shared_modules') or not self.peft_config.shared_modules:
            return
            
        # Get feature dimensions from first shared module as reference
        first_shared_key = self.peft_config.shared_modules[0]
        try:
            parent, target, target_name = self._get_submodules(first_shared_key)
            in_features = target.in_features
            r = self.peft_config.r
            
            # r
            rr = []
            if isinstance(r, int):
                # Convert integer to string and extract each digit
                r_str = str(r)
                rr = [int(digit) for digit in r_str]
            else:
                # r is string
                ll = str(r)
                for i in range(len(ll)):
                    rr.append(int(ll[i]))
            
            # Create shared A matrices for each modality
            for i in range(self.peft_config.lora_nums):
                shared_A = nn.Linear(in_features, rr[i], bias=False)
                # Use kaiming initialization
                nn.init.kaiming_uniform_(shared_A.weight, a=math.sqrt(5))
                self.shared_A_matrices[f'shared_A{i}'] = shared_A
                
            # print(f"create {self.peft_config.lora_nums} shared-A matrice")
            
        except Exception as e:
            print(f"some fault rise when create shared-A matrice: {e}")

    def _create_shared_A_matrices(self,target_modules):
        """Create shared A matrices for each projection type separately"""
        if not hasattr(self.peft_config, 'shared_modules') or not self.peft_config.shared_modules:
            return
        
        # Initialize shared matrices dictionary structure
        # Format: {proj_type: {'shared_A0': Linear, 'shared_A1': Linear, ...}}
        self.shared_A_params = {}
        
        # Group shared modules by projection type
        shared_modules_by_type = self._group_shared_modules_by_type(target_modules)
        
        # Parse r into dimensions
        r = self.peft_config.r
        rr = []
        if isinstance(r, int):
            r_str = str(r)
            rr = [int(digit) for digit in r_str]
        else:
            rr = [int(char) for char in str(r)]

        default_dimensions = {
                'q_proj': 4096, 'k_proj': 4096, 'v_proj': 4096, 'o_proj': 4096,
                'gate_proj': 4096, 'up_proj': 4096, 'down_proj': 11008
            }

        # Create shared matrices for each projection type
        # 为每种投影类型创建共享参数
        for proj_type in target_modules:
            # 确定输入维度
            in_features = None
            
            # 尝试从实际模块获取维度
            if proj_type in shared_modules_by_type and shared_modules_by_type[proj_type]:
                try:
                    first_module_key = shared_modules_by_type[proj_type][0]
                    parent, target, target_name = self._get_submodules(first_module_key)
                    in_features = target.in_features
                except Exception as e:
                    print(f"Warning: Could not get input features for {proj_type}: {e}")
            
            # 如果无法获取，使用默认维度
            if in_features is None:
                in_features = default_dimensions.get(proj_type, 4096)
                print(f"Using default input features {in_features} for {proj_type}")
            
            # 为该类型创建一组共享参数
            type_params = {}
            for i in range(min(self.peft_config.lora_nums, len(rr))):
                # 创建参数（不创建Linear模块）
                param = nn.Parameter(torch.empty(rr[i],in_features)) #TODO
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                type_params[f'shared_A{i}'] = param
                
                # print(f"Created shared parameter for {proj_type}.shared_A{i}: "
                #     f"shape=({in_features}, {rr[i]})")
            
            self.shared_A_params[proj_type] = type_params
        
        # 为了向后兼容，也设置旧的shared_A_matrices（但内容改为参数）
        # 这样现有的代码可能还能工作
        self.shared_A_matrices = self.shared_A_params
        
        print("\nShared parameter creation summary:")
        for proj_type, params in self.shared_A_params.items():
            for key, param in params.items():
                print(f"  {proj_type}.{key}: shape={tuple(param.shape)}")

    def _group_shared_modules_by_type(self, target_modules):
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

    def _create_default_matrices_for_type(self, proj_type, rr):
        """Create default shared matrices for a projection type when module access fails"""
        # Default input dimensions based on projection type
        default_dimensions = {
            'q_proj': 4096, 'k_proj': 4096, 'v_proj': 4096, 'o_proj': 4096,
            'gate_proj': 4096, 'up_proj': 4096, 'down_proj': 11008  # down_proj has different input dim
        }
        
        in_features = default_dimensions.get(proj_type, 4096)
        
        # Create matrices for this type
        type_matrices = {}
        for i in range(min(self.peft_config.lora_nums, len(rr))):
            shared_A = nn.Linear(in_features, rr[i], bias=False)
            nn.init.kaiming_uniform_(shared_A.weight, a=math.sqrt(5))
            type_matrices[f'shared_A{i}'] = shared_A
        
        self.shared_A_matrices[proj_type] = type_matrices
        # print(f"Created default shared matrices for {proj_type} (in_features={in_features})")  

    def _extract_layer_name(self, module_name):
        """Extract layer names from module names"""
        import re
        # example: 'model.layers.0.self_attn.q_proj' -> 'model.layers.0'
        match = re.match(r'(.*\.layers\.\d+)', module_name)
        if match:
            return match.group(1)
        return module_name

    def _find_and_replace_hierarchical1(self,target_modules):
        """Replace modules hierarchically"""
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
            # "hierarchical_type": "important"  
        }
        
        # print("Process important layers (independent A matrix)...")
        for key in self.peft_config.important_modules:
            kwargs = base_kwargs.copy()
            # kwargs["hierarchical_type"] = "important"
            # kwargs["shared_A_matrices"] = None
            hierarchical_type = "important"
            shared_A_matrices = None
            self._replace_single_module(key, kwargs,hierarchical_type,shared_A_matrices)
        
        # print("Process shared layers (shared A matrix)...")
        for key in self.peft_config.shared_modules:
            kwargs = base_kwargs.copy()
            # kwargs["hierarchical_type"] = "shared"
            # kwargs["shared_A_matrices"] = self.shared_A_matrices
            hierarchical_type = "shared"
            shared_A_matrices = self.shared_A_matrices
            self._replace_single_module(key, kwargs,hierarchical_type,shared_A_matrices)

    def _find_and_replace_hierarchical(self,target_modules):
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
        
        # print("Process important layers (independent A matrix)...")
        for key in self.peft_config.important_modules:
            kwargs = base_kwargs.copy()
            hierarchical_type = "important"
            shared_A_matrices = None
            self._replace_single_module(key, kwargs, hierarchical_type, shared_A_matrices)
        
        # print("Process shared layers (type-specific shared A matrix)...")
        for key in self.peft_config.shared_modules:
            kwargs = base_kwargs.copy()
            hierarchical_type = "shared"
            
            # Determine projection type for this module
            proj_type = self._get_projection_type_from_key(key, target_modules)
            
            # Pass the appropriate shared matrices for this projection type
            if proj_type and proj_type in self.shared_A_matrices:
                shared_params = self.shared_A_matrices[proj_type]
                # print(f"  {key}: using {proj_type} shared params")
            else:
                # If no matrices found for this type, use None (will create independent)
                shared_params = None
                # print(f"  WARNING: {key}: no shared matrices for type {proj_type}")
            
            self._replace_single_module(key, kwargs, hierarchical_type, shared_params)

    def _get_projection_type_from_key(self, key, target_modules):
        """Extract projection type from module key"""
        for proj_type in target_modules:
            if f'.{proj_type}' in key or key.endswith(f'.{proj_type}'):
                return proj_type
        return None

    def _replace_single_module(self, key, kwargs,hierarchical_type,shared_A_matrices):
        try:
            parent, target, target_name = self._get_submodules(key)
            bias = target.bias is not None

            if isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                # print(f"Replacing module: {key}")
                # print(f"  Original shape: in_features={target.in_features}, out_features={target.out_features}")
                # print(f"  Hierarchical type: {hierarchical_type}")
            
                # if hierarchical_type == "shared" and shared_A_matrices:
                #     # print(f"  Shared matrices available: {list(shared_A_matrices.keys())}")
                #     for name, param in shared_A_matrices.items():
                #         # print(f"    {name}: shape={param.shape}")

                new_module = LinearW(
                    target.in_features, 
                    target.out_features, 
                    bias=bias, 
                    hierarchical_type = hierarchical_type,
                    shared_A_matrices = shared_A_matrices,
                    **kwargs
                )
                
                self._replace_module(parent, target_name, new_module, target)
        except Exception as e:
            print(f"some fault rise when replace module {key} : {e}")

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
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found: # here
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None

                if isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    # new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs) 
                    new_module = LinearW(target.in_features, target.out_features, bias=bias, **kwargs) # NOTE: replace with new lora

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
        self.r=[]
        ll=str(r)
        for i in range(len(ll)):
            self.r.append(int(ll[i]))
        
        # print(self.r)
        # print('$$$$$$$$$$$$$$')
        

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
        hierarchical_type: str = "important",  #NOTE # Added: Layer types
        shared_A_matrices: dict = None, # TODO
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.loramethod= loramethod

        self.lora_num = lora_nums
        self.blc_alpha = blc_alpha
        self.blc_weight = blc_weight

        self.reserved_modality=reserved_modality
        
        
        self.fan_in_fan_out = fan_in_fan_out

        rr=[]
        ll=str(r)
        for i in range(len(ll)):
            rr.append(int(ll[i]))


        if rr[0] > 0:
            # for i in range(self.lora_num):
            #     setattr(self, f"lora_A{i}", nn.Linear(in_features, rr[i], bias=False))

            # for i in range(1):
            #     setattr(self, f"lora_B{i}", nn.Linear(rr[0], out_features, bias=False))
            
            #NOTE share-v1
            # Determine the initialization method for Matrix A based on the layer type
            if self.hierarchical_type == "important":
                # important layers (independent A matrix)
                for i in range(self.lora_num):
                    setattr(self, f"lora_A{i}", nn.Linear(in_features, rr[i], bias=False))
                    setattr(self, f"lora_A{i}_is_shared", False)  # 标记为非共享
                # print(f"Create an independent matrix A for the important layers: {self.hierarchical_type}")
            else:
                if shared_A_matrices and isinstance(shared_A_matrices, dict):
                    for i in range(self.lora_num):
                        shared_A_key = f'shared_A{i}'
                        if shared_A_key in shared_A_matrices:
                            # 获取共享参数
                            shared_param = shared_A_matrices[shared_A_key]
                            
                            # 创建Linear层，使用共享参数作为权重
                            lora_A = nn.Linear(in_features, rr[i], bias=False)
                            
                            # 关键：将共享参数赋值给Linear层的权重
                            # 这样所有使用该参数的层都共享同一参数对象
                            # lora_A.weight.data = shared_param.t().contiguous() #TODO 逆置了一下
                            lora_A.weight = shared_param

                            setattr(self, f"lora_A{i}", lora_A)
                            # print(f"Using shared parameter {shared_A_key}")
                        else:
                            # 如果找不到共享参数，创建独立矩阵
                            lora_A = nn.Linear(in_features, rr[i], bias=False)
                            nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5)) #TODO
                            setattr(self, f"lora_A{i}", lora_A)
                            print(f"Warning: {shared_A_key} not found, created independent")
                else:
                    for i in range(self.lora_num):
                        lora_A = nn.Linear(in_features, rr[i], bias=False)
                        nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5)) #TODO
                        setattr(self, f"lora_A{i}", lora_A)
                    print(f"No shared parameters provided, created independent matrices")

            self.scaling = []

            ## scaling text lora
            for i in range(1):
                self.scaling.append(self.lora_alpha / self.r[i])

            
             # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

        if hasattr(self, "lora_A0"): 
            for i in range(self.lora_num):
                nn.init.kaiming_uniform_(getattr(self, f"lora_A{i}").weight, a=math.sqrt(5))
            
            for i in range(1): 
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

                N_t=torch.sum(question_mask[i,:,:])


                score = torch.matmul(query, key.transpose(-2, -1))/ math.sqrt(N_t)  # shape: (1, token_num, question_length)
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

                N_t=torch.sum(question_mask[i,:,:])

                score = torch.matmul(query, key.transpose(-2, -1))/ math.sqrt(N_t)  # shape: (1, token_num, question_length)
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

                N_t=torch.sum(question_mask[i,:,:])


                score = torch.matmul(query, key.transpose(-2, -1))/ math.sqrt(N_t) 
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

                N_t=torch.sum(question_mask[i,:,:])

                score = torch.matmul(query, key.transpose(-2, -1))/ math.sqrt(N_t)
                score = torch.softmax(score, dim=-1)
                output = torch.matmul(score, value)  # shape: (1, token_num, 4)
                attention_outputs=audio_mask[i,:,:]*output
                new_audio[i,:,:]=audio_token[i,:,:]+attention_outputs*self.blc_weight
            

            input_b=[output_a[0],new_video,new_audio]
            input_b=sum(input_b)


            output_b=getattr(self, f"lora_B0")(input_b)

            result=output_b+result
            
            return result

def init_lora_mem(lora_mem, k1):
    """
    Initialize lora_mem as:
    1. First k1 rows as positive vectors (e.g., text modality), 
       remaining (k-k1) rows as negative vectors (e.g., image modality)
    2. Numerical range matches the output energy of kaiming_uniform initialized lora_A
    3. Non-orthogonal, preserving weak correlations between memory units
    """
    k, r = lora_mem.shape
    k2 = k - k1  # Number of negative vectors
    
    # Core parameters: balance polarity and energy matching
    mean_pos = 0.05    # Positive vector mean (slightly positive, ensuring polarity)
    mean_neg = -0.05   # Negative vector mean (slightly negative, ensuring polarity)
    std = 0.14         # Standard deviation, calculated to match kaiming_uniform output energy
    
    # First k1 rows: positive vectors (mean > 0, matching text modality polarity)
    lora_mem.data[:k1] = torch.normal(mean=mean_pos, std=std, size=(k1, r))
    
    # Remaining k2 rows: negative vectors (mean < 0, matching image modality polarity)
    lora_mem.data[k1:] = torch.normal(mean=mean_neg, std=std, size=(k2, r))

class LinearW(nn.Linear, LoraLayer):
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
        hierarchical_type: str = "important",  #NOTE # Added: Layer types
        shared_A_matrices: dict = None, # TODO
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        #TODO
        # self.hierarchical_type = kwargs.pop('hierarchical_type', "important")
        # self.shared_A_matrices = kwargs.pop('shared_A_matrices', None)

        self.loramethod= loramethod

        self.lora_num = lora_nums
        self.blc_alpha = blc_alpha
        self.blc_weight = blc_weight

        self.reserved_modality=reserved_modality
    
        self.fan_in_fan_out = fan_in_fan_out
        self.hierarchical_type = hierarchical_type
        self.shared_A_matrices = shared_A_matrices


        rr=[]
        ll=str(r)
        for i in range(len(ll)):
            rr.append(int(ll[i]))


        if rr[0] > 0:
            k = 32
            self.kernel = nn.SiLU()
            # for i in range(self.lora_num):
            #     setattr(self, f"lora_A{i}", nn.Linear(in_features, rr[i], bias=False))

            # Determine the initialization method for Matrix A based on the layer type
            if self.hierarchical_type == "important":
                # important layers (independent A matrix)
                for i in range(self.lora_num):
                    setattr(self, f"lora_A{i}", nn.Linear(in_features, rr[i], bias=False))
                    setattr(self, f"lora_A{i}_is_shared", False)  # 标记为非共享
                # print(f"Create an independent matrix A for the important layers: {self.hierarchical_type}")
            else:
                if shared_A_matrices and isinstance(shared_A_matrices, dict):
                    for i in range(self.lora_num):
                        shared_A_key = f'shared_A{i}'
                        if shared_A_key in shared_A_matrices:
                            # 获取共享参数
                            shared_param = shared_A_matrices[shared_A_key]
                            
                            # 创建Linear层，使用共享参数作为权重
                            lora_A = nn.Linear(in_features, rr[i], bias=False)
                            
                            # 关键：将共享参数赋值给Linear层的权重
                            # 这样所有使用该参数的层都共享同一参数对象
                            # lora_A.weight.data = shared_param.t().contiguous() #TODO 逆置了一下
                            lora_A.weight = shared_param

                            setattr(self, f"lora_A{i}", lora_A)
                            # print(f"Using shared parameter {shared_A_key}")
                        else:
                            # 如果找不到共享参数，创建独立矩阵
                            lora_A = nn.Linear(in_features, rr[i], bias=False)
                            nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5)) #TODO
                            setattr(self, f"lora_A{i}", lora_A)
                            print(f"Warning: {shared_A_key} not found, created independent")
                else:
                    for i in range(self.lora_num):
                        lora_A = nn.Linear(in_features, rr[i], bias=False)
                        nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5)) #TODO
                        setattr(self, f"lora_A{i}", lora_A)
                    print(f"No shared parameters provided, created independent matrices")

            lora_mem_matrix = nn.Parameter(torch.empty(k, rr[i]))
            setattr(self, f"lora_mem", lora_mem_matrix)
            # 0 for text | 1 for video | 2 for audio
            self.lora_trace = nn.Parameter(torch.zeros(3, k))  
            for i in range(2):
                setattr(self, f"lora_B{i}", nn.Linear(rr[0], out_features, bias=False))
            

            self.scaling = []

            ## scaling text lora
            for i in range(1):
                self.scaling.append(self.lora_alpha / self.r[i])

            
             # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

        if hasattr(self, "lora_A0"): 
            # Initialize A matrices only for important layers (independent matrices)
            if self.hierarchical_type == "important":
                for i in range(self.lora_num):
                    nn.init.kaiming_uniform_(getattr(self, f"lora_A{i}").weight, a=math.sqrt(5))
            # Initialize memory matrix
            nn.init.kaiming_uniform_(getattr(self, f"lora_mem"), a=math.sqrt(5))
                # Alternative initialization methods (commented out):
            # init_lora_mem(getattr(self, f"lora_mem{i}"), k1=16)
            # nn.init.orthogonal_(getattr(self, f"lora_mem{i}"))
            m = 32 // 3 # k=32, divide into 3 regions
            # 1. Apply initial weights to each scenario's dedicated area (weak bias, e.g., 0.02)
            init_weight = 0.05  # Initial weight value (can be adjusted based on dependency level)
            
            # Row 0: text modality -> activate first dedicated area (0 ~ m-1)
            self.lora_trace.data[0, 0:m] = init_weight
            
            # Row 1: video modality -> activate second dedicated area (m ~ 2m-1)
            self.lora_trace.data[1, m:2*m] = init_weight
            
            # Row 2: audio modality -> activate third dedicated area (2m ~ 3m-1)
            self.lora_trace.data[2, 2*m:] = init_weight

            for i in range(2): # Initialize B matrices (up-projection matrices)
                # nn.init.kaiming_uniform_(getattr(self, f"lora_Ar{i}"), a=math.sqrt(5))
                nn.init.zeros_(getattr(self, f"lora_B{i}").weight)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)

        for i in range(self.lora_num):
            getattr(self, f"lora_A{i}").train(mode)
            # getattr(self, f"lora_mem{i}").train(mode)
        for i in range(2):
            # getattr(self, f"lora_Ar{i}").train(mode)
            getattr(self, f"lora_B{i}").train(mode)

    def eval(self):
        nn.Linear.eval(self)
        for i in range(self.lora_num):
            getattr(self, f"lora_A{i}").eval()
            # getattr(self, f"lora_mem{i}").eval()
        for i in range(2):
            # getattr(self, f"lora_Ar{i}").eval()
            getattr(self, f"lora_B{i}").eval()

    def act_memory_soft(self, x, mode=0):
        k, r = self.lora_mem.shape # [k, r]
        b, n, _ = x.shape
        #----------- Compute current memory activation -----------#
        # score: [B, N, k: memory_num]
        score_save = torch.einsum('bnr,kr->bnk', x, self.lora_mem) / math.sqrt(n) # [B, N, k]
        score = torch.softmax(score_save, dim=-1)
        #----------- LTP/LTD -----------#
        if mode == 0:
            mis_type = torch.ones(b, dtype=torch.long)*0
        elif mode == 1:
            mis_type = torch.ones(b, dtype=torch.long)*1
        elif mode == 2:
            mis_type = torch.ones(b, dtype=torch.long)*2
        else:
            raise Exception("Wrong mode setting!")
        lora_traces = self.lora_trace[mis_type]
        # print(self.lora_trace)
        adjusted_score = score + lora_traces.unsqueeze(1)

        act_mem = adjusted_score@self.lora_mem   # [b, n, r]

        return act_mem  # [batch, length, r]

    def cross_attention(self, query, key_value):
        len_k = key_value.shape[1]
        # 1. feature projection
        q = self.kernel(query)
        k = self.kernel(key_value)

        kv = torch.einsum('bnr, bns -> brs', k, key_value)  # (b, r, r)

        # 2. (Q * kv) * z -> [b, q_len, r]
        out = torch.einsum('bnr, brs -> bns', q, kv)  
        out = out
        return out + query  

    def forward(self, x: torch.Tensor, modality_mask:List[torch.Tensor]=None):

        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)        

            
        ## infer not first forward only text token
        # move scaling to last
        if(('test' in self.loramethod) &(x.size(1)==1) ):
            # only text token
            output_a=getattr(self, f"lora_A0")(self.lora_dropout(x))*self.scaling[0]
            input_b=output_a
            new_loraA = self.act_memory_soft(x=input_b, mode=0)  # [b, n, r]
            output_b=getattr(self, f"lora_B0")(input_b) + getattr(self, f"lora_B1")(new_loraA)

            result=output_b+result

            return result


        ## infer first forward
        if(('test' in self.loramethod) & (x.size(1)!=1)):

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
            new_loraAAA = []
            for i in range(self.lora_num):
                input_b = getattr(self, f"lora_A{i}")(self.lora_dropout(only_inputs[i]))
                new_loraA = self.act_memory_soft(x=input_b, mode=i)
                output_a.append(input_b)    # store multimodal A proj
                new_loraAAA.append(new_loraA)   # store multimodal mem act

            ### video_token: cross attention per sample
            video_token=new_loraAAA[1]
            video_token_base=output_a[1]
            ### audio_token: cross attention per sample
            audio_token=new_loraAAA[2]
            audio_token_base=output_a[2]
            question_token_base=output_a[0]*question_mask
            new_video=torch.zeros_like(video_token)
            new_audio=torch.zeros_like(audio_token)
            # new_quest=torch.zeros_like(question_token)

            for i in range(question_token_base.size(0)):
                mem_v=video_token[i,:,:].unsqueeze(0)
                mem_a=audio_token[i,:,:].unsqueeze(0)
                base_v=video_token_base[i,-1:,:].unsqueeze(0)
                base_a=audio_token_base[i,-1:,:].unsqueeze(0)
                ## get question tokens
                indices = torch.where(question_mask[i,:,:] == 1)[0]
                base_t=question_token_base[i,indices[0]:indices[-1]+1,:].unsqueeze(0)

                supp_v = self.cross_attention(query=mem_v, key_value=torch.cat([base_a,base_t], dim=1))  # shape: (1, token_num, 4)
                supp_a = self.cross_attention(query=mem_a, key_value=torch.cat([base_v,base_t], dim=1))  # shape: (1, token_num, 4)
                # supp_v = self.cross_attention(query=mem_v, key_value=torch.cat([base_t, base_a], dim=1))
                # supp_a = self.cross_attention(query=mem_a, key_value=torch.cat([base_t, base_v], dim=1)) #TODO
                # supp_t = self.cross_attention(query=mem_t, key_value=torch.cat([mem_a,mem_v], dim=1))  # shape: (1, token_num, 4)

                supp_vi=video_mask[i,:,:]*supp_v
                supp_au=audio_mask[i,:,:]*supp_a

                new_video[i,:,:]=supp_vi
                new_audio[i,:,:]=supp_au
                # new_quest[i,indices[0]:indices[-1]+1,:]=supp_t
            

            input_b=sum(output_a)    # base
            input_b1 = [new_loraAAA[0], new_video, new_audio]
            input_b1 = sum(input_b1)

            output_b=getattr(self, f"lora_B0")(input_b)*self.scaling[0] + getattr(self, f"lora_B1")(input_b1)*self.scaling[0]

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
            new_loraAAA = []
            for i in range(self.lora_num):
                input_b = getattr(self, f"lora_A{i}")(self.lora_dropout(only_inputs[i]))
                new_loraA = self.act_memory_soft(x=input_b, mode=i)
                output_a.append(input_b)
                new_loraAAA.append(new_loraA)
            
            ### video_token: cross attention per sample
            video_token=new_loraAAA[1]
            video_token_base=output_a[1]
            ### audio_token: cross attention per sample
            audio_token=new_loraAAA[2]
            audio_token_base=output_a[2]
            question_token_base=output_a[0]*question_mask
            new_video=torch.zeros_like(video_token)
            new_audio=torch.zeros_like(audio_token)
            # new_quest=torch.zeros_like(question_token)

            for i in range(question_token_base.size(0)):
                mem_v=video_token[i,:,:].unsqueeze(0)
                mem_a=audio_token[i,:,:].unsqueeze(0)
                base_v=video_token_base[i,-1:,:].unsqueeze(0)
                base_a=audio_token_base[i,-1:,:].unsqueeze(0)
                ## get question tokens
                indices = torch.where(question_mask[i,:,:] == 1)[0]
                base_t=question_token_base[i,indices[0]:indices[-1]+1,:].unsqueeze(0)

                supp_v = self.cross_attention(query=mem_v, key_value=torch.cat([base_t, base_a], dim=1))  # shape: (1, token_num, 4)
                supp_a = self.cross_attention(query=mem_a, key_value=torch.cat([base_t, base_v], dim=1))  # shape: (1, token_num, 4)
                # supp_t = self.cross_attention(query=mem_t, key_value=torch.cat([mem_a,mem_v], dim=1))  # shape: (1, token_num, 4)

                new_video[i,:,:]=video_mask[i,:,:]*supp_v
                new_audio[i,:,:]=audio_mask[i,:,:]*supp_a
                # new_quest[i,indices[0]:indices[-1]+1,:]=supp_t
            

            input_b=sum(output_a)    # base
            input_b1 = [new_loraAAA[0], new_video, new_audio]
            input_b1 = sum(input_b1)

            output_b=getattr(self, f"lora_B0")(input_b)*self.scaling[0] + getattr(self, f"lora_B1")(input_b1)*self.scaling[0]

            result=output_b+result
            
            return result



        
