import os
import sys

sys.path.append(os.getcwd())
import itertools
import pathlib
from os.path import exists, join

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    import torch_npu
    import torch_npu_acc
    from torch_npu.contrib import transfer_to_npu
except:
    print('no npu!')
import argparse

import torch.distributed as dist
import transformers
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from configs.unified_config import (DataArguments, InferenceArguments,
                                    ModelArguments, TrainingArguments)
from dataset.unified_dataset import get_dataset_collator
from utils.deepspeed_utils import *
from typing import Any
from utils.util import (find_all_linear_names, load_ckpt, prepare_sample,
                        set_seed, write2json)

local_rank = None

from torch.utils.data.distributed import DistributedSampler


def print_main(*args: Any, **kwargs: Any):
    if local_rank == 0:
        print(*args, **kwargs)
    return

class Test_DistributedSampler(DistributedSampler): 
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False): 
        super(Test_DistributedSampler, self).__init__(dataset, num_replicas, rank, shuffle)
        N = len(self.dataset)
        R = self.num_replicas
        base_num_samples = N // R
        remainder = N % R
        if self.rank < remainder:
            self.num_samples = base_num_samples + 1
        else:
            self.num_samples = base_num_samples

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples


def inference(dataloader,ckpt_dir,model,tokenizer,task, mode):
    save_dir = join(ckpt_dir,f'inference_results_bs_6_{mode}')
    os.makedirs(save_dir,exist_ok=True)

    #==================== CHECK GENERATION CONFIGS ====================#
    # 1. definition of used generation parameters
    gen_params = {
        'use_cache': True,
        'max_new_tokens': 500,
        # you can change the used parameetrs, such as:
        #--------- here is for Qwen3-8B [no thinking mode] ---------#
        'temperature': 0.7,
        'top_p': 0.8,
        'repetition_penalty': 1.1
        #--------- here is for Qwen3-8B [no thinking mode] ---------#
    }

    # 2. print generation configs
    print_main("="*50)
    print_main("Generation Configuration (Before Inference Loop)")
    print_main("="*50)
    # print manual settings
    print_main("Manual Generation Parameters:")
    for k, v in gen_params.items():
        print_main(f"  - {k}: {v}")

    # print default configs
    print_main("\nModel Default Generation Parameters (not overridden):")
    default_gen_config = model.generation_config.to_dict()
    # select the key generation paramters
    core_gen_keys = ['do_sample', 'temperature', 'top_p', 'top_k', 
                    'repetition_penalty', 'num_beams', 'eos_token_id', 'pad_token_id']
    for k in core_gen_keys:
        if k not in gen_params and k in default_gen_config:
            print_main(f"  - {k}: {default_gen_config[k]}")
    print_main("="*50 + "\n")
    #==================== CHECK GENERATION CONFIGS ====================#

    pbar = tqdm(total=len(dataloader),desc=f'inference {task}')
    fp = join(save_dir,f'inference_{task}.jsonl')
    for step, sample in enumerate(dataloader):
        batch_metadata = sample.pop('batch_metadata')
        bs = len(batch_metadata)
        sample = prepare_sample(data = sample)

        # # ==================== 修复逻辑：将 List[Tensor] 转换为 Tensor ====================
        # device = next(model.parameters()).device
        # for k, v in sample.items():
        #     if isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
        #         # 将 [Tensor, Tensor, Tensor, Tensor] 堆叠成 [4, Seq_Len]
        #         sample[k] = torch.stack(v, dim=0).to(device)
        #     elif isinstance(v, torch.Tensor):
        #         sample[k] = v.to(device)
        # # ==============================================================================

        # # ==================== DEBUG SAMPLE TYPE ====================
        # if step == 0:  # 只打印第一个 batch 避免刷屏
        #     print_main(f"\n[DEBUG] Global Sample Type: {type(sample)}")
        #     if isinstance(sample, dict):
        #         for k, v in sample.items():
        #             if isinstance(v, torch.Tensor):
        #                 print_main(f"  - Key: {k:15} | Type: Tensor | Shape: {list(v.shape)} | Device: {v.device}")
        #             elif isinstance(v, list):
        #                 # 如果是 list，尝试展示其内部元素的类型（如 [Tensor, Tensor] 或 [int, int]）
        #                 inner_type = type(v[0]) if len(v) > 0 else "Empty"
        #                 print_main(f"  - Key: {k:15} | Type: LIST   | Length: {len(v)} | Inner Type: {inner_type}")
        #             else:
        #                 print_main(f"  - Key: {k:15} | Type: {type(v)}")
        #     elif isinstance(sample, tuple):
        #         print_main(f"  [!!! WARNING !!!] Sample is a TUPLE. Length: {len(sample)}")
        #         if len(sample) > 0:
        #             print_main(f"  First element type: {type(sample[0])}")
        # # ===========================================================

        # #NOTE qwen3
        # device = next(model.parameters()).device
        # for k, v in sample.items():
        #     if isinstance(v, list):
        #         sample[k] = torch.tensor(v).to(device)
        #     elif isinstance(v, torch.Tensor):
        #         sample[k] = v.to(device)

        sample.update(
            gen_params
        )
        with torch.no_grad():
            # print(sample.keys())
            output = model.generate(**sample)
            output = tokenizer.batch_decode(output,skip_special_tokens=False)
        for i in range(bs):
            metadata = batch_metadata[i]
            metadata['predict'] = output[i]

            write2json(fp=fp,dict_data=metadata)
        
        pbar.update(1)
    pbar.close()


def train(attn_implementation=None):
    global local_rank
    # Add this to set seed manually
    extra_parser = argparse.ArgumentParser()
    extra_parser.add_argument("--seed", type=int, default=42, help="Manual set random seed")
    extra_parser.add_argument("--mode", type=str, default='test', help="Manual set testing mode")
    extra_args, remaining_argv = extra_parser.parse_known_args()  # 与原有参数不冲突
    set_seed(extra_args.seed)
    print_main(f"[CHECK THIS] Seed is set to {extra_args.seed}")

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, InferenceArguments))
    model_args, data_args, training_args, infer_args = parser.parse_args_into_dataclasses(args=remaining_argv)

    if model_args.llm_name == 'llama':
        d_model = 4096
    elif model_args.llm_name == 'qwen2-1.5b':    # 1.5B-instruct
        d_model = 1536
    elif model_args.llm_name == 'qwen2-7b':    # 7B-instruct
        d_model = 3584
    elif model_args.llm_name == 'qwen3-4b':    # 4B-instruct-2507
        d_model = 2560
    elif model_args.llm_name == 'qwen3-8b':    # 8B
        d_model = 4096

    if '-' in model_args.llm_name:
        model_type = model_args.llm_name.split('-')[0]
        model_args.llm_name = model_type

    local_rank = training_args.local_rank
    compute_dtype = torch.float32
    if training_args.fp16:
        compute_dtype = torch.float16
    elif training_args.bf16:
        compute_dtype = torch.bfloat16
    
    pretrain_model_name_or_path = model_args.model_name_or_path
    if model_args.llm_name == 'llama':
        from transformers import LlamaConfig

        from models.unified_llama import UnifiedForCausalLM
        config = LlamaConfig.from_pretrained(pretrain_model_name_or_path, local_files_only=True)
        config._attn_implementation = attn_implementation
        model = UnifiedForCausalLM.from_pretrained(
            pretrain_model_name_or_path,
            config=config,
            dtype=compute_dtype
        )

    elif model_args.llm_name == 'qwen2':
        from transformers import Qwen2Config
        from models.unified_qwen2 import UnifiedForCausalLM
        config = Qwen2Config.from_pretrained(pretrain_model_name_or_path, local_files_only=True)
        config._attn_implementation = attn_implementation
        model = UnifiedForCausalLM.from_pretrained(
            pretrain_model_name_or_path,
            config = config,
            dtype = compute_dtype,
            trust_remote_code=True
        )

    elif model_args.llm_name == 'qwen3':
        from transformers import Qwen3Config
        from models.unified_qwen3 import UnifiedForCausalLM
        config = Qwen3Config.from_pretrained(pretrain_model_name_or_path, local_files_only=True)
        config._attn_implementation = attn_implementation
        model = UnifiedForCausalLM.from_pretrained(
            pretrain_model_name_or_path,
            config = config,
            dtype = compute_dtype,
            trust_remote_code=True  # for control qwen3-8B's thinking mode
        )

    model.config.use_cache = True

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft_hyper import LoraConfig, get_peft_model
        lora_trainable="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"
        target_modules = lora_trainable.split(',')
        lora_rank = training_args.lora_r
        lora_alpha = training_args.lora_alpha
        print('lora_alpha: ',lora_alpha)
        print()
        lora_dropout = 0.05
        # lora_nums = int(len(str(training_args.lora_r))) #NOTE moka
        lora_nums = 3 #NOTE fix

        modules_to_save = None
        peft_config = LoraConfig(
            task_type = "CAUSAL_LM",
            lora_type = training_args.loratype,
            target_modules = target_modules,
            inference_mode = False,
            r = lora_rank, 
            loramethod= training_args.loramethod,
            reserved_modality=training_args.reserved_modality,
            lora_alpha = lora_alpha,
            lora_dropout = lora_dropout,
            lora_nums = lora_nums,
            blc_alpha= training_args.blc_alpha,
            blc_weight=training_args.blc_weight,
        )
        model = get_peft_model(model, peft_config)

    
    if model_args.llm_name == 'llama':
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            pretrain_model_name_or_path,
            padding_side="left",
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    elif model_args.llm_name == 'qwen2':
        from transformers import Qwen2Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(
            pretrain_model_name_or_path,
            padding_side="left",
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    elif model_args.llm_name == 'qwen3':
        from transformers import Qwen2Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(
            pretrain_model_name_or_path,
            padding_side="left",
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"[CHECK THIS] Switch pad_token_id to {tokenizer.eos_token_id}.")
    
    ori_tokenizer_vocab_nums = len(tokenizer)
    model.get_model().pad_token_id = tokenizer.pad_token_id
    model.get_model().init_multimodal_modules(visual_branch=training_args.visual_branch,
                                              audio_branch=training_args.audio_branch,
                                              d_model=d_model,vit_ckpt_path=model_args.vit_ckpt_path,
                                              select_layer_list=model_args.select_layer_list,
                                              select_feature=model_args.select_feature,image_size=model_args.image_size,
                                              patch_size=model_args.patch_size,visual_query_token_nums=model_args.visual_query_token_nums,
                                              audio_query_token_nums=model_args.audio_query_token_nums,BEATs_ckpt_path=model_args.BEATs_ckpt_path)

    model.initialize_MM_tokenizer(tokenizer)
    MM_tokenizer_vocab_nums = len(tokenizer)
    print_main('ori_tokenizer_vocab_nums: ',ori_tokenizer_vocab_nums, ' MM_tokenizer_vocab_nums: ',MM_tokenizer_vocab_nums)


    ckpt_dir = infer_args.ckpt_dir

    if "checkpoint" in ckpt_dir:
        print_main("[LOAD] loading merged checkpoint")
        # merged checkpoints
        ckpt_path = join(ckpt_dir,'finetune_weights.bin')
        ckpt = torch.load(ckpt_path,map_location='cpu')
        model.load_state_dict(ckpt,strict=False)
        print_main(f'load ckpt from {ckpt_path} finished...')

    else:
        print_main("[LOAD] loading seperated checkpoints")
        ckpt_path = join(ckpt_dir,'non_lora_trainables.bin')
        ckpt = torch.load(ckpt_path,map_location='cpu')
        model.load_state_dict(ckpt,strict=False)
        print_main(f'load ckpt from {ckpt_path} finished...')
        # print("===== 从 non_lora_trainables.bin 加载的权重 =====")
        # for weight_name in ckpt.keys():
        #     print(weight_name)
        ckpt_path = join(ckpt_dir,'adapter_model.bin')
        ckpt = torch.load(ckpt_path,map_location='cpu')
        model.load_state_dict(ckpt,strict=False)
        print_main(f'load ckpt from {ckpt_path} finished...')
        # print("===== 从 adapter_model.bin 加载的权重 =====")
        # for weight_name in ckpt.keys():
        #     print(weight_name)

    model.eval()
    model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
    
    image_processor = model.module.get_model().visual_encoder.image_processor if training_args.visual_branch else None
    dataset, collator = get_dataset_collator(data_args=data_args, tokenizer=tokenizer, 
                                             image_processor=image_processor,mode=extra_args.mode)
    
    # qwen3-8B need to set batch_size = 4, otherwise the cuda will be out of memory
    batch_size = 4

    sampler = Test_DistributedSampler(dataset,num_replicas=torch.distributed.get_world_size(),rank=local_rank,shuffle=False)

    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,sampler=sampler,collate_fn=collator,drop_last=False,num_workers=4)

    if data_args.avqa_task:
        inference(dataloader=dataloader,ckpt_dir=ckpt_dir,model=model.module,tokenizer=tokenizer,task = 'avqa', mode=extra_args.mode)
    if data_args.ave_task:
        inference(dataloader=dataloader,ckpt_dir=ckpt_dir,model=model.module,tokenizer=tokenizer,task = 'ave')

    if dist.is_initialized():   # check whether the process group is initialized
        dist.destroy_process_group()
        print(f"Rank {local_rank} destroyed process group successfully")


if __name__ == "__main__":
    train()

