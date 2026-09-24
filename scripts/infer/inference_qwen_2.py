# -*- coding: utf-8 -*-
import os,sys
sys.path.append(os.getcwd())
from os.path import join,exists
import pathlib
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Subset
from PIL import Image
import torch
import itertools
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    print('no npu!')

from torch.utils.data import DataLoader
import transformers

from configs.unified_config import ModelArguments,DataArguments,TrainingArguments,InferenceArguments

from dataset.unified_dataset import get_dataset_collator
from utils.util import set_seed,find_all_linear_names,prepare_sample,write2json,load_ckpt
from utils.deepspeed_utils import *
import argparse

local_rank = None



def inference_avqa(dataset,collator,dataloader,ckpt_dir,model,tokenizer,num, seed: int, mode: str):
    print(f"lcoal rank: {num}")
    save_dir = join(ckpt_dir,f'inference_avqa_bs6_seed42/bs6_tf32_mt500_seed{seed}_{mode}')
    os.makedirs(save_dir,exist_ok=True)
    
    name='results'+str(num)+'.jsonl'
    fp = join(save_dir,name)

    all_num=len(dataloader)
    avg=int(all_num/4)

    if(num==0):
        start_point=0
        end_point=avg
    elif(num==1):
        start_point=avg
        end_point=avg*2
    elif(num==2):
        start_point=avg*2
        end_point=avg*3
    elif(num==3):
        start_point=avg*3
        end_point=all_num
    # elif(num==4):
    #     start_point=avg*4
    #     end_point=all_num
    
    # start_point=0
    # end_point=all_num



    print('this is fold: ', num)
    print('start_point: ', start_point)
    print('end_point: ', end_point)
    subset_indices = list(range(start_point, end_point))
    subset = Subset(dataset, subset_indices)
    subset_dataloader = DataLoader(subset, batch_size=6, shuffle=False,collate_fn=collator,drop_last=False,num_workers=4,pin_memory=True,prefetch_factor=2)
    
    pbar = tqdm(total=len(subset_dataloader),desc=f'inference avqa')
    # pbar = tqdm(total=len(dataloader),desc=f'inference avqa')

    for step, sample in enumerate(subset_dataloader):
    # for step, sample in enumerate(dataloader):

        batch_metadata = sample.pop('batch_metadata')
        bs = len(batch_metadata)
        sample = prepare_sample(data = sample)
        sample.update(
            {
                'use_cache':True,
                'max_new_tokens':500,
                # NOTE for qwen2-7b, note this is not qwen2-7b-instruct 
                "do_sample": True,
                "repetition_penalty": 1.05,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
            }
        )
        with torch.no_grad():
            # NOTE: Not recommended to exceed 1.5
            # output = model.generate(**sample, repetition_penalty=1.1)
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
    extra_args, remaining_argv = extra_parser.parse_known_args()  
    set_seed(extra_args.seed)
    print(f"[CHECK THIS] Seed is set to {extra_args.seed}")

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, InferenceArguments))
    model_args, data_args, training_args, infer_args = parser.parse_args_into_dataclasses(args=remaining_argv)

    if model_args.llm_name == 'llama':
        d_model = 4096
    elif model_args.llm_name == 'qwen2':    # 1.5B-instruct
        d_model = 1536
    elif model_args.llm_name == 'qwen2-7b':    # 7B
        d_model = 3584

    local_rank = training_args.local_rank
    compute_dtype = torch.float32
    if training_args.fp16:
        compute_dtype = torch.float16
    elif training_args.bf16:
        compute_dtype = torch.bfloat16
    
    pretrain_model_name_or_path = model_args.model_name_or_path
    if model_args.llm_name == 'llama':
        from models.unified_llama import UnifiedForCausalLM
        from transformers import LlamaConfig
        config = LlamaConfig.from_pretrained(pretrain_model_name_or_path, local_files_only=True)
        config._attn_implementation = attn_implementation
        model = UnifiedForCausalLM.from_pretrained(
            pretrain_model_name_or_path,
            config=config,
            torch_dtype=compute_dtype
        )
    elif model_args.llm_name == 'qwen2':
        from models.unified_qwen2 import UnifiedForCausalLM
        from transformers import Qwen2Config
        config = Qwen2Config.from_pretrained(pretrain_model_name_or_path, local_files_only=True)
        config._attn_implementation = attn_implementation
        model = UnifiedForCausalLM.from_pretrained(
            pretrain_model_name_or_path,
            config = config,
            torch_dtype = compute_dtype
        )
    elif model_args.llm_name == 'qwen2-7b':
        from models.unified_qwen2 import UnifiedForCausalLM
        from transformers import Qwen2Config
        config = Qwen2Config.from_pretrained(pretrain_model_name_or_path, local_files_only=True)
        config._attn_implementation = attn_implementation
        model = UnifiedForCausalLM.from_pretrained(
            pretrain_model_name_or_path,
            config = config,
            torch_dtype = compute_dtype
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
        from peft_hyper import LoraConfig,get_peft_model
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
            **({"safe_importance": True} if training_args.dash_lora_safe_importance else {}),
            top_k_layers=training_args.top_k_layers,
            ratio=training_args.ratio
        )
        model = get_peft_model(model, peft_config)

    
    if model_args.llm_name == 'llama':
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            pretrain_model_name_or_path,
            padding_side="left",
            use_fast=True,
        )
    elif model_args.llm_name == 'qwen2':
        from transformers import Qwen2Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(
            pretrain_model_name_or_path,
            padding_side="left",
            use_fast=True,
        )
    elif model_args.llm_name == 'qwen2-7b':
        from transformers import Qwen2Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(
            pretrain_model_name_or_path,
            padding_side="left",
            use_fast=True,
        )
        print(f"[CHECK THIS] pad token id: {tokenizer.pad_token_id}")
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
    print('ori_tokenizer_vocab_nums: ',ori_tokenizer_vocab_nums, ' MM_tokenizer_vocab_nums: ',MM_tokenizer_vocab_nums)


    ckpt_dir = infer_args.ckpt_dir

    print(f"DEBUG: Looking for weights in: {os.path.abspath(join(ckpt_dir, 'finetune_weights.bin'))}")
    ckpt_path = join(ckpt_dir,'finetune_weights.bin')
    # ckpt_path = join(ckpt_dir,'exp_norm_strategy_cpu.bin') #NOTE

    ckpt = torch.load(ckpt_path,map_location='cpu')
    model.load_state_dict(ckpt,strict=False)
    print(f'load ckpt from {ckpt_path} finished...')



    # device = infer_args.device
    # torch.cuda.set_device(device)
    # model.cuda() #
    print(f"Process rank {local_rank} is binding to GPU {local_rank}")
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)

    model.eval()
    
    image_processor = model.get_model().visual_encoder.image_processor if training_args.visual_branch else None
    dataset, collator = get_dataset_collator(data_args=data_args, tokenizer=tokenizer, 
                                             image_processor=image_processor,mode=extra_args.mode)
    

    batch_size = 1
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=False,collate_fn=collator,drop_last=False)

    print(model.config)

    # inference_avqa(dataset=dataset,collator=collator,dataloader=dataloader,ckpt_dir=ckpt_dir,model=model,tokenizer=tokenizer,num=infer_args.cut_folds)
    inference_avqa(dataset=dataset,collator=collator,dataloader=dataloader,ckpt_dir=ckpt_dir,model=model,tokenizer=tokenizer,num=local_rank,
        seed=extra_args.seed, mode=extra_args.mode)



if __name__ == "__main__":
    train()

