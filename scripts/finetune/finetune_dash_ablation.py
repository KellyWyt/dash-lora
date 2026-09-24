# -*- coding: utf-8 -*-
import argparse
import json
import os
import pathlib
import sys
from dataclasses import asdict
from os.path import join

sys.path.append(os.getcwd())

import torch
import transformers

from configs.unified_config import DataArguments, ModelArguments, TrainingArguments
from dataset.unified_dataset import get_dataset_collator
from peft_hyper_dash_ablation.tuners.dash_ablation_importance import load_topology, normalize_mode, save_topology
from scripts.finetune.dash_ablation_gradient import compute_gradient_topology
from trainer import UnifiedTrainer
from utils.deepspeed_utils import *
from utils.util import rank0_print, set_seed

local_rank = None


def _parse_ablation_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dash_ablation_mode", type=str, default="energy")
    parser.add_argument("--dash_ablation_seed", type=int, default=123)
    parser.add_argument("--dash_ablation_topology_path", type=str, default=None)
    parser.add_argument("--dash_gradient_calibration_samples", type=int, default=8)
    parser.add_argument("--dash_gradient_calibration_batch_size", type=int, default=1)
    return parser.parse_known_args()


def _distributed_state():
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    return distributed, rank


def _build_lora(model, training_args, ablation_args, topology=None):
    from peft_hyper_dash_ablation import LoraConfig, get_peft_model

    target_modules = "q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj".split(",")
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        inference_mode=False,
        r=training_args.lora_r,
        loramethod=training_args.loramethod,
        reserved_modality=training_args.reserved_modality,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        lora_nums=3,
        blc_alpha=training_args.blc_alpha,
        blc_weight=training_args.blc_weight,
        safe_importance=training_args.dash_lora_safe_importance,
        top_k_layers=training_args.top_k_layers,
        ratio=training_args.ratio,
        dash_ablation_mode=normalize_mode(ablation_args.dash_ablation_mode),
        dash_ablation_seed=ablation_args.dash_ablation_seed,
        dash_ablation_topology_path=ablation_args.dash_ablation_topology_path,
        dash_ablation_precomputed_topology=topology,
    )
    return get_peft_model(model, peft_config)


def train(attn_implementation=None):
    global local_rank
    set_seed(42)

    ablation_args, remaining_argv = _parse_ablation_args()
    ablation_args.dash_ablation_mode = normalize_mode(ablation_args.dash_ablation_mode)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=remaining_argv)
    local_rank = training_args.local_rank

    if ablation_args.dash_ablation_topology_path is None:
        ablation_args.dash_ablation_topology_path = join(training_args.output_dir, "dash_ablation_topology.json")

    output_dir = training_args.output_dir
    saved_config = {
        "model_args": asdict(model_args),
        "data_args": asdict(data_args),
        "training_args": asdict(training_args),
        "dash_ablation_args": vars(ablation_args),
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(join(output_dir, "saved_config.json"), "w") as f:
        f.write(json.dumps(saved_config, indent=4))

    if model_args.llm_name == "llama":
        d_model = 4096
    else:
        raise ValueError("Dash ablation runner is fixed to LLaMA2 for this experiment set.")

    compute_dtype = torch.float32
    if training_args.fp16:
        compute_dtype = torch.float16
    elif training_args.bf16:
        compute_dtype = torch.bfloat16

    pretrain_model_name_or_path = model_args.model_name_or_path
    from models.unified_llama import UnifiedForCausalLM
    from transformers import LlamaConfig, LlamaTokenizer

    config = LlamaConfig.from_pretrained(pretrain_model_name_or_path, local_files_only=True)
    config._attn_implementation = attn_implementation
    model = UnifiedForCausalLM.from_pretrained(
        pretrain_model_name_or_path,
        config=config,
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = LlamaTokenizer.from_pretrained(
        pretrain_model_name_or_path,
        padding_side="left",
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ori_tokenizer_vocab_nums = len(tokenizer)
    model.get_model().pad_token_id = tokenizer.pad_token_id
    model.initialize_MM_tokenizer(tokenizer)
    mm_tokenizer_vocab_nums = len(tokenizer)
    print("ori_tokenizer_vocab_nums: ", ori_tokenizer_vocab_nums, " MM_tokenizer_vocab_nums: ", mm_tokenizer_vocab_nums)

    model.get_model().init_multimodal_modules(
        visual_branch=training_args.visual_branch,
        audio_branch=training_args.audio_branch,
        d_model=d_model,
        vit_ckpt_path=model_args.vit_ckpt_path,
        select_layer_list=model_args.select_layer_list,
        select_feature=model_args.select_feature,
        image_size=model_args.image_size,
        patch_size=model_args.patch_size,
        visual_query_token_nums=model_args.visual_query_token_nums,
        audio_query_token_nums=model_args.audio_query_token_nums,
        BEATs_ckpt_path=model_args.BEATs_ckpt_path,
    )

    audio_ckpt_dir = "/nfs1/outdated/WYT/models/pretrained"
    visual_ckpt_dir = "/nfs1/outdated/WYT/models/pretrained"

    ckpt = torch.load(join(audio_ckpt_dir, "audio_pretrain.bin"), map_location="cpu")
    weight = ckpt.pop("model.embed_tokens.weight")
    model.model.load_state_dict(ckpt, strict=False)
    print(f"pop embed weight, shape: {weight.shape} load ckpt from path: {audio_ckpt_dir}")

    ckpt = torch.load(join(visual_ckpt_dir, "visual_pretrain.bin"), map_location="cpu")
    weight = ckpt.pop("model.embed_tokens.weight")
    model.model.load_state_dict(ckpt, strict=False)
    print(f"pop embed weight, shape: {weight.shape}  load ckpt from path: {visual_ckpt_dir}")

    image_processor = model.get_model().visual_encoder.image_processor if training_args.visual_branch else None
    dataset, collator = get_dataset_collator(data_args=data_args, tokenizer=tokenizer, image_processor=image_processor)

    topology = None
    if training_args.lora_enable and ablation_args.dash_ablation_mode == "gradient":
        distributed, rank = _distributed_state()
        if rank == 0:
            if os.path.exists(ablation_args.dash_ablation_topology_path):
                topology = load_topology(ablation_args.dash_ablation_topology_path)
            else:
                topology = compute_gradient_topology(
                    model=model,
                    dataset=dataset,
                    collator=collator,
                    target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj".split(","),
                    top_k=training_args.top_k_layers,
                    ratio=training_args.ratio,
                    local_rank=max(0, local_rank),
                    calibration_samples=ablation_args.dash_gradient_calibration_samples,
                    calibration_batch_size=ablation_args.dash_gradient_calibration_batch_size,
                    seed=ablation_args.dash_ablation_seed,
                )
                save_topology(ablation_args.dash_ablation_topology_path, topology)
        if distributed:
            payload = [topology]
            broadcast_device = torch.device("cuda", max(0, local_rank)) if torch.cuda.is_available() else None
            torch.distributed.broadcast_object_list(payload, src=0, device=broadcast_device)
            topology = payload[0]

    if training_args.lora_enable:
        print("dash_ablation_mode:", ablation_args.dash_ablation_mode)
        print("dash_ablation_topology_path:", ablation_args.dash_ablation_topology_path)
        model = _build_lora(model, training_args, ablation_args, topology=topology)

    save_modules = training_args.save_modules
    print(f"save_modules: {save_modules}")
    save_modules = save_modules.split(",")
    for name, param in model.named_parameters():
        require_grad = any(target in name for target in save_modules)
        param.requires_grad_(require_grad)

    if local_rank == 0:
        with open(join(output_dir, "model_trainable_params.txt"), "w") as f:
            f.write("\n")
        params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                with open(join(output_dir, "model_trainable_params.txt"), "a") as f:
                    f.write(name + "  " + str(param.shape))
                    f.write("\n")
                params.append(param.numel())
        trainable_params = sum(params) / 1e6
        with open(join(output_dir, "model_trainable_params.txt"), "a") as f:
            f.write(f"trainable_params: {trainable_params:.3f}MB")
        print(f"trainable_params: {trainable_params:.3f}MB")
        with open(join(output_dir, "model.txt"), "w") as f:
            f.write(str(model))

    trainer = UnifiedTrainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=dataset, data_collator=collator)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    final_checkpoint_dir = trainer.save_final_checkpoint()
    rank0_print(local_rank, f"final checkpoint saved at: {final_checkpoint_dir}")
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))


if __name__ == "__main__":
    train()
