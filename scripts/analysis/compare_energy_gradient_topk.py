#!/usr/bin/env python3
"""Compare DASH-LoRA's energy ranking with task-gradient sensitivity.

This is a calibration-only analysis: it does not fine-tune or update the model.
Each target projection receives a scalar gate.  The absolute loss gradient of
that gate is the first-order Taylor sensitivity of the module output.
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import LlamaConfig, LlamaTokenizer

sys.path.append(str(Path(__file__).resolve().parents[2]))

from configs.unified_config import DataArguments
from dataset.unified_dataset import get_dataset_collator
from models.unified_llama import UnifiedForCausalLM
from utils.util import prepare_sample, set_seed


TARGET_RE = re.compile(
    r"^model\.layers\.(\d+)\.(self_attn|mlp)\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|down_proj|up_proj)$"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path,
                        default=Path("/nfs1/outdated/WYT/models/Llama-2-7b-chat-hf"))
    parser.add_argument("--energy-csv", type=Path,
                        default=Path("paper/figs/llama2_moka_avqa_weight_update.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("paper/figs/gradient_topk"))
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text-k", type=int, default=66)
    parser.add_argument("--other-k", type=int, default=22)
    return parser.parse_args()


def load_multimodal_model(args):
    config = LlamaConfig.from_pretrained(args.base_model, local_files_only=True)
    config.use_cache = False
    model = UnifiedForCausalLM.from_pretrained(
        args.base_model, config=config, torch_dtype=torch.bfloat16
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        args.base_model, padding_side="left", use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.get_model().pad_token_id = tokenizer.pad_token_id
    model.initialize_MM_tokenizer(tokenizer)
    model.get_model().init_multimodal_modules(
        visual_branch=True,
        audio_branch=True,
        d_model=4096,
        vit_ckpt_path="/nfs1/outdated/WYT/models/clip-vit-large-patch14",
        select_layer_list=[14, 23],
        select_feature="patch",
        image_size=224,
        patch_size=14,
        visual_query_token_nums=32,
        audio_query_token_nums=32,
        BEATs_ckpt_path="/nfs1/outdated/WYT/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
    )

    pretrained_dir = Path("/nfs1/outdated/WYT/models/pretrained")
    for filename in ("audio_pretrain.bin", "visual_pretrain.bin"):
        state = torch.load(pretrained_dir / filename, map_location="cpu")
        state.pop("model.embed_tokens.weight", None)
        model.model.load_state_dict(state, strict=False)

    model.requires_grad_(False)
    model.eval()
    model.to(args.device)
    return model, tokenizer


def attach_scalar_gates(model, device):
    gates = {}
    handles = []
    for name, module in model.named_modules():
        if not TARGET_RE.match(name) or not isinstance(module, nn.Linear):
            continue
        gate = nn.Parameter(torch.ones((), device=device, dtype=torch.float32))
        gates[name] = gate

        def hook(_module, _inputs, output, gate=gate):
            # Keep the activation dtype unchanged while retaining a scalar
            # gradient path. The gate itself remains FP32 for stable scoring.
            return output * gate.to(dtype=output.dtype)

        handles.append(module.register_forward_hook(hook))
    if len(gates) != 224:
        raise RuntimeError(f"Expected 224 LLaMA target modules, found {len(gates)}")
    return gates, handles


def typewise_minmax(scores, projections):
    calibrated = {}
    for projection in sorted(set(projections.values())):
        names = [name for name, proj in projections.items() if proj == projection]
        values = [scores[name] for name in names]
        lo, hi = min(values), max(values)
        for name in names:
            calibrated[name] = 0.5 if hi - lo < 1e-12 else (scores[name] - lo) / (hi - lo)
    return calibrated


def load_energy_scores(path):
    scores = {}
    projections = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            name = f"model.layers.{row['layer']}.{row['block']}.{row['projection']}"
            scores[name] = float(row["weight_frobenius"])
            projections[name] = row["projection"]
    if len(scores) != 224:
        raise RuntimeError(f"Expected 224 energy rows, found {len(scores)}")
    return scores, projections


def topk_overlap(left, right, k):
    left_set, right_set = set(left[:k]), set(right[:k])
    count = len(left_set & right_set)
    return {"k": k, "count": count, "rate": count / k}


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_multimodal_model(args)
    gates, handles = attach_scalar_gates(model, args.device)
    energy_raw, projections = load_energy_scores(args.energy_csv)

    data_args = DataArguments(video_frame_nums=10, avqa_task=True, ave_task=False)
    image_processor = model.get_model().visual_encoder.image_processor
    dataset, collator = get_dataset_collator(
        data_args=data_args, tokenizer=tokenizer,
        image_processor=image_processor, mode="train"
    )
    count = min(args.num_samples, len(dataset))
    indices = random.Random(args.seed).sample(range(len(dataset)), count)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=1, shuffle=False,
        collate_fn=collator, num_workers=2
    )

    accumulated = defaultdict(float)
    successful = 0
    for batch in tqdm(loader, desc="gradient calibration"):
        batch.pop("batch_metadata", None)
        batch = prepare_sample(batch, device=args.device)
        for gate in gates.values():
            gate.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**batch)
            loss = output.loss
        loss.backward()
        if not torch.isfinite(loss):
            continue
        successful += 1
        for name, gate in gates.items():
            if gate.grad is not None:
                accumulated[name] += abs(float(gate.grad.detach()))

    for handle in handles:
        handle.remove()
    if successful == 0:
        raise RuntimeError("No successful calibration batches")

    gradient_raw = {name: accumulated[name] / successful for name in gates}
    energy = typewise_minmax(energy_raw, projections)
    gradient = typewise_minmax(gradient_raw, projections)
    energy_rank = sorted(energy, key=energy.get, reverse=True)
    gradient_rank = sorted(gradient, key=gradient.get, reverse=True)

    results = {
        "definition": {
            "gradient": "mean_batch |dL/d scalar_module_gate|",
            "calibration_dataset": "MUSIC-AVQA train",
            "samples": successful,
            "seed": args.seed,
            "normalization": "projection-family-wise min-max",
        },
        "overlap": {
            "text": topk_overlap(energy_rank, gradient_rank, args.text_k),
            "visual": topk_overlap(energy_rank, gradient_rank, args.other_k),
            "audio": topk_overlap(energy_rank, gradient_rank, args.other_k),
        },
        "random_expected_rate": {
            "text": args.text_k / 224,
            "visual": args.other_k / 224,
            "audio": args.other_k / 224,
        },
    }

    rows = []
    energy_position = {name: i + 1 for i, name in enumerate(energy_rank)}
    gradient_position = {name: i + 1 for i, name in enumerate(gradient_rank)}
    for name in sorted(gates):
        match = TARGET_RE.match(name)
        rows.append({
            "module": name,
            "layer": int(match.group(1)),
            "projection": match.group(3),
            "energy_raw": energy_raw[name],
            "energy_calibrated": energy[name],
            "energy_rank": energy_position[name],
            "gradient_raw": gradient_raw[name],
            "gradient_calibrated": gradient[name],
            "gradient_rank": gradient_position[name],
        })

    csv_path = args.output_dir / "llama_avqa_energy_gradient_ranking.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output_dir / "llama_avqa_energy_gradient_topk.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
