# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import Iterable
import re
import types

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from peft_hyper_dash_ablation.tuners.dash_ablation_importance import DashAblationAnalyzer, PROJECTION_TYPES


def _move_to_device(data, device):
    if isinstance(data, dict):
        return {k: _move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(_move_to_device(v, device) for v in data)
    if isinstance(data, torch.Tensor):
        return data.to(device=device)
    return data


def _target_module_names(model, target_suffixes: Iterable[str]):
    suffixes = tuple(target_suffixes)
    llama_backbone_pattern = re.compile(r"(^|\.)model\.layers\.\d+\.")
    names = []
    for name, module in model.named_modules():
        if (
            isinstance(module, torch.nn.Linear)
            and name.endswith(suffixes)
            and llama_backbone_pattern.search(name) is not None
        ):
            names.append(name)
    return names


def _typewise_gradient_scores(raw_scores):
    grouped = defaultdict(dict)
    for name, value in raw_scores.items():
        for proj_type in PROJECTION_TYPES:
            if name.endswith(proj_type) or f".{proj_type}" in name:
                grouped[proj_type][name] = float(value)
                break

    final_scores = {}
    for values in grouped.values():
        vals = list(values.values())
        lo = min(vals)
        hi = max(vals)
        if hi - lo < 1e-12:
            final_scores.update({name: 0.5 for name in values})
        else:
            for name, value in values.items():
                normed = (value - lo) / (hi - lo)
                final_scores[name] = 1.0 - normed
    return final_scores


def compute_gradient_topology(
    model,
    dataset,
    collator,
    target_modules,
    top_k,
    ratio,
    local_rank=0,
    calibration_samples=8,
    calibration_batch_size=1,
    seed=123,
):
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    model.to(device)
    model.eval()
    model.config.use_cache = False

    module_names = _target_module_names(model, target_modules)
    modules = {name: model.get_submodule(name) for name in module_names}
    scores = {name: 0.0 for name in module_names}
    counts = {name: 0 for name in module_names}
    handles = []
    original_forwards = {}

    # Multimodal LLaMA passes modality_masks as a second positional argument.
    # Calibration runs before Dash-LoRA is attached, so temporarily give the
    # native Linear projections a compatible signature and restore it below.
    def linear_forward_ignoring_modality_mask(module, inputs, *args, **kwargs):
        return F.linear(inputs, module.weight, module.bias)

    for name, module in modules.items():
        original_forwards[name] = module.forward
        module.forward = types.MethodType(linear_forward_ignoring_modality_mask, module)

    def make_forward_hook(module_name):
        def forward_hook(module, inputs, output):
            if not torch.is_tensor(output) or not output.requires_grad:
                return

            detached_output = output.detach()

            def grad_hook(grad):
                value = (detached_output.float() * grad.detach().float()).abs().mean().item()
                scores[module_name] += value
                counts[module_name] += 1

            output.register_hook(grad_hook)

        return forward_hook

    for name, module in modules.items():
        handles.append(module.register_forward_hook(make_forward_hook(name)))

    sample_count = min(int(calibration_samples), len(dataset))
    subset = Subset(dataset, list(range(sample_count)))
    loader = DataLoader(
        subset,
        batch_size=int(calibration_batch_size),
        shuffle=False,
        collate_fn=collator,
        drop_last=False,
        num_workers=0,
    )

    try:
        for step, batch in enumerate(loader):
            batch.pop("batch_metadata", None)
            batch = _move_to_device(batch, device)
            model.zero_grad(set_to_none=True)
            output = model(**batch)
            loss = output.loss if hasattr(output, "loss") else output[0]
            if loss is None:
                raise RuntimeError("Gradient topology calibration did not receive a training loss.")
            loss.backward()
            model.zero_grad(set_to_none=True)
            if step + 1 >= len(loader):
                break
    finally:
        for handle in handles:
            handle.remove()
        for name, module in modules.items():
            module.forward = original_forwards[name]

    averaged = {name: scores[name] / max(1, counts[name]) for name in module_names}
    final_scores = _typewise_gradient_scores(averaged)
    analyzer = DashAblationAnalyzer(mode="energy", seed=seed)
    topology = analyzer._select_from_scores(final_scores, top_k, ratio)
    topology["raw_gradient_scores"] = averaged
    topology["stats"]["mode"] = "gradient"
    topology["stats"]["calibration_samples"] = sample_count
    topology["stats"]["calibration_batch_size"] = int(calibration_batch_size)
    topology["stats"]["score_type"] = "mean_abs_activation_times_activation_grad"
    return topology
