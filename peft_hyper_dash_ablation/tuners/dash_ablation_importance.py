# -*- coding: utf-8 -*-

import json
import os
import random
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.multiprocessing as mp


PROJECTION_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]


class DashAblationAnalyzer:
    """Unified topology selector for Dash-LoRA ablations.

    Scores are always sorted ascending: lower score means the module is selected
    earlier for exclusive LoRA capacity.
    """

    def __init__(self, mode: str = "energy", seed: int = 123):
        self.mode = normalize_mode(mode)
        self.seed = seed
        self.available_gpus = self._detect_available_gpus()

    def _detect_available_gpus(self):
        if not torch.cuda.is_available():
            return []
        available_gpus = []
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                x = torch.randn(10, 10, device=f"cuda:{i}")
                del x
                available_gpus.append(i)
            except Exception:
                pass
        return available_gpus

    def analyze_model_only_target_modules(self, model, target_modules, target_module_keys, top_k=65, ratio=3):
        num_gpus = len(self.available_gpus)
        if self.mode == "random_topology":
            scores = {name: float(i) for i, name in enumerate(self._random_order(target_module_keys))}
            return self._select_from_scores(scores, top_k, ratio)
        if num_gpus == 0:
            return self._analyze_cpu(model, target_module_keys, top_k, ratio)
        if len(target_module_keys) < 20 or num_gpus == 1:
            return self.analyze_single_gpu_safe(model, target_module_keys, top_k, ratio)
        return self._analyze_multi_gpu(model, target_modules, target_module_keys, top_k, ratio)

    def analyze_single_gpu_safe(self, model, target_module_keys, top_k, ratio):
        if self.mode == "random_topology":
            scores = {name: float(i) for i, name in enumerate(self._random_order(target_module_keys))}
            return self._select_from_scores(scores, top_k, ratio)
        if not self.available_gpus:
            raise RuntimeError("Dash-LoRA safe importance requires one CUDA device")
        device = torch.device(f"cuda:{self.available_gpus[0]}")
        raw = {}
        for name in target_module_keys:
            try:
                module = model.get_submodule(name)
                w_gpu = module.weight.data.to(device=device, dtype=torch.float32)
                raw[name] = {"norm": torch.norm(w_gpu).item()}
            except Exception as exc:
                raise RuntimeError(f"Importance analysis failed for {name}") from exc
        if len(raw) != len(target_module_keys):
            raise RuntimeError(f"Importance analysis collected {len(raw)}/{len(target_module_keys)} modules")
        scores = self._compute_scores(raw)
        return self._select_from_scores(scores, top_k, ratio)

    def _analyze_multi_gpu(self, model, projection_types, target_module_keys, top_k, ratio):
        valid_modules = []
        for name in target_module_keys:
            try:
                module = model.get_submodule(name)
                if hasattr(module, "weight"):
                    valid_modules.append((name, module.weight.data))
            except Exception:
                continue
        chunks = self._split_into_chunks(valid_modules, len(self.available_gpus))
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(self.available_gpus)) as pool:
            results = []
            for i, chunk in enumerate(chunks):
                if chunk:
                    results.append(pool.apply_async(self._analyze_chunk_gpu_raw, (chunk, self.available_gpus[i])))
            raw = {}
            for res in results:
                try:
                    raw.update(res.get(timeout=300))
                except Exception:
                    pass
        scores = self._compute_scores(raw)
        return self._select_from_scores(scores, top_k, ratio)

    def _analyze_cpu(self, model, target_module_keys, top_k, ratio):
        raw = {}
        for name in target_module_keys:
            try:
                w = model.get_submodule(name).weight.data
                raw[name] = {"norm": torch.norm(w.float()).item()}
            except Exception:
                continue
        scores = self._compute_scores(raw)
        return self._select_from_scores(scores, top_k, ratio)

    def _analyze_chunk_gpu_raw(self, chunk, gpu_id):
        device = torch.device(f"cuda:{gpu_id}")
        res = {}
        for name, weight in chunk:
            try:
                w = weight.to(device=device, dtype=torch.float32)
                res[name] = {"norm": torch.norm(w).item()}
            except Exception:
                res[name] = {"norm": 0.0}
        return res

    def _compute_scores(self, raw_metrics_dict):
        norms = {name: float(m["norm"]) for name, m in raw_metrics_dict.items()}
        if self.mode == "global_energy":
            normed = _minmax_by_name(norms)
            return {name: 1.0 - value for name, value in normed.items()}
        if self.mode == "reverse_energy":
            return self._typewise_norm_scores(norms, reverse=True)
        return self._typewise_norm_scores(norms, reverse=False)

    def _typewise_norm_scores(self, norms: Dict[str, float], reverse: bool):
        grouped = {}
        for name, value in norms.items():
            proj_type = projection_type_for(name)
            if proj_type is None:
                continue
            grouped.setdefault(proj_type, {})[name] = value
        scores = {}
        for values in grouped.values():
            normed = _minmax_by_name(values)
            if reverse:
                scores.update(normed)
            else:
                scores.update({name: 1.0 - value for name, value in normed.items()})
        return scores

    def _select_from_scores(self, all_scores, total_budget, ratio):
        sorted_items = sorted(all_scores.items(), key=lambda x: (x[1], x[0]))
        all_modules_sorted = [x[0] for x in sorted_items]
        max_layers = len(all_modules_sorted)
        actual_budget = total_budget * 2
        limit_others = int(actual_budget / (ratio + 2))
        limit_text = int(limit_others * ratio)
        limit_text = min(max(1, limit_text), max_layers)
        limit_others = min(max(1, limit_others), limit_text)
        text_exclusive = all_modules_sorted[:limit_text]
        others_exclusive = all_modules_sorted[:limit_others]
        stats = {
            "mode": self.mode,
            "seed": self.seed,
            "ratio": ratio,
            "text_limit": limit_text,
            "others_limit": limit_others,
            "total_slot_budget": actual_budget,
        }
        print("\n" + "=" * 60)
        print(f"DASH ABLATION TOPOLOGY: {self.mode}")
        print("=" * 60)
        print(f"Text exclusive:   {limit_text}")
        print(f"Others exclusive: {limit_others}")
        print(f"Actual ratio: {limit_text / max(1, limit_others):.2f}:1:1")
        return {
            "text_exclusive": text_exclusive,
            "others_exclusive": others_exclusive,
            "scores": all_scores,
            "ranking": all_modules_sorted,
            "stats": stats,
        }

    def _random_order(self, module_names: Iterable[str]) -> List[str]:
        modules = sorted(module_names)
        rng = random.Random(self.seed)
        rng.shuffle(modules)
        return modules

    def _split_into_chunks(self, data, n):
        if n == 0:
            return [data]
        s = len(data) // n + 1
        return [data[i : i + s] for i in range(0, len(data), s)]


def normalize_mode(mode: str) -> str:
    aliases = {
        "full": "energy",
        "scored": "energy",
        "typewise_energy": "energy",
        "energy": "energy",
        "random": "random_topology",
        "random_order": "random_topology",
        "random_topology": "random_topology",
        "global": "global_energy",
        "global_energy": "global_energy",
        "reverse": "reverse_energy",
        "reverse_energy": "reverse_energy",
        "gradient": "gradient",
        "gradient_relation": "gradient",
    }
    key = str(mode).strip().lower().replace("-", "_")
    if key not in aliases:
        raise ValueError(f"Unknown dash ablation mode: {mode}")
    return aliases[key]


def projection_type_for(name: str):
    for proj_type in PROJECTION_TYPES:
        if name.endswith(proj_type) or f".{proj_type}" in name:
            return proj_type
    return None


def _minmax_by_name(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.array(list(values.values()), dtype=np.float64)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        return {name: 0.5 for name in values}
    return {name: (float(value) - lo) / (hi - lo) for name, value in values.items()}


def load_topology(path: str):
    with open(path, "r") as f:
        return json.load(f)


def save_topology(path: str, topology: dict):
    if not path:
        return
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w") as f:
        json.dump(topology, f, indent=2)
