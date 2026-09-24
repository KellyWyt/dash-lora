#!/usr/bin/env python3
"""Plot frozen-weight energy against the learned MokA update magnitude."""
import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from safetensors import safe_open


KEY_RE = re.compile(
    r"layers\.(?P<layer>\d+)\.(?P<block>self_attn|mlp)\."
    r"(?P<proj>[a-z]+_proj)\.lora_A(?P<modality>[012])\.weight$"
)
COLORS = {
    "q_proj": "#3B82F6", "k_proj": "#60A5FA", "v_proj": "#2563EB",
    "o_proj": "#1D4ED8", "gate_proj": "#F59E0B", "up_proj": "#F97316",
    "down_proj": "#DC2626",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("paper/figs"))
    p.add_argument("--alpha", type=float, default=16.0)
    return p.parse_args()


def base_norm_loader(model_dir, chunk_rows=128):
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    handles = {}

    def norm(name):
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = safe_open(str(model_dir / shard), framework="pt", device="cpu")
        tensor = handles[shard].get_tensor(name)
        shape = tensor.shape
        squared_sum = 0.0
        for start in range(0, shape[0], chunk_rows):
            chunk = tensor[start:min(start + chunk_rows, shape[0])].float()
            squared_sum += float(torch.sum(chunk * chunk, dtype=torch.float64))
        return math.sqrt(squared_sum), math.prod(shape)
    return norm


def correlations(x, y):
    p = pearsonr(x, y)
    s = spearmanr(x, y)
    return {"pearson_r": float(p.statistic), "pearson_p": float(p.pvalue),
            "spearman_rho": float(s.statistic), "spearman_p": float(s.pvalue)}


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter = torch.load(args.adapter, map_location="cpu")
    get_base_norm = base_norm_loader(args.base_model)
    rows = []

    for key, a0 in adapter.items():
        match = KEY_RE.search(key)
        if not match or match.group("modality") != "0":
            continue
        prefix = key[:-len("lora_A0.weight")]
        matrices_a = [adapter[prefix + f"lora_A{i}.weight"].float() for i in range(3)]
        matrix_b = adapter[prefix + "lora_B0.weight"].float()
        rank = matrices_a[0].shape[0]
        scale = args.alpha / rank
        update_norms = [float(torch.linalg.vector_norm(matrix_b @ a) * scale) for a in matrices_a]
        aggregate = math.sqrt(sum(v * v for v in update_norms))

        layer = int(match.group("layer"))
        block, proj = match.group("block"), match.group("proj")
        base_key = f"model.layers.{layer}.{block}.{proj}.weight"
        weight_norm, weight_numel = get_base_norm(base_key)
        rows.append({
            "layer": layer, "block": block, "projection": proj,
            "weight_frobenius": weight_norm,
            "intrinsic_weight_energy": weight_norm ** 2,
            "weight_rms": weight_norm / math.sqrt(weight_numel),
            "update_magnitude": aggregate,
            "update_text": update_norms[0], "update_visual": update_norms[1],
            "update_audio": update_norms[2], "rank": rank, "scale": scale,
        })

    rows.sort(key=lambda r: (r["layer"], r["projection"]))
    if len(rows) != 224:
        raise RuntimeError(f"Expected 224 LLaMA2 modules, found {len(rows)}")

    csv_path = args.output_dir / "llama2_moka_avqa_weight_update.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    energy = np.array([r["intrinsic_weight_energy"] for r in rows])
    magnitude = np.array([r["update_magnitude"] for r in rows])
    log_energy, log_magnitude = np.log10(energy), np.log10(magnitude)
    stats = {
        "definition": {
            "x": "||W||_F^2", "y": "sqrt(sum_m ||(alpha/r) B A_m||_F^2)",
            "alpha": args.alpha, "modules": len(rows), "dataset": "MUSIC-AVQA",
        },
        "raw": correlations(energy, magnitude),
        "log10": correlations(log_energy, log_magnitude),
        "spearman_invariant_note": "Spearman is identical before/after log transform.",
        "by_projection": {},
    }
    for proj in COLORS:
        idx = np.array([r["projection"] == proj for r in rows])
        stats["by_projection"][proj] = correlations(log_energy[idx], log_magnitude[idx])

    stats_path = args.output_dir / "llama2_moka_avqa_weight_update_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.35, 2.65))
    for proj, color in COLORS.items():
        idx = np.array([r["projection"] == proj for r in rows])
        ax.scatter(energy[idx], magnitude[idx], s=14, alpha=.76, color=color,
                   edgecolor="white", linewidth=.25, label=proj.replace("_proj", ""))
    coef = np.polyfit(log_energy, log_magnitude, 1)
    grid = np.linspace(log_energy.min(), log_energy.max(), 200)
    ax.plot(10 ** grid, 10 ** np.polyval(coef, grid), color="#111827", lw=1.25,
            linestyle="--", label="log--log fit")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Intrinsic Weight Energy $\|W\|_F^2$")
    ax.set_ylabel(r"LoRA Update Magnitude")
    log_stats = stats["log10"]
    ax.text(.035, .965, f"Pearson $r$ = {log_stats['pearson_r']:.3f}\n"
            f"Spearman $\\rho$ = {log_stats['spearman_rho']:.3f}\n$n$ = {len(rows)}",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=.28", facecolor="white", alpha=.9,
                      edgecolor="#D1D5DB"))
    ax.grid(True, which="major", alpha=.18, linewidth=.5)
    ax.legend(ncol=4, fontsize=6.4, loc="lower center", bbox_to_anchor=(.5, 1.01),
              frameon=False, handletextpad=.25, columnspacing=.65)
    fig.tight_layout(pad=.5)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output_dir / f"llama2_moka_avqa_weight_update.{suffix}",
                    dpi=400, bbox_inches="tight")
    print(json.dumps(stats, indent=2))
    print(f"Wrote {csv_path}, {stats_path}, and PDF/PNG figures")


if __name__ == "__main__":
    main()
