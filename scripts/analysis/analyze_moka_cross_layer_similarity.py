#!/usr/bin/env python3
"""Cross-layer similarity analysis for modality-specific MокA LoRA weights.

The checkpoint is split by modality and projection type.  For every group this
script compares all layer pairs using both the LoRA A factor and the effective
update Delta W = B @ A.  Delta-W similarities are evaluated with low-rank
identities, so full hidden_size x hidden_size matrices are never materialized.
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch


LORA_KEY = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\."
    r"(?P<block>[^.]+)\.(?P<projection>[^.]+)\."
    r"lora_(?P<factor>[AB])(?P<modality>\d+)(?:\.[^.]+)?\.weight$"
)
DEFAULT_MODALITIES = {0: "text", 1: "video", 2: "audio"}


def load_state_dict(path):
    state = torch.load(path, map_location="cpu")
    for wrapper in ("state_dict", "model"):
        if isinstance(state, dict) and wrapper in state and isinstance(state[wrapper], dict):
            state = state[wrapper]
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint must contain a state dict, got {type(state).__name__}")
    return state


def parse_weights(state):
    weights = defaultdict(lambda: {"A": {}, "B": {}})
    unmatched = []
    for key, value in state.items():
        if "lora_" not in key:
            continue
        match = LORA_KEY.search(key)
        if not match or not torch.is_tensor(value):
            unmatched.append(key)
            continue
        fields = match.groupdict()
        group = (fields["block"], fields["projection"], int(fields["layer"]))
        weights[group][fields["factor"]][int(fields["modality"])] = value.float().cpu()
    return weights, unmatched


def choose_b(factors, modality):
    """Use modality-matched B, otherwise the MокA shared B0."""
    if modality in factors["B"]:
        return factors["B"][modality], modality
    if 0 in factors["B"]:
        return factors["B"][0], 0
    if len(factors["B"]) == 1:
        b_modality, tensor = next(iter(factors["B"].items()))
        return tensor, b_modality
    return None, None


def cosine_flat(left, right):
    if left.shape != right.shape:
        return math.nan
    x, y = left.reshape(-1).double(), right.reshape(-1).double()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    return float(torch.dot(x, y) / denom) if denom > 0 else math.nan


def delta_cosine(b1, a1, b2, a2):
    """cos(vec(B1 A1), vec(B2 A2)) without constructing either product."""
    if b1.shape[0] != b2.shape[0] or a1.shape[1] != a2.shape[1]:
        return math.nan
    b1, a1, b2, a2 = (x.double() for x in (b1, a1, b2, a2))
    cross = torch.trace((b1.T @ b2) @ (a2 @ a1.T))
    norm1_sq = torch.trace((b1.T @ b1) @ (a1 @ a1.T))
    norm2_sq = torch.trace((b2.T @ b2) @ (a2 @ a2.T))
    denom = torch.sqrt(torch.clamp(norm1_sq * norm2_sq, min=0))
    return float(cross / denom) if denom > 0 else math.nan


def build_records(weights, modality_names):
    grouped = defaultdict(dict)
    b_sources = defaultdict(set)
    for (block, projection, layer), factors in weights.items():
        for modality, a_tensor in factors["A"].items():
            b_tensor, b_modality = choose_b(factors, modality)
            if b_tensor is None:
                continue
            grouped[(modality, block, projection)][layer] = (a_tensor, b_tensor)
            b_sources[(modality, block, projection)].add(b_modality)

    records = []
    for (modality, block, projection), layer_map in sorted(grouped.items()):
        layers = sorted(layer_map)
        for pos, layer_i in enumerate(layers):
            a_i, b_i = layer_map[layer_i]
            for layer_j in layers[pos:]:
                a_j, b_j = layer_map[layer_j]
                records.append({
                    "modality_id": modality,
                    "modality": modality_names.get(modality, f"modality_{modality}"),
                    "block": block,
                    "projection_type": projection,
                    "layer_i": layer_i,
                    "layer_j": layer_j,
                    "layer_distance": layer_j - layer_i,
                    "a_cosine": cosine_flat(a_i, a_j),
                    "delta_w_cosine": delta_cosine(b_i, a_i, b_j, a_j),
                })
    return records, grouped, b_sources


def summarize(records):
    buckets = defaultdict(lambda: {"a": [], "delta": [], "adj_delta": []})
    for row in records:
        if row["layer_i"] == row["layer_j"]:
            continue
        key = (row["modality_id"], row["modality"], row["block"], row["projection_type"])
        if not math.isnan(row["a_cosine"]):
            buckets[key]["a"].append(row["a_cosine"])
        if not math.isnan(row["delta_w_cosine"]):
            buckets[key]["delta"].append(row["delta_w_cosine"])
            if row["layer_distance"] == 1:
                buckets[key]["adj_delta"].append(row["delta_w_cosine"])

    output = []
    for key, values in sorted(buckets.items()):
        modality_id, modality, block, projection = key
        mean = lambda xs: sum(xs) / len(xs) if xs else math.nan
        output.append({
            "modality_id": modality_id,
            "modality": modality,
            "block": block,
            "projection_type": projection,
            "pair_count": len(values["delta"]),
            "mean_a_cosine": mean(values["a"]),
            "mean_delta_w_cosine": mean(values["delta"]),
            "mean_adjacent_delta_w_cosine": mean(values["adj_delta"]),
            "min_delta_w_cosine": min(values["delta"], default=math.nan),
            "max_delta_w_cosine": max(values["delta"], default=math.nan),
        })
    return output


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmaps(records, output_dir):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy unavailable: CSV files were created; skipping heatmaps.")
        return

    buckets = defaultdict(list)
    for row in records:
        buckets[(row["modality"], row["block"], row["projection_type"])].append(row)
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    for (modality, block, projection), rows in sorted(buckets.items()):
        layers = sorted({r["layer_i"] for r in rows} | {r["layer_j"] for r in rows})
        index = {layer: idx for idx, layer in enumerate(layers)}
        matrix = np.full((len(layers), len(layers)), np.nan)
        for row in rows:
            i, j = index[row["layer_i"]], index[row["layer_j"]]
            matrix[i, j] = matrix[j, i] = row["delta_w_cosine"]
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
        step = max(1, len(layers) // 8)
        ticks = list(range(0, len(layers), step))
        ax.set_xticks(ticks, [layers[i] for i in ticks])
        ax.set_yticks(ticks, [layers[i] for i in ticks])
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        ax.set_title(f"{modality} | {block}.{projection} | cos(Delta W)")
        fig.colorbar(image, ax=ax, label="cosine similarity")
        fig.tight_layout()
        fig.savefig(heatmap_dir / f"{modality}__{block}__{projection}.png", dpi=180)
        plt.close(fig)


def parse_modality_names(items):
    names = dict(DEFAULT_MODALITIES)
    for item in items:
        idx, sep, name = item.partition("=")
        if not sep or not idx.isdigit() or not name:
            raise ValueError(f"Invalid --modality-name {item!r}; expected ID=NAME")
        names[int(idx)] = name
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("moka_cross_layer_similarity"))
    parser.add_argument("--modality-name", action="append", default=[], metavar="ID=NAME")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    modality_names = parse_modality_names(args.modality_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights, unmatched = parse_weights(load_state_dict(args.checkpoint))
    if not weights:
        raise RuntimeError("No MокA keys matched the expected layers.*.lora_[AB]<id>.weight format")
    records, grouped, b_sources = build_records(weights, modality_names)
    if not records:
        raise RuntimeError("Parsed LoRA tensors, but found no usable A/B pairs")
    summary = summarize(records)
    write_csv(args.output_dir / "layer_pair_similarity.csv", records)
    write_csv(args.output_dir / "group_summary.csv", summary)

    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "parsed_layer_modules": len(weights),
        "analysis_groups": len(grouped),
        "pair_records_including_diagonal": len(records),
        "unmatched_lora_key_count": len(unmatched),
        "unmatched_lora_keys": unmatched[:50],
        "modality_names": modality_names,
        "b_modality_sources": {
            f"{modality_names.get(k[0], k[0])}/{k[1]}/{k[2]}": sorted(v)
            for k, v in sorted(b_sources.items())
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.no_plots:
        plot_heatmaps(records, args.output_dir)
    print(f"Parsed {len(weights)} layer/modules into {len(grouped)} modality/projection groups")
    print(f"Wrote {len(records)} layer-pair rows to {args.output_dir}")


if __name__ == "__main__":
    main()
