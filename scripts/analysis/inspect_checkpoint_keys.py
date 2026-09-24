#!/usr/bin/env python3
import argparse
from collections import Counter

import torch
from torch._subclasses.fake_tensor import FakeTensorMode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    with FakeTensorMode():
        state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    print(f"entries={len(state)}")
    counts = Counter()
    for key in state:
        if "lora_A" in key:
            counts["lora_A"] += 1
        elif "lora_B" in key:
            counts["lora_B"] += 1
        elif key.endswith(".weight"):
            counts["other_weight"] += 1
        else:
            counts["other"] += 1
    print(dict(counts))
    for key in list(state)[: args.limit]:
        value = state[key]
        shape = tuple(value.shape) if hasattr(value, "shape") else None
        print(f"{key}\t{shape}")


if __name__ == "__main__":
    main()
