#!/usr/bin/env python3
"""Keep selected CUDA GPUs busy with configurable matrix multiplications."""

import argparse
import multiprocessing as mp
import os
import signal
import time
from typing import Sequence


def parse_duration(value: str) -> float:
    """Parse durations such as 30s, 15m, 7h, or 1.5d into seconds."""
    normalized = value.strip().lower()
    if normalized in {"0", "none", "unlimited"}:
        return 0.0
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if len(normalized) < 2 or normalized[-1] not in units:
        raise argparse.ArgumentTypeError(
            "duration must end in s, m, h, or d, for example: 30s, 15m, 7h"
        )
    try:
        amount = float(normalized[:-1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid duration: {value}") from exc
    if amount <= 0:
        raise argparse.ArgumentTypeError("duration must be positive")
    return amount * units[normalized[-1]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controllable matrix-multiplication workload on multiple GPUs. "
            "Stop cleanly with Ctrl+C or SIGTERM."
        )
    )
    parser.add_argument(
        "--gpus",
        default="0,1,2,3",
        help="Comma-separated physical GPU IDs (default: 0,1,2,3).",
    )
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=8192,
        help="Square matrix size used by each worker (default: 8192).",
    )
    parser.add_argument(
        "--memory-mib",
        type=int,
        default=23000,
        help="Minimum PyTorch memory allocated per GPU in MiB (default: 23000).",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Computation dtype (default: float16).",
    )
    parser.add_argument(
        "--busy-percent",
        type=float,
        default=100.0,
        help="Approximate compute duty cycle from 1 to 100 (default: 100).",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=1.0,
        help="Duty-cycle control window in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--status-seconds",
        type=float,
        default=30.0,
        help="Worker status print interval; 0 disables it (default: 30).",
    )
    parser.add_argument(
        "--duration",
        type=parse_duration,
        default=0.0,
        metavar="TIME",
        help="Stop after TIME (for example 30m, 7h, or 1d); 0 is unlimited.",
    )
    args = parser.parse_args()

    if args.matrix_size < 512:
        parser.error("--matrix-size must be at least 512")
    if args.memory_mib < 1024:
        parser.error("--memory-mib must be at least 1024")
    if not 1.0 <= args.busy_percent <= 100.0:
        parser.error("--busy-percent must be between 1 and 100")
    if args.window_seconds <= 0:
        parser.error("--window-seconds must be positive")
    if args.status_seconds < 0:
        parser.error("--status-seconds cannot be negative")
    return args


def gpu_worker(
    local_gpu: int,
    physical_gpu: str,
    matrix_size: int,
    memory_mib: int,
    dtype_name: str,
    busy_percent: float,
    window_seconds: float,
    status_seconds: float,
    stop_event,
) -> None:
    import torch

    torch.cuda.set_device(local_gpu)
    device = torch.device(f"cuda:{local_gpu}")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]

    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"GPU {physical_gpu} does not support bfloat16")

    properties = torch.cuda.get_device_properties(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(2026 + local_gpu)

    try:
        left = torch.randn(
            (matrix_size, matrix_size),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        right = torch.randn(
            (matrix_size, matrix_size),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        output = torch.empty_like(left)
        allocated_bytes = torch.cuda.memory_allocated(device)
        target_bytes = memory_mib * 2**20
        reserve_bytes = max(0, target_bytes - allocated_bytes)
        memory_reserve = torch.empty(
            reserve_bytes, device=device, dtype=torch.uint8
        )
        memory_reserve.fill_(1)
        torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError as exc:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        raise RuntimeError(
            f"GPU {physical_gpu} cannot allocate matrix-size={matrix_size}; "
            f"free={free_bytes / 2**30:.1f} GiB, total={total_bytes / 2**30:.1f} GiB. "
            "Retry with a smaller --memory-mib or --matrix-size."
        ) from exc

    torch.mm(left, right, out=output)
    torch.cuda.synchronize(device)
    print(
        f"[GPU {physical_gpu}] started on {properties.name}; "
        f"matrix={matrix_size}x{matrix_size}, dtype={dtype_name}, "
        f"busy={busy_percent:.0f}%, "
        f"allocated={torch.cuda.memory_allocated(device) / 2**20:.0f} MiB",
        flush=True,
    )

    busy_seconds = window_seconds * busy_percent / 100.0
    iterations = 0
    last_status = time.monotonic()

    while not stop_event.is_set():
        window_start = time.monotonic()
        busy_deadline = window_start + busy_seconds

        while time.monotonic() < busy_deadline and not stop_event.is_set():
            torch.mm(left, right, out=output)
            iterations += 1

        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - window_start
        remaining = window_seconds - elapsed
        if remaining > 0:
            stop_event.wait(remaining)

        now = time.monotonic()
        if status_seconds and now - last_status >= status_seconds:
            allocated = torch.cuda.memory_allocated(device) / 2**30
            print(
                f"[GPU {physical_gpu}] alive; iterations={iterations}, "
                f"allocated={allocated:.2f} GiB",
                flush=True,
            )
            last_status = now


def parse_gpu_ids(value: str) -> Sequence[str]:
    gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("No GPU IDs were supplied")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"Duplicate GPU IDs: {value}")
    if any(not item.isdigit() for item in gpu_ids):
        raise ValueError(f"GPU IDs must be non-negative integers: {value}")
    return gpu_ids


def main() -> None:
    args = parse_args()
    try:
        gpu_ids = parse_gpu_ids(args.gpus)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # Each child sees only the selected physical GPUs, renumbered from zero.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in the active Python environment")
    if torch.cuda.device_count() != len(gpu_ids):
        raise SystemExit(
            f"Requested {len(gpu_ids)} GPUs ({','.join(gpu_ids)}), but PyTorch "
            f"can see only {torch.cuda.device_count()} after applying CUDA_VISIBLE_DEVICES"
        )

    context = mp.get_context("spawn")
    stop_event = context.Event()

    def request_stop(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    workers = []
    for local_gpu, physical_gpu in enumerate(gpu_ids):
        process = context.Process(
            target=gpu_worker,
            args=(
                local_gpu,
                physical_gpu,
                args.matrix_size,
                args.memory_mib,
                args.dtype,
                args.busy_percent,
                args.window_seconds,
                args.status_seconds,
                stop_event,
            ),
            name=f"gpu-worker-{physical_gpu}",
        )
        process.start()
        workers.append(process)

    print(
        f"Started {len(workers)} GPU workers on physical GPUs "
        f"{','.join(gpu_ids)}. "
        + (
            f"Automatic stop in {args.duration:.0f} seconds."
            if args.duration
            else "No time limit."
        )
        + " Press Ctrl+C to stop.",
        flush=True,
    )

    deadline = time.monotonic() + args.duration if args.duration else None
    try:
        while any(process.is_alive() for process in workers):
            for process in workers:
                process.join(timeout=0.2)
            failed = [
                process
                for process in workers
                if process.exitcode not in (None, 0)
            ]
            if failed:
                stop_event.set()
                names = ", ".join(
                    f"{process.name}(exit={process.exitcode})"
                    for process in failed
                )
                raise SystemExit(f"GPU worker failed: {names}")
            if deadline is not None and time.monotonic() >= deadline:
                print("Configured duration reached; stopping workers.", flush=True)
                stop_event.set()
                break
    finally:
        stop_event.set()
        for process in workers:
            process.join(timeout=10)
        for process in workers:
            if process.is_alive():
                process.terminate()
                process.join()
        print("All GPU workers stopped.", flush=True)


if __name__ == "__main__":
    main()
