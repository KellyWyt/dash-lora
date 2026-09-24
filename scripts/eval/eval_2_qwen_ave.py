# -*- coding: utf-8 -*-

import jsonlines
import re
import os
import json
import argparse

# 答案列表（用于参照，宽容模式下主要用于验证）
answer_list = ['zero', 'pipa', 'middle', 'congas', 'eight', 'saxophone', 'tuba', 'no', 'guzheng', 
               'left', 'ten', 'four', 'five', 'nine', 'more than ten', 'drum', 'suona', 'indoor', 
               'two', 'simultaneously', 'piano', 'right', 'acoustic_guitar', 'trumpet', 'seven', 
               'outdoor', 'six', 'yes', 'violin', 'flute', 'clarinet', 'bagpipe', 'one', 'three', 
               'accordion', 'cello', 'electric_bass', 'erhu', 'ukulele', 'bassoon', 'banjo', 
               'xylophone']

def safe_div(numerator, denominator):
    return 100 * numerator / denominator if denominator != 0 else 0.0

def normalize_event(text):
    text = (text or '').strip().lower().replace('_', ' ')
    return re.sub(r'\s+', ' ', text).strip(' .,:;')


def check_lenient(path):
    """Evaluate event classification and AVE temporal localization metrics."""
    stats = {}
    total = 0
    strict_correct_total = 0
    lenient_correct_total = 0
    range_exact_total = 0
    joint_exact_total = 0
    segment_correct_total = 0
    segment_total = 0
    event_parse_failures = 0
    range_parse_failures = 0

    print(f"Checking: {path}")

    with jsonlines.open(path, 'r') as f:
        for sample in f:
            total += 1
            gt_output = sample.get('output', '').strip().lower()
            gt_match = re.search(
                r'event:\s*(.*?)\s*start_time:\s*([+-]?\d+(?:\.\d+)?)\s*'
                r'end_time:\s*([+-]?\d+(?:\.\d+)?)',
                gt_output,
                flags=re.I | re.S,
            )
            if not gt_match:
                raise ValueError(f"Cannot parse ground truth at row {total}: {gt_output!r}")
            gt_event = normalize_event(gt_match.group(1))
            gt_start = float(gt_match.group(2))
            gt_end = float(gt_match.group(3))

            pred_raw = sample.get('predict', '')
            pred_text = pred_raw.lower()
            for token in ('<s>', '</s>', '<|endoftext|>', '<unk>'):
                pred_text = pred_text.replace(token, ' ')
            pred_first_line = pred_text.split('\n')[0].strip()

            event_tag = re.search(
                r'<event>\s*(.*?)\s*</event>', pred_first_line, flags=re.I | re.S
            )
            event_colon = re.search(
                r'event:\s*(.*?)\s*start_time', pred_first_line, flags=re.I | re.S
            )
            if event_tag:
                pred_event = normalize_event(event_tag.group(1))
            elif event_colon:
                pred_event = normalize_event(event_colon.group(1))
            else:
                pred_event = ''
                event_parse_failures += 1

            range_match = re.search(
                r'<range>\s*([+-]?\d+(?:\.\d+)?)\s*,\s*'
                r'([+-]?\d+(?:\.\d+)?)\s*</range>',
                pred_first_line,
                flags=re.I | re.S,
            )
            if range_match:
                pred_start = float(range_match.group(1))
                pred_end = float(range_match.group(2))
            else:
                pred_start = pred_end = None
                range_parse_failures += 1

            strict_hit = int(pred_event == gt_event)
            lenient_hit = int(gt_event in pred_first_line)
            range_exact_hit = int(
                pred_start is not None
                and pred_end is not None
                and pred_start == gt_start
                and pred_end == gt_end
            )
            joint_exact_hit = int(strict_hit and range_exact_hit)

            strict_correct_total += strict_hit
            lenient_correct_total += lenient_hit
            range_exact_total += range_exact_hit
            joint_exact_total += joint_exact_hit
            stats.setdefault(gt_event, []).append((strict_hit, lenient_hit))

            # AVE segment-level localization: ten 1-second segments per clip.
            if pred_start is not None and pred_end is not None and pred_event:
                for second in range(10):
                    gt_label = gt_event if gt_start <= second < gt_end else '__background__'
                    pred_label = pred_event if pred_start <= second < pred_end else '__background__'
                    segment_correct_total += int(gt_label == pred_label)
                    segment_total += 1

    print(f"{'Event Category':<32} | {'Strict Acc':<12} | {'Lenient Acc':<12} | {'Count':<8}")
    print('-' * 82)
    for event, values in stats.items():
        count = len(values)
        strict = sum(x[0] for x in values)
        lenient = sum(x[1] for x in values)
        print(
            f"{event:<32} | {safe_div(strict, count):>6.2f} %     | "
            f"{safe_div(lenient, count):>6.2f} %     | {count:<8}"
        )
    print('-' * 82)
    print(
        f"{'Overall event accuracy':<32} | "
        f"{safe_div(strict_correct_total, total):>6.2f} %     | "
        f"{safe_div(lenient_correct_total, total):>6.2f} %     | {total:<8}"
    )

    print("\nAVE metrics")
    print('-' * 72)
    metrics = (
        ('Event strict accuracy', strict_correct_total, total),
        ('Event lenient hit accuracy', lenient_correct_total, total),
        ('Exact time-range accuracy', range_exact_total, total),
        ('Exact event + range accuracy', joint_exact_total, total),
        ('AVE segment localization accuracy', segment_correct_total, segment_total),
    )
    for name, correct, count in metrics:
        print(f"{name:<36}: {safe_div(correct, count):>6.2f} % ({correct}/{count})")
    print(f"Parse failures: event={event_parse_failures}, range={range_parse_failures}")
    return total

def main():
    parser = argparse.ArgumentParser(description="Evaluate AVE inference JSONL files.")
    # parser.add_argument("--results", nargs="+", required=True)
    args = parser.parse_args()
    # 替换你的路径
    # results=['/nfs1/outdated/WYT/MokA-copy/results/finetune-qwen2/qwen2-7B_music-moka-ave-r-4-alpha-16/checkpoint-567/inference_avqa_bs6_seed42/bs6_tf32_mt500_seed42_test/results.jsonl']
    # results=['results/finetune-qwen2/qwen2-7B_moka-ave-r-4-alpha-16-7_23_19:50/checkpoint-279/inference_avqa_bs6_seed42/bs6_tf32_mt500_seed42_test/merged_results.jsonl']
    # results=['results/finetune-llama-ave/llama_music-moka/checkpoint-189/inference_avqa/bs6_tf32_mt150_seed42_test/merged_results.jsonl'] #72.74
    # results=['results/finetune-llama-ave/llama_music-moka-7.24/checkpoint-207/inference_ave/bs1_tf32_mt150_seed42_test/merged_results.jsonl'] #75.3 llama2
    # results=['results/finetune-llama-ave/llama_music-dash-lora/checkpoint-189/inference_avqa/bs1_tf32_mt150_seed42_test/merged_results.jsonl']
    # results=['results/finetune-llama-ave/llama_music-dash-lora/checkpoint-189/inference_ave/bs6_tf32_mt150_seed42_test/merged_results.jsonl'] #3:1:1 72.57
    # results=['results/finetune-llama-ave/llama_music-dash-lora-7.24-20:35/checkpoint-207/inference_ave/bs1_tf32_mt150_seed42_test/merged_results.jsonl'] # 3:1:1 75.15 llama
    # results=['results/finetune-qwen2/qwen2-7B_moka-ave-r-4-alpha-16-7_24_23:02/checkpoint-309/inference_avqa_bs6_seed42/bs6_tf32_mt500_seed42_test/merged_results.jsonl']  #moka 73.31 qwen2
    # results=['results/finetune-qwen2/qwen2-7B_ave-dash-lora-r-4-alpha-16/checkpoint-309/inference_avqa_bs6_seed42/bs6_tf32_mt500_seed42_test/merged_results.jsonl'] # 3:1:1 69.80 qwen2
    # results=['results/finetune-qwen2/qwen2-7B_ave-dash-lora-r-4-alpha-16-ratio-1-7.25-10:36/checkpoint-309/inference_avqa_bs6_seed42/bs6_tf32_mt500_seed42_test/merged_results.jsonl'] #1:1:1  72.79 qwen2
    # results=['results/finetune-qwen2/qwen2-7B_moka-ave-r-4-alpha-16-blc_weight-0.25_7_25_12:06/checkpoint-309/inference_avqa_bs6_seed42/bs6_tf32_mt500_seed42_test/merged_results.jsonl'] #moka 72.79 qwen2 blc_weight-0.25
    # Qwen3-8B Moka, checkpoint-312 (402 samples)
    # JSONL: results/finetune-qwen3/qwen3-8B_ave_moka-r-4/checkpoint-312/inference_ave/bs1_bf16_mt500_test/merged_results.jsonl
    # 最终结果: event=86.82% (349/402), range=62.69% (252/402), event+range=54.48% (219/402), segment=71.07% (2857/4020)
    #
    # Qwen3-8B Dash-LoRA, r=4, alpha=16, ratio=1, top_k_layers=65, checkpoint-312 (402 samples)
    # JSONL: results/finetune-qwen3/qwen3-8B_ave_dash-lora-r-4-alpha-16-ratio-1/checkpoint-312/inference_ave/bs1_bf16_mt500_test/merged_results.jsonl
    # 最终结果: event=89.30% (359/402), range=64.93% (261/402), event+range=58.46% (235/402), segment=74.25% (2985/4020), parse_failures=0
    # Dash-LoRA 相对 Moka: event +2.48 pp, range +2.24 pp, event+range +3.98 pp, segment +3.18 pp
    # results = args.results
    # results = ['/nfs1/outdated/WYT/MokA-copy/results/finetune-llama-ave/llama2-7b-ave-layerwise-shared-countmatched-ratio1/checkpoint-207/inference_ave/bs1_tf32_mt150_seed42_test/merged_results.jsonl'] #layerwise-shared 75.7 llama
    # results = ['/nfs1/outdated/WYT/MokA-copy/results/finetune-llama-ave/llama2-7b-ave-dash-random-ratio1/checkpoint-207/inference_ave/bs1_tf32_mt150_seed42_test/merged_results.jsonl'] #62.44 random-order llama2
    # results=['/nfs1/outdated/WYT/MokA-copy/results/finetune-llama-ave/llama2-7b-ave-dash-scored-ratio1/checkpoint-207/inference_ave/bs1_tf32_mt150_seed42_test/merged_results.jsonl'] # 1:1:1 75.10 llama2
    results=['/nfs1/outdated/WYT/MokA-copy/results/finetune-llama-ave/llama2-7b-ave-dash-crossmodal-strict-ratio1/checkpoint-207/inference_ave/bs1_tf32_mt150_seed42_test/merged_results.jsonl'] # 75.3 llama2

    for result in results:
        if os.path.exists(result):
            check_lenient(result)
        else:
            print(f"File not found: {result}")

if __name__ == "__main__":
    main()


    