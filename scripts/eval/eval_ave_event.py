# -*- coding: utf-8 -*-
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def normalize_event(text):
    text = (text or "").strip().lower().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;")


def extract_gt_event(output):
    output = (output or "").strip().lower()
    match = re.search(r"event:\s*(.*?)\s*start_time", output, flags=re.I | re.S)
    if match:
        return normalize_event(match.group(1))
    return normalize_event(output.splitlines()[0] if output else "")


def clean_prediction(prediction):
    prediction = prediction or ""
    for token in ("<|endoftext|>", "<s>", "</s>", "<unk>"):
        prediction = prediction.replace(token, " ")
    return prediction.strip()


def extract_pred_event(prediction):
    pred = clean_prediction(prediction)
    pred_lower = pred.lower()

    event_tag_match = re.search(r"<event>\s*(.*?)\s*</event>", pred_lower, flags=re.I | re.S)
    if event_tag_match:
        return normalize_event(event_tag_match.group(1))

    first_line = pred_lower.splitlines()[0].strip() if pred_lower else ""
    event_colon_match = re.search(r"event:\s*(.*?)\s*start_time", first_line, flags=re.I | re.S)
    if event_colon_match:
        return normalize_event(event_colon_match.group(1))

    answer_match = re.search(r"the answer is\s*(.*?)(?:\.|$)", first_line, flags=re.I | re.S)
    if answer_match:
        return normalize_event(answer_match.group(1))

    return normalize_event(first_line)


def evaluate(path, show_confusions=20):
    path = Path(path)
    stats = defaultdict(lambda: {"strict": 0, "lenient": 0, "count": 0})
    pred_counter = Counter()
    confusion_counter = Counter()
    bad_records = []

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            gt_event = extract_gt_event(sample.get("output", ""))
            pred_event = extract_pred_event(sample.get("predict", ""))
            pred_text = normalize_event(clean_prediction(sample.get("predict", "")))

            strict_hit = int(pred_event == gt_event)
            lenient_hit = int(gt_event in pred_event or gt_event in pred_text)

            stats[gt_event]["strict"] += strict_hit
            stats[gt_event]["lenient"] += lenient_hit
            stats[gt_event]["count"] += 1
            pred_counter[pred_event] += 1
            confusion_counter[(gt_event, pred_event)] += 1

            if not strict_hit and len(bad_records) < 5:
                bad_records.append((idx, gt_event, pred_event, sample.get("predict", "")[:240]))

    total = sum(v["count"] for v in stats.values())
    strict_total = sum(v["strict"] for v in stats.values())
    lenient_total = sum(v["lenient"] for v in stats.values())

    print(f"Checking: {path}")
    print(f"{'Event Category':<32} | {'Strict Acc':<12} | {'Lenient Acc':<12} | {'Count':<8}")
    print("-" * 78)
    for event in sorted(stats):
        item = stats[event]
        count = item["count"]
        strict_acc = 100 * item["strict"] / count if count else 0.0
        lenient_acc = 100 * item["lenient"] / count if count else 0.0
        print(f"{event:<32} | {strict_acc:>6.2f} %     | {lenient_acc:>6.2f} %     | {count:<8}")
    print("-" * 78)
    print(f"{'Overall':<32} | {100 * strict_total / total:>6.2f} %     | {100 * lenient_total / total:>6.2f} %     | {total:<8}")

    print("\nTop predicted events:")
    for event, count in pred_counter.most_common(show_confusions):
        print(f"{count:>5}  {event}")

    print("\nTop confusions:")
    shown = 0
    for (gt_event, pred_event), count in confusion_counter.most_common():
        if gt_event == pred_event:
            continue
        print(f"{count:>5}  GT={gt_event}  PRED={pred_event}")
        shown += 1
        if shown >= show_confusions:
            break

    if bad_records:
        print("\nFirst wrong examples:")
        for idx, gt_event, pred_event, pred in bad_records:
            print(f"#{idx}: GT={gt_event} | PRED={pred_event} | raw={pred!r}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AVE event predictions from jsonl results.")
    parser.add_argument(
        "result_path",
        nargs="?",
        default="results/finetune-qwen2/qwen2-7B_moka-ave-r-4-alpha-16/checkpoint-288/inference_avqa_bs6_seed42/bs6_tf32_mt500_seed42_test/merged_results.jsonl",
    )
    parser.add_argument("--show-confusions", type=int, default=20)
    args = parser.parse_args()
    evaluate(args.result_path, show_confusions=args.show_confusions)


if __name__ == "__main__":
    main()
