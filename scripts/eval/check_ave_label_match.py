import json
import re
from collections import Counter
from pathlib import Path


def norm(text):
    text = str(text or "").strip().lower().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def get_vid(sample):
    return Path(str(sample.get("vid") or sample.get("video_id") or sample.get("id") or sample.get("filename"))).stem


def get_event(sample):
    return norm(sample.get("event") or sample.get("label") or sample.get("category"))


def extract_label_event(text):
    match = re.search(r"<event>\s*(.*?)\s*</event>", text.lower(), flags=re.I | re.S)
    if not match:
        return "<no_event>"
    return norm(match.group(1))


def main():
    json_path = Path("MokA_AudioVisualText/AVE_data/train_samples_ave.json")
    label_root = Path("MokA_AudioVisualText/converted_label")
    samples = json.load(open(json_path, "r", encoding="utf-8"))

    print("json:", json_path)
    print("type:", type(samples).__name__, "len:", len(samples))
    print("first:", samples[0])

    total = match = missing = no_event = 0
    mismatches = Counter()
    examples = []

    for sample in samples:
        vid = get_vid(sample)
        gt = get_event(sample)
        label_path = label_root / f"{vid}.txt"
        if not label_path.exists():
            missing += 1
            continue

        pred = extract_label_event(label_path.read_text(encoding="utf-8", errors="ignore"))
        total += 1
        if pred == "<no_event>":
            no_event += 1
        if pred == gt:
            match += 1
        else:
            mismatches[(gt, pred)] += 1
            if len(examples) < 10:
                examples.append((vid, gt, pred))

    print("checked:", total)
    print("match:", match)
    print("accuracy:", match / total if total else 0)
    print("missing labels:", missing)
    print("no <event> labels:", no_event)
    print("top mismatches:")
    for (gt, pred), count in mismatches.most_common(20):
        print(count, "json=" + gt, "label=" + pred)
    print("examples:")
    for item in examples:
        print(item)


if __name__ == "__main__":
    main()
