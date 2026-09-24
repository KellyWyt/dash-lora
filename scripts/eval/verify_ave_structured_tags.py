import json
import re
from pathlib import Path


ANNOTATION_PATH = Path(
    "MokA_AudioVisualText/AVE_data/train_samples_ave_deduplicated.json"
)
LABEL_ROOT = Path("MokA_AudioVisualText/converted_label_1/converted_label")


def normalize_event(value):
    return re.sub(r"\s+", " ", str(value).strip().lower().replace("_", " "))


def main():
    with ANNOTATION_PATH.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    missing_files = []
    missing_tags = []
    event_mismatches = []
    range_mismatches = []

    for sample in samples:
        label_path = LABEL_ROOT / f"{sample['vid']}.txt"
        if not label_path.exists():
            missing_files.append(sample["vid"])
            continue

        text = label_path.read_text(encoding="utf-8", errors="strict")
        event_match = re.search(r"<event>\s*(.*?)\s*</event>", text, re.I | re.S)
        range_match = re.search(
            r"<range>\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*</range>",
            text,
            re.I | re.S,
        )
        if not event_match or not range_match:
            missing_tags.append(sample["vid"])
            continue

        if normalize_event(event_match.group(1)) != normalize_event(sample["event"]):
            event_mismatches.append(sample["vid"])

        actual_range = (float(range_match.group(1)), float(range_match.group(2)))
        expected_range = (float(sample["start_time"]), float(sample["end_time"]))
        if actual_range != expected_range:
            range_mismatches.append(sample["vid"])

    print(f"samples: {len(samples)}")
    print(f"missing_files: {len(missing_files)}")
    print(f"missing_tags: {len(missing_tags)}")
    print(f"event_mismatches: {len(event_mismatches)}")
    print(f"range_mismatches: {len(range_mismatches)}")

    if missing_files or missing_tags or event_mismatches or range_mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
