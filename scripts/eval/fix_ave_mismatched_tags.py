import json
import re
import shutil
from pathlib import Path


ANNOTATION_PATH = Path(
    "MokA_AudioVisualText/AVE_data/train_samples_ave_deduplicated.json"
)
LABEL_ROOT = Path("MokA_AudioVisualText/converted_label_1/converted_label")
BACKUP_ROOT = Path(
    "MokA_AudioVisualText/converted_label_1/backup_before_adding_tags"
)


def normalize_event(value):
    return re.sub(r"\s+", " ", str(value).strip().lower().replace("_", " "))


def main():
    with ANNOTATION_PATH.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    modified = []
    for sample in samples:
        label_path = LABEL_ROOT / f"{sample['vid']}.txt"
        text = label_path.read_text(encoding="utf-8", errors="strict")
        updated = text

        event_match = re.search(r"<event>\s*(.*?)\s*</event>", text, re.I | re.S)
        range_match = re.search(
            r"<range>\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*</range>",
            text,
            re.I | re.S,
        )
        if not event_match or not range_match:
            raise ValueError(f"Missing structured tag after completion: {label_path}")

        if normalize_event(event_match.group(1)) != normalize_event(sample["event"]):
            updated = re.sub(
                r"<event>.*?</event>",
                f"<event> {sample['event']} </event>",
                updated,
                count=1,
                flags=re.I | re.S,
            )

        actual_range = (float(range_match.group(1)), float(range_match.group(2)))
        expected_range = (float(sample["start_time"]), float(sample["end_time"]))
        if actual_range != expected_range:
            updated = re.sub(
                r"<range>.*?</range>",
                f"<range> {sample['start_time']},{sample['end_time']} </range>",
                updated,
                count=1,
                flags=re.I | re.S,
            )

        if updated != text:
            backup_path = BACKUP_ROOT / label_path.name
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if not backup_path.exists():
                shutil.copy2(label_path, backup_path)
            label_path.write_text(updated, encoding="utf-8")
            modified.append(label_path.name)

    print(f"corrected_mismatched_files: {len(modified)}")


if __name__ == "__main__":
    main()
