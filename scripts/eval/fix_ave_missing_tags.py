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


def main():
    with ANNOTATION_PATH.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    annotations = {sample["vid"]: sample for sample in samples}
    modified = []

    for vid, sample in annotations.items():
        label_path = LABEL_ROOT / f"{vid}.txt"
        text = label_path.read_text(encoding="utf-8", errors="strict").strip()
        has_event = re.search(r"<event>.*?</event>", text, flags=re.I | re.S)
        has_range = re.search(r"<range>.*?</range>", text, flags=re.I | re.S)

        if has_event and has_range:
            continue

        backup_path = BACKUP_ROOT / label_path.name
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if not backup_path.exists():
            shutil.copy2(label_path, backup_path)

        additions = []
        if not has_event:
            additions.append(
                "The audible and visible event in the video is "
                f"<event> {sample['event']} </event>."
            )
        if not has_range:
            additions.append(
                "The time range is "
                f"<range> {sample['start_time']},{sample['end_time']} </range>."
            )
        suffix = " " + " ".join(additions)
        label_path.write_text(text + suffix + "\n", encoding="utf-8")
        modified.append(label_path.name)

    print(f"modified: {len(modified)}")
    print(f"backup: {BACKUP_ROOT}")


if __name__ == "__main__":
    main()
