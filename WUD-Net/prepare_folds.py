"""Assign five development folds while preserving an explicitly curated test split."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from dataset import read_manifest
from common import DATASET_COUNTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Curated image_path,label,split,group_id CSV")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=DATASET_COUNTS)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"image_path", "label", "split", "group_id"}.issubset(reader.fieldnames or []):
            parser.error("Input requires image_path,label,split,group_id")
        rows = list(reader)
    for row in rows:
        image = Path(row["image_path"])
        row["image_path"] = str((args.input.resolve().parent / image).resolve()) if not image.is_absolute() else str(image.resolve())
        row["fold"] = "-1"
    development = [r for r in rows if r["split"] == "development"]
    labels = np.array([int(r["label"]) for r in development])
    groups = np.array([r["group_id"] for r in development])
    selected = None
    for offset in range(100):
        split_seed = args.seed + offset
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
        candidate = list(splitter.split(np.zeros(len(development)), labels, groups))
        if all(set(labels[training]) == {0, 1} and set(labels[validation]) == {0, 1}
               for training, validation in candidate):
            selected = candidate
            break
    if selected is None:
        raise ValueError("Unable to assign five group-disjoint folds containing both classes; inspect curated groups")
    for fold, (_, validation) in enumerate(selected):
        for index in validation:
            development[index]["fold"] = str(fold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_path", "label", "split", "fold", "group_id"])
            writer.writeheader()
            writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
        read_manifest(temporary, args.dataset)
        temporary.replace(args.output)
        args.output.with_suffix(".split_info.json").write_text(json.dumps({
            "seed": split_seed, "selection": "both classes required in every fold; no performance-based selection",
            "fold_images": [len(validation) for _, validation in selected],
        }, indent=2), encoding="utf-8")
    finally:
        if temporary.exists():
            temporary.unlink()
    print(args.output)


if __name__ == "__main__":
    main()
