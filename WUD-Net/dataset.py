"""Load explicitly curated fracture labels and fixed development/test memberships."""
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from common import DATASET_COUNTS


def read_manifest(path, dataset_name, check_files=True):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_path", "label", "split", "fold", "group_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires image_path,label,split,fold,group_id; see README.md")
        rows = list(reader)
    seen, groups = set(), defaultdict(set)
    for row in rows:
        if row["label"] not in {"0", "1"}:
            raise ValueError("Use manually verified fracture labels: 1=fracture, 0=non-fracture")
        row["label"] = int(row["label"])
        if row["split"] not in {"development", "test"} or not row["group_id"].strip():
            raise ValueError("Every image requires split=development/test and a nonempty group_id")
        row["fold"] = int(row["fold"]) if row["split"] == "development" else -1
        if row["split"] == "development" and row["fold"] not in range(5):
            raise ValueError("Development fold must be 0,1,2,3,4")
        image = Path(row["image_path"])
        image = (path.parent / image).resolve() if not image.is_absolute() else image.resolve()
        row["image_path"] = str(image)
        identity = str(image).casefold()
        if identity in seen:
            raise ValueError(f"Image occurs more than once: {image}")
        seen.add(identity)
        if check_files and not image.is_file():
            raise FileNotFoundError(image)
        groups[row["group_id"]].add((row["split"], row["fold"]))
    if any(len(assignments) != 1 for assignments in groups.values()):
        raise ValueError("A patient/source group crosses folds or the development/test boundary")
    total, development, test = DATASET_COUNTS[dataset_name]
    counts = Counter(row["split"] for row in rows)
    if (len(rows), counts["development"], counts["test"]) != (total, development, test):
        raise ValueError(f"{dataset_name} manuscript counts require total/development/test={total}/{development}/{test}; received {len(rows)}/{counts['development']}/{counts['test']}")
    for fold in range(5):
        if {r["label"] for r in rows if r["fold"] == fold} != {0, 1}:
            raise ValueError(f"Validation fold {fold} must contain both classes")
    if dataset_name == "mura":
        if Counter(r["label"] for r in rows) != {0: 656, 1: 656}:
            raise ValueError("The curated MURA subset must contain 656 fractures and 656 controls")
        if Counter(r["label"] for r in rows if r["split"] == "test") != {0: 116, 1: 81}:
            raise ValueError("The manuscript MURA test set contains 116 non-fractures and 81 fractures")
    return rows


def manifest_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FractureImages(Dataset):
    def __init__(self, rows, image_size=384, training=False):
        self.rows = rows
        operations = [transforms.Resize((image_size, image_size))]
        if training:
            operations += [transforms.RandomResizedCrop(image_size), transforms.RandomHorizontalFlip(0.5),
                           transforms.RandomRotation(30), transforms.ColorJitter(brightness=0.2, contrast=0.2)]
        operations += [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            x = self.transform(image.convert("RGB"))
        return {"image": x, "label": torch.tensor(row["label"], dtype=torch.float32), "index": index}
