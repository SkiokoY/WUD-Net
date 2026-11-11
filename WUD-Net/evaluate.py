"""Evaluate frozen checkpoints on the fixed independent test set."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import Config, DATASET_COUNTS
from dataset import FractureImages, read_manifest, manifest_sha256
from model import FractureDetectionWaveletDenseGNN
from engine import predict, metrics, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=DATASET_COUNTS)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=["vote", "single"])
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    rows = read_manifest(args.manifest, args.dataset)
    test_rows = [r for r in rows if r["split"] == "test"]
    mode = args.mode or ("vote" if args.dataset == "mura" else "single")
    folds = list(range(5)) if mode == "vote" else [args.fold]
    manifest_hash = manifest_sha256(args.manifest)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose a new test output directory")
    probabilities, identities, labels = [], [], None
    for fold in folds:
        checkpoint = torch.load(args.run / f"fold_{fold}" / "best_model.pth", map_location="cpu", weights_only=False)
        if checkpoint["manifest_sha256"] != manifest_hash or checkpoint["dataset"] != args.dataset or checkpoint["fold"] != fold:
            raise ValueError("Checkpoint does not match the manifest/dataset/fold")
        cfg = Config(**checkpoint["config"])
        cfg.device = args.device
        cfg.validate()
        identities.append((checkpoint["loss"], checkpoint["gamma"], checkpoint["use_gcn"],
                           cfg.alpha, cfg.beta, cfg.wavelet_levels, cfg.memory_bank_size))
        if any(identity != identities[0] for identity in identities):
            raise ValueError("Ensemble checkpoints use different model/training settings")
        model = FractureDetectionWaveletDenseGNN(cfg, checkpoint["use_gcn"]).to(cfg.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        loader = DataLoader(FractureImages(test_rows, cfg.image_size), batch_size=args.batch_size, shuffle=False)
        current_labels, p = predict(model, loader, cfg)
        if labels is not None and not np.array_equal(labels, current_labels):
            raise ValueError("Test sample order changed across models")
        labels = current_labels
        probabilities.append(p)
        del model
    probabilities = np.stack(probabilities)
    mean_probability = probabilities.mean(axis=0)
    votes = (probabilities >= 0.5).sum(axis=0)
    predictions = (votes > len(folds)/2).astype(int)
    result = metrics(labels, mean_probability, predictions)
    result.update(mode=mode, folds=folds, threshold=0.5, manifest_sha256=manifest_hash,
                  auc_score="mean model probability" if mode == "vote" else "single model probability")
    args.output.mkdir(parents=True)
    write_json(args.output / "test_metrics.json", result)
    with (args.output / "test_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_path", "label", "mean_probability", "positive_votes", "prediction"] + [f"probability_fold_{fold}" for fold in folds])
        for i, row in enumerate(test_rows):
            writer.writerow([row["image_path"], int(labels[i]), float(mean_probability[i]), int(votes[i]), int(predictions[i])] + probabilities[:, i].tolist())
    # Rows are predictions and columns are reference classes, matching the manuscript.
    matrix = np.array([[result["tn"], result["fn"]], [result["fp"], result["tp"]]])
    np.savetxt(args.output / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max()/2 else "black", fontsize=16)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Non-fracture", "Fracture"],
           yticklabels=["Non-fracture", "Fracture"], xlabel="Reference class", ylabel="Predicted class")
    fig.tight_layout()
    fig.savefig(args.output / "confusion_matrix.png", dpi=300)
    plt.close(fig)
    print(result)


if __name__ == "__main__":
    main()
