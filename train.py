"""Train development folds; select checkpoints using validation AUC only."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import Config, DATASET_COUNTS
from dataset import FractureImages, read_manifest, manifest_sha256
from model import FractureDetectionWaveletDenseGNN
from engine import seed_all, MemoryBank, train_epoch, predict, metrics, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=DATASET_COUNTS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.)
    parser.add_argument("--beta", type=float, default=1.)
    parser.add_argument("--wavelet-levels", type=int, default=2)
    parser.add_argument("--memory-capacity", type=int, default=2048)
    parser.add_argument("--loss", choices=["bce", "focal", "dice"], default="bce")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--without-gcn", action="store_true")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not set(args.folds).issubset(range(5)) or len(set(args.folds)) != len(args.folds):
        parser.error("Folds must be distinct integers from 0 to 4")
    if args.epochs < 1 or args.batch_size < 1 or args.memory_capacity < 1 or args.wavelet_levels < 1:
        parser.error("Epochs, batch size, memory capacity and decomposition depth must be positive")
    rows = read_manifest(args.manifest, args.dataset)
    if args.check_only:
        print(f"Validated {len(rows)} images, five development folds and independent test membership.")
        return
    cfg = Config(device=args.device, batch_size=args.batch_size, epochs=args.epochs,
                 alpha=args.alpha, beta=args.beta, wavelet_levels=args.wavelet_levels,
                 wavelet_ms_channels=12*args.wavelet_levels, memory_bank_size=args.memory_capacity,
                 seed=args.seed).validate()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_hash = manifest_sha256(args.manifest)
    run = {"config": cfg.to_dict(), "dataset": args.dataset, "manifest_sha256": manifest_hash,
           "loss": args.loss, "gamma": args.gamma, "use_gcn": not args.without_gcn,
           "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(cfg.device) if cfg.device.startswith("cuda") else "CPU"}
    results = []
    for fold in args.folds:
        folder = args.output / f"fold_{fold}"
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"Choose a new output directory; fold already contains results: {folder}")
        folder.mkdir(parents=True, exist_ok=True)
        seed_all(cfg.seed + fold)
        training_rows = [r for r in rows if r["split"] == "development" and r["fold"] != fold]
        validation_rows = [r for r in rows if r["split"] == "development" and r["fold"] == fold]
        train_data = FractureImages(training_rows, cfg.image_size, True)
        val_data = FractureImages(validation_rows, cfg.image_size, False)
        train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False, num_workers=args.workers)
        model = FractureDetectionWaveletDenseGNN(cfg, run["use_gcn"]).to(cfg.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.device.startswith("cuda"))
        bank = MemoryBank(cfg.memory_bank_size)
        best_auc, best_metrics = -1., None
        write_json(folder / "config.json", {**run, "fold": fold, "seed": cfg.seed+fold,
                    "train_images": len(train_data), "validation_images": len(val_data)})
        log_path = folder / "metrics.csv"
        for epoch in range(1, cfg.epochs+1):
            train_loss = train_epoch(model, train_loader, optimizer, scaler, bank, cfg, epoch, args.loss, args.gamma)
            y, p = predict(model, val_loader, cfg)
            result = metrics(y, p, threshold=cfg.threshold)
            scheduler.step(result["auc"])
            log = {"epoch": epoch, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"],
                   "memory_entries": len(bank.entries), **result}
            with log_path.open("a", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(log))
                if epoch == 1:
                    writer.writeheader()
                writer.writerow(log)
            if result["auc"] > best_auc:
                best_auc, best_metrics = result["auc"], result
                checkpoint = {**run, "fold": fold, "epoch": epoch, "model_state_dict": model.state_dict(),
                              "validation_metrics": result, "optimizer_state_dict": optimizer.state_dict(),
                              "scheduler_state_dict": scheduler.state_dict()}
                torch.save(checkpoint, folder / "best_model.pth")
                with (folder / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["image_path", "label", "probability", "prediction"])
                    writer.writerows((r["image_path"], int(label), float(prob), int(prob >= cfg.threshold))
                                     for r, label, prob in zip(validation_rows, y, p))
            print(f"fold={fold} epoch={epoch}/{cfg.epochs} train_loss={train_loss:.5f} val_auc={result['auc']:.5f} val_f1={result['f1']:.5f}", flush=True)
        write_json(folder / "best_validation_metrics.json", best_metrics)
        results.append({"fold": fold, **best_metrics})
    summary = {"fold_results": results, "completed_folds": len(results)}
    if len(results) == 5:
        summary["five_fold"] = {name: {"mean": float(np.mean([r[name] for r in results])),
                                          "std": float(np.std([r[name] for r in results], ddof=1))}
                                for name in ["precision", "recall", "f1", "auc", "accuracy"]}
    write_json(args.output / "cross_validation_summary.json", summary)


if __name__ == "__main__":
    main()
