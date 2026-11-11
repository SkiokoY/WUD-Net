"""Training losses, staged FIFO replay, and fixed-threshold evaluation."""
from collections import deque
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def targets(p, y):
    detached = p.detach()
    return 1 - (detached - y).abs(), 4 * detached * (1 - detached)


def difficulty(c_logit, u_logit, alpha=1., beta=1.):
    return alpha * u_logit.sigmoid() + beta * (1 - c_logit.sigmoid())


def replay_ratio(epoch, cfg):
    return 0. if epoch <= cfg.stage1_end else cfg.stage2_mem_ratio if epoch <= cfg.stage2_end else cfg.stage3_mem_ratio


class MemoryBank:
    """FIFO of training-example indices; replay applies training augmentation anew."""
    def __init__(self, capacity):
        self.entries = deque(maxlen=capacity)

    def add(self, indices):
        self.entries.extend(int(i) for i in indices)

    def sample(self, n):
        return random.sample(list(self.entries), min(n, len(self.entries)))


def objective(logits, y, cfg, loss_name="bce", gamma=1.5):
    p_logit, c_logit, u_logit = [item.flatten() for item in logits]
    p, c, u = p_logit.sigmoid(), c_logit.sigmoid(), u_logit.sigmoid()
    binary = F.binary_cross_entropy_with_logits(p_logit, y, reduction="none")
    if loss_name == "bce":
        classification = binary.mean()
    elif loss_name == "focal":
        pt = torch.where(y == 1, p, 1 - p)
        classification = ((1 - pt).pow(gamma) * binary).mean()
    elif loss_name == "dice":
        classification = 1 - (2 * (p * y).sum() + 1e-6) / (p.sum() + y.sum() + 1e-6)
    else:
        raise ValueError(loss_name)
    c_target, u_target = targets(p, y)
    return classification + cfg.conf_loss_weight * F.mse_loss(c, c_target) + cfg.uncert_loss_weight * F.mse_loss(u, u_target)


def train_epoch(model, loader, optimizer, scaler, bank, cfg, epoch, loss_name="bce", gamma=1.5):
    model.train()
    total, count = 0., 0
    ratio = replay_ratio(epoch, cfg)
    for batch in loader:
        x, y = batch["image"].to(cfg.device), batch["label"].to(cfg.device)
        model.eval()
        with torch.no_grad(), torch.autocast(device_type=torch.device(cfg.device).type, enabled=cfg.device.startswith("cuda")):
            _, c, u = model(x)
            k = max(1, int(np.ceil(len(y) * cfg.memory_select_ratio)))
            selected = difficulty(c.flatten(), u.flatten(), cfg.alpha, cfg.beta).topk(k).indices.cpu()
        bank.add(batch["index"][selected].tolist())
        model.train()
        replay = bank.sample(round(len(y) * ratio)) if ratio > 0 else []
        if replay:
            current = torch.randperm(len(y), device=x.device)[:len(y) - len(replay)]
            examples = [loader.dataset[i] for i in replay]
            x = torch.cat([x[current], torch.stack([r["image"] for r in examples]).to(cfg.device)])
            y = torch.cat([y[current], torch.stack([r["label"] for r in examples]).to(cfg.device)])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=torch.device(cfg.device).type, enabled=cfg.device.startswith("cuda")):
            loss = objective(model(x), y, cfg, loss_name, gamma)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach()) * len(y)
        count += len(y)
    return total / count


def metrics(labels, probabilities, predictions=None, threshold=0.5):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int) if predictions is None else np.asarray(predictions)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {"precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "auc": float(roc_auc_score(labels, probabilities)), "accuracy": float(accuracy_score(labels, predictions)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n": len(labels)}


@torch.no_grad()
def predict(model, loader, cfg):
    model.eval()
    labels, probabilities = [], []
    for batch in loader:
        with torch.autocast(device_type=torch.device(cfg.device).type, enabled=cfg.device.startswith("cuda")):
            p, _, _ = model(batch["image"].to(cfg.device))
        labels.extend(batch["label"].tolist())
        probabilities.extend(p.flatten().sigmoid().float().cpu().tolist())
    return np.asarray(labels), np.asarray(probabilities)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
