import argparse
import csv
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import LunaPatchDataset, make_balanced_sampler
from model import NoduleCNN, count_parameters


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, loss_fn):
    model.eval()
    all_logits, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for patches, labels in loader:
            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(patches)
                loss = loss_fn(logits, labels)
            total_loss += loss.item() * len(labels)
            all_logits.append(logits.float().cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    return {
        "loss": total_loss / len(labels),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs),
    }


def main():
    parser = argparse.ArgumentParser(description="Train 2D nodule CNN")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--resume", action="store_true",
                        help="resume from checkpoints/last{tag}.pt")
    parser.add_argument("--hard-neg", action="store_true",
                        help="add mined hard negatives to train + val")
    parser.add_argument("--hard-neg-weight", type=float, default=3.0,
                        help="over-sampling factor for mined hard negatives")
    parser.add_argument("--tag", type=str, default="",
                        help="suffix for checkpoint / history filenames "
                             "(e.g. _hardneg) so runs don't overwrite")
    args = parser.parse_args()

    set_seed(config.TRAIN_SEED)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = LunaPatchDataset(config.TRAIN_SUBSETS, augment=True,
                                include_hard_neg=args.hard_neg)
    val_ds = LunaPatchDataset(config.VAL_SUBSETS, augment=False,
                              include_hard_neg=args.hard_neg)
    neg, pos = train_ds.class_counts()
    print(f"Train patches: {len(train_ds)} ({pos} pos / {neg} neg), "
          f"val patches: {len(val_ds)}")
    if args.hard_neg:
        print(f"Hard-negative mining ON (over-sampling x{args.hard_neg_weight})")

    sampler = make_balanced_sampler(
        train_ds, hard_neg_weight=args.hard_neg_weight if args.hard_neg else 1.0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler,
                              num_workers=config.NUM_WORKERS, pin_memory=True,
                              persistent_workers=config.NUM_WORKERS > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True,
                            persistent_workers=config.NUM_WORKERS > 0)

    model = NoduleCNN().to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch, best_auc = 0, 0.0
    best_ckpt = config.CHECKPOINT_DIR / f"best{args.tag}.pt"
    last_ckpt = config.CHECKPOINT_DIR / f"last{args.tag}.pt"
    if args.resume and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_auc = state["best_auc"]
        print(f"Resumed from epoch {start_epoch} (best AUC {best_auc:.4f})")

    history_path = config.RESULTS_DIR / f"history{args.tag}.csv"
    history_fields = ["epoch", "train_loss", "val_loss", "val_accuracy",
                      "val_precision", "val_recall", "val_f1", "val_auc", "lr"]
    if not (args.resume and history_path.exists()):
        with open(history_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=history_fields).writeheader()

    epochs_without_improvement = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss, seen = 0.0, 0
        t0 = time.time()
        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for patches, labels in bar:
            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(patches)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * len(labels)
            seen += len(labels)
            bar.set_postfix(loss=f"{running_loss / seen:.4f}")

        scheduler.step()
        train_loss = running_loss / seen

        metrics = evaluate(model, val_loader, device, loss_fn)
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f} "
              f"val_loss={metrics['loss']:.4f} val_f1={metrics['f1']:.4f} "
              f"val_auc={metrics['auc']:.4f} "
              f"({time.time() - t0:.0f}s)")

        with open(history_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=history_fields).writerow({
                "epoch": epoch + 1, "train_loss": round(train_loss, 6),
                "val_loss": round(metrics["loss"], 6),
                "val_accuracy": round(metrics["accuracy"], 6),
                "val_precision": round(metrics["precision"], 6),
                "val_recall": round(metrics["recall"], 6),
                "val_f1": round(metrics["f1"], 6),
                "val_auc": round(metrics["auc"], 6),
                "lr": optimizer.param_groups[0]["lr"],
            })

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            epochs_without_improvement = 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_metrics": metrics,
                        "patch_size_px": config.PATCH_SIZE_PX,
                        "patch_size_mm": config.PATCH_SIZE_MM},
                       best_ckpt)
            print(f"  -> new best model saved (val AUC {best_auc:.4f})")
        else:
            epochs_without_improvement += 1

        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch, "best_auc": best_auc}, last_ckpt)

        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping: no val-AUC improvement for "
                  f"{config.EARLY_STOPPING_PATIENCE} epochs.")
            break

    print(f"Training finished. Best val AUC: {best_auc:.4f}. "
          f"Best model: {best_ckpt}")


if __name__ == "__main__":
    main()
