"""Evaluate the trained 2D CNN on the held-out test subset (subset 9).

The test subset keeps ALL candidate negatives (no subsampling), so the
reported metrics reflect the real candidate distribution of the LUNA16
false-positive-reduction task.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (auc, average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import LunaPatchDataset
from model import NoduleCNN


def predict(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for patches, labels in tqdm(loader, desc="Predicting"):
            patches = patches.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(patches)
            all_probs.append(torch.sigmoid(logits.float()).cpu())
            all_labels.append(labels)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy().astype(int)


def main():
    parser = argparse.ArgumentParser(description="Evaluate 2D nodule CNN")
    parser.add_argument("--checkpoint", type=str,
                        default=str(config.CHECKPOINT_DIR / "best.pt"))
    args = parser.parse_args()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = NoduleCNN().to(device)
    model.load_state_dict(state["model"])
    print(f"Loaded {args.checkpoint} (epoch {state['epoch'] + 1}, "
          f"val AUC {state['val_metrics']['auc']:.4f})")

    test_ds = LunaPatchDataset(config.TEST_SUBSETS, augment=False)
    neg, pos = test_ds.class_counts()
    print(f"Test patches: {len(test_ds)} ({pos} pos / {neg} neg)")
    loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS, pin_memory=True)

    probs, labels = predict(model, loader, device)
    preds = (probs > 0.75).astype(int)

    # ----- text report -------------------------------------------------
    report = classification_report(labels, preds,
                                   target_names=["non-nodule", "nodule"],
                                   digits=4)
    cm = confusion_matrix(labels, preds)
    auc_score = roc_auc_score(labels, probs)
    ap_score = average_precision_score(labels, probs)

    lines = [
        f"Checkpoint: {args.checkpoint}",
        f"Test subset(s): {config.TEST_SUBSETS} "
        f"({len(test_ds)} patches, {pos} positive)",
        "",
        report,
        f"Confusion matrix (rows=true, cols=pred):\n{cm}",
        "",
        f"ROC AUC:           {auc_score:.4f}",
        f"Average precision: {ap_score:.4f}",
    ]
    text = "\n".join(lines)
    print(text)
    (config.RESULTS_DIR / "report.txt").write_text(text)

    # ----- predictions csv ---------------------------------------------
    pd.DataFrame({
        "seriesuid": test_ds.index["seriesuid"],
        "label": labels,
        "probability": probs,
    }).to_csv(config.RESULTS_DIR / "test_predictions.csv", index=False)

    # ----- plots --------------------------------------------------------
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.4f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve - test subset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "roc_curve.png", dpi=150)
    plt.close()

    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap_score:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve - test subset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "pr_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.hist(probs[labels == 0], bins=50, alpha=0.6, label="non-nodule",
             log=True)
    plt.hist(probs[labels == 1], bins=50, alpha=0.6, label="nodule", log=True)
    plt.xlabel("Predicted probability")
    plt.ylabel("Count (log)")
    plt.title("Predicted probability distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "probability_distribution.png", dpi=150)
    plt.close()

    print(f"\nPlots and report saved to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
