import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from dataset import LunaPatchDataset
from model import NoduleCNN

THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]


def predict(checkpoint, loader, device):
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model = NoduleCNN().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    probs = []
    with torch.no_grad():
        for patches, _ in loader:
            patches = patches.to(device)
            probs.append(torch.sigmoid(model(patches).float()).cpu())
    return torch.cat(probs).numpy()


def report(name, probs, labels, n_scans):
    pos, neg = labels == 1, labels == 0
    print(f"\n=== {name} ===")
    print(f"{'thr':>5} {'sensitivity':>12} {'FP/scan':>9} {'nodules kept':>14}")
    for thr in THRESHOLDS:
        tp = int((probs[pos] >= thr).sum())
        fp = int((probs[neg] >= thr).sum())
        print(f"{thr:>5.2f} {tp / pos.sum():>11.1%} {fp / n_scans:>9.1f} "
              f"{tp:>7d}/{int(pos.sum())}")


def main():
    parser = argparse.ArgumentParser(description="Compare checkpoints on "
                                                 "real subset-9 candidates")
    parser.add_argument("--a", type=str,
                        default=str(config.CHECKPOINT_DIR / "best.pt"))
    parser.add_argument("--b", type=str,
                        default=str(config.CHECKPOINT_DIR / "best_hardneg.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Subset-9 watershed candidates only (hard-neg source, all kept).
    ds = LunaPatchDataset(config.TEST_SUBSETS, augment=False,
                          include_cand=False, include_hard_neg=True)
    labels = ds.labels.astype(int)
    n_scans = ds.index["seriesuid"].nunique()
    print(f"Subset 9 real pipeline candidates: {len(ds)} "
          f"({int((labels == 1).sum())} positive) over {n_scans} scans")
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS, pin_memory=True)

    report(f"A: {args.a}", predict(args.a, loader, device), labels, n_scans)
    report(f"B: {args.b}", predict(args.b, loader, device), labels, n_scans)


if __name__ == "__main__":
    main()
