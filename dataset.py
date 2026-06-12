import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

import config


class LunaPatchDataset(Dataset):
    """2D patch dataset.

    Args:
        subsets:          LUNA16 subset indices to include.
        augment:          random flips / rotations / intensity jitter (train).
        include_cand:     include the candidates_V2 source at all.
        include_cand_neg: within the candidates_V2 source, include negatives
                          (positives are always kept when include_cand=True).
        include_hard_neg: include the mined hard-negative source.
    """

    def __init__(self, subsets, augment=False, include_cand=True,
                 include_cand_neg=True, include_hard_neg=False):
        self.augment = augment
        self.arrays = {}   # keyed by (source, subset)
        frames = []

        if include_cand and config.INDEX_CSV.exists():
            cand = pd.read_csv(config.INDEX_CSV)
            cand = cand[cand["subset"].isin(subsets)].copy()
            if not include_cand_neg:
                cand = cand[cand["label"] == 1]
            cand["source"] = "cand"
            frames.append(cand)
            for s in subsets:
                p = config.PATCHES_DIR / f"subset{s}_patches.npy"
                if p.exists():
                    self.arrays[("cand", s)] = np.load(p, mmap_mode="r")

        if include_hard_neg and config.HARDNEG_INDEX_CSV.exists():
            hard = pd.read_csv(config.HARDNEG_INDEX_CSV)
            hard = hard[hard["subset"].isin(subsets)].copy()
            hard["source"] = "hardneg"
            frames.append(hard)
            for s in subsets:
                p = config.PATCHES_DIR / f"hardneg_subset{s}.npy"
                if p.exists():
                    self.arrays[("hardneg", s)] = np.load(p, mmap_mode="r")

        if not frames:
            raise RuntimeError("No patch sources found for the requested "
                               "subsets. Run preprocessing.py / "
                               "mine_hard_negatives.py first.")

        self.index = pd.concat(frames, ignore_index=True)
        self.labels = self.index["label"].to_numpy()
        self.sources = self.index["source"].to_numpy()

    def __len__(self):
        return len(self.index)

    def _augment(self, patch):
        k = int(torch.randint(0, 4, (1,)))
        if k:
            patch = torch.rot90(patch, k, dims=(-2, -1))
        if torch.rand(1) < 0.5:
            patch = torch.flip(patch, dims=(-1,))
        if torch.rand(1) < 0.5:
            patch = torch.flip(patch, dims=(-2,))
        patch = patch + 0.05 * (torch.rand(1) - 0.5)
        patch = patch + 0.01 * torch.randn_like(patch)
        return patch.clamp(0.0, 1.0)

    def __getitem__(self, i):
        row = self.index.iloc[i]
        arr = self.arrays[(row["source"], row["subset"])]
        patch = np.asarray(arr[row["idx"]], dtype=np.float32)
        patch = torch.from_numpy(patch).unsqueeze(0)  # (1, H, W)
        if self.augment:
            patch = self._augment(patch)
        return patch, torch.tensor(float(row["label"]))

    def class_counts(self):
        neg = int((self.labels == 0).sum())
        pos = int((self.labels == 1).sum())
        return neg, pos


def make_balanced_sampler(dataset, hard_neg_weight=1.0):
    """WeightedRandomSampler giving each batch ~balanced classes.

    `hard_neg_weight` > 1 additionally over-samples mined hard negatives
    relative to the easy candidates_V2 negatives, focusing training on the
    false positives that matter.
    """
    neg, pos = dataset.class_counts()
    base = {0: 1.0 / max(neg, 1), 1: 1.0 / max(pos, 1)}
    weights = np.array([base[int(l)] for l in dataset.labels], dtype=np.float64)
    if hard_neg_weight != 1.0:
        is_hard_neg = (dataset.labels == 0) & (dataset.sources == "hardneg")
        weights[is_hard_neg] *= hard_neg_weight
    return WeightedRandomSampler(torch.from_numpy(weights),
                                 num_samples=len(dataset), replacement=True)
