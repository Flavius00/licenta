import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
from lung_segmentation import segment_lungs
from model import NoduleCNN
from nodule_candidates import (add_world_coordinates, generate_candidates,
                               match_to_annotations)
from preprocessing import extract_patch, find_scans, load_volume

EMPTY = np.zeros((0, config.PATCH_SIZE_PX, config.PATCH_SIZE_PX), np.float16)


def process_scan(uid, mhd_path, max_neg):
    annotations = pd.read_csv(config.ANNOTATIONS_CSV)
    annotations = annotations[annotations["seriesuid"] == uid]

    volume, origin, spacing = load_volume(mhd_path)
    lung_mask = segment_lungs(volume, spacing)
    cands = generate_candidates(volume, spacing, lung_mask)
    cands = add_world_coordinates(cands, origin, spacing)
    cands = match_to_annotations(cands, annotations, origin, spacing)

    patches, labels = [], []
    for cand in cands:
        voxel_xyz = cand["centroid_voxel"][::-1]  # (z,y,x) -> (x,y,z)
        patch = extract_patch(volume, spacing, voxel_xyz)
        if patch is None:
            continue
        patches.append(np.asarray(patch, dtype=np.float16))
        labels.append(int(cand["label"]))

    if not patches:
        return uid, EMPTY, np.zeros(0, np.int64)
    patches = np.stack(patches)
    labels = np.array(labels, dtype=np.int64)

    if max_neg is not None:
        neg_idx = np.flatnonzero(labels == 0)
        pos_idx = np.flatnonzero(labels == 1)
        if len(neg_idx) > max_neg:
            seed = int.from_bytes(uid.encode()[-4:], "little")
            rng = np.random.default_rng(seed)
            neg_idx = rng.choice(neg_idx, max_neg, replace=False)
        keep = np.concatenate([pos_idx, neg_idx])
        patches, labels = patches[keep], labels[keep]
    return uid, patches, labels


def score_patches(model, device, patches):
    probs = np.empty(len(patches), dtype=np.float32)
    for start in range(0, len(patches), 512):
        batch = torch.from_numpy(
            patches[start:start + 512].astype(np.float32)).unsqueeze(1).to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(batch).float()).cpu().numpy()
        probs[start:start + len(p)] = p
    return probs


def main():
    parser = argparse.ArgumentParser(description="Mine hard negatives")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--subsets", type=int, nargs="+",
                        default=list(range(10)))
    args = parser.parse_args()

    config.PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NoduleCNN().to(device)
    state = torch.load(config.CHECKPOINT_DIR / "best.pt", map_location=device,
                       weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Loaded current model (val AUC {state['val_metrics']['auc']:.4f}) "
          f"on {device}")

    scans = find_scans(config.DATA_DIR)
    index_rows = []
    if config.HARDNEG_INDEX_CSV.exists():
        index_rows = pd.read_csv(config.HARDNEG_INDEX_CSV).to_dict("records")

    for subset in args.subsets:
        out_npy = config.PATCHES_DIR / f"hardneg_subset{subset}.npy"
        if out_npy.exists():
            print(f"subset{subset}: already mined, skipping.")
            continue

        subset_scans = {uid: p for uid, (p, s) in scans.items() if s == subset}
        is_trainval = subset in config.TRAINVAL_SUBSETS
        max_neg = config.MINE_MAX_NEG_PER_SCAN if is_trainval else None
        print(f"\nsubset{subset}: {len(subset_scans)} scans "
              f"({'hard-mined' if is_trainval else 'test: keep all'})")

        all_patches, all_labels, all_uids = [], [], []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_scan, uid, p, max_neg): uid
                       for uid, p in subset_scans.items()}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"subset{subset}"):
                uid = futures[fut]
                try:
                    uid, patches, labels = fut.result()
                except Exception as exc:
                    print(f"\nFAILED {uid}: {exc}")
                    continue
                if len(patches):
                    all_patches.append(patches)
                    all_labels.append(labels)
                    all_uids.extend([uid] * len(labels))

        if not all_patches:
            print(f"subset{subset}: no candidates produced.")
            continue
        patches = np.concatenate(all_patches)
        labels = np.concatenate(all_labels)
        uids = np.array(all_uids)

        # hard selection (train/val only): keep positives + hard negatives
        if is_trainval:
            probs = score_patches(model, device, patches)
            is_pos = labels == 1
            is_hard_neg = (labels == 0) & (probs >= config.MINE_HARD_PROB)
            keep = is_pos | is_hard_neg
            patches, labels, uids = patches[keep], labels[keep], uids[keep]
            print(f"  kept {int(is_pos.sum())} positives + "
                  f"{int(is_hard_neg.sum())} hard negatives "
                  f"(of {len(keep)} candidates)")
        else:
            print(f"  kept all {len(labels)} candidates "
                  f"({int((labels == 1).sum())} positive)")

        np.save(out_npy, patches)
        for i in range(len(labels)):
            index_rows.append({"subset": subset, "idx": i,
                               "label": int(labels[i]), "seriesuid": uids[i]})
        pd.DataFrame(index_rows).to_csv(config.HARDNEG_INDEX_CSV, index=False)
        print(f"  saved {out_npy} {patches.shape}")

    total = pd.DataFrame(index_rows)
    print(f"\nDone. {len(total)} hard-negative-set patches "
          f"({int((total['label'] == 1).sum())} positive) across "
          f"{total['subset'].nunique()} subsets.")


if __name__ == "__main__":
    main()
