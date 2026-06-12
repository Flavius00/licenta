"""LUNA16 preprocessing: break every CT scan into 2D axial slice patches.

Instead of saving each CT volume as a single big .npy file 
 , every scan is decomposed into small 2D patches:

  * one 64x64 patch per candidate location from candidates_V2.csv,
  * each patch covers a fixed 50x50 mm physical area (spacing aware), so
    patches are comparable across scans with different voxel spacings,
  * positive candidates additionally contribute the neighbouring axial
    slices (z-1, z, z+1) to multiply the rare positive class,
  * negatives are subsampled per scan for train/val; the test subset keeps
    all negatives so the final evaluation is realistic.

Patches are normalised (HU clipped to [-1000, 400] then scaled to [0, 1]),
stored as one stacked float16 array per subset
(`patches/subset{i}_patches.npy`, shape [N, 64, 64]) plus a global
`patches/index.csv` with one row per patch:

    subset, idx, label, seriesuid

Run with:
    python preprocessing.py
"""
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config


def find_scans(data_dir):
    """Map seriesuid -> (mhd path, subset index) by scanning subset folders."""
    uid_to_scan = {}
    for subset_idx in range(10):
        subset_dir = data_dir / f"subset{subset_idx}"
        if not subset_dir.exists():
            continue
        for mhd in subset_dir.glob("*.mhd"):
            uid_to_scan[mhd.stem] = (mhd, subset_idx)
    return uid_to_scan


def load_volume(mhd_path):
    """Read a MetaImage volume.

    Returns:
        volume:  ndarray (Z, Y, X) of HU values
        origin:  ndarray (x, y, z) in mm
        spacing: ndarray (x, y, z) in mm/voxel
    """
    image = sitk.ReadImage(str(mhd_path))
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    origin = np.array(image.GetOrigin(), dtype=np.float64)
    spacing = np.array(image.GetSpacing(), dtype=np.float64)
    return volume, origin, spacing


def world_to_voxel(coord_xyz, origin, spacing):
    """Convert world (mm) coordinates to continuous voxel indices (x, y, z)."""
    return (np.asarray(coord_xyz, dtype=np.float64) - origin) / spacing


def extract_patch(volume, spacing, voxel_xyz, z_offset=0):
    """Extract one axial patch around a candidate location.

    The patch covers PATCH_SIZE_MM x PATCH_SIZE_MM in physical space and is
    resampled to PATCH_SIZE_PX x PATCH_SIZE_PX pixels.  Out-of-bounds regions
    are padded with air (HU_MIN).

    Returns a float16 array of shape (PATCH_SIZE_PX, PATCH_SIZE_PX) with
    values in [0, 1], or None if the requested slice is outside the volume.
    """
    num_z = volume.shape[0]
    z_idx = int(round(voxel_xyz[2])) + z_offset
    if z_idx < 0 or z_idx >= num_z:
        return None

    axial = volume[z_idx]  # (Y, X)

    # Half patch size in voxels for each in-plane axis (spacing is x, y, z).
    half_x = config.PATCH_SIZE_MM / 2.0 / spacing[0]
    half_y = config.PATCH_SIZE_MM / 2.0 / spacing[1]
    cx, cy = voxel_xyz[0], voxel_xyz[1]

    x_lo, x_hi = int(round(cx - half_x)), int(round(cx + half_x))
    y_lo, y_hi = int(round(cy - half_y)), int(round(cy + half_y))

    # Crop with air padding for parts that fall outside the slice.
    crop = np.full((y_hi - y_lo, x_hi - x_lo), config.HU_MIN, dtype=np.float32)
    src_y_lo, src_y_hi = max(y_lo, 0), min(y_hi, axial.shape[0])
    src_x_lo, src_x_hi = max(x_lo, 0), min(x_hi, axial.shape[1])
    if src_y_lo >= src_y_hi or src_x_lo >= src_x_hi:
        return None
    crop[src_y_lo - y_lo:src_y_hi - y_lo, src_x_lo - x_lo:src_x_hi - x_lo] = \
        axial[src_y_lo:src_y_hi, src_x_lo:src_x_hi]

    # Resample to the fixed pixel size (bilinear).
    tensor = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(config.PATCH_SIZE_PX, config.PATCH_SIZE_PX),
                           mode="bilinear", align_corners=False)
    patch = tensor.squeeze().numpy()

    # HU windowing + normalisation to [0, 1].
    patch = np.clip(patch, config.HU_MIN, config.HU_MAX)
    patch = (patch - config.HU_MIN) / (config.HU_MAX - config.HU_MIN)
    return patch.astype(np.float16)


def select_candidates(scan_cands, subset_idx, rng):
    """Pick which candidates of one scan to extract.

    All positives are kept; negatives are subsampled for train/val subsets.
    Returns a DataFrame.
    """
    pos = scan_cands[scan_cands["class"] == 1]
    neg = scan_cands[scan_cands["class"] == 0]
    if subset_idx not in config.KEEP_ALL_NEG_SUBSETS and len(neg) > config.MAX_NEG_PER_SCAN:
        neg = neg.iloc[rng.choice(len(neg), config.MAX_NEG_PER_SCAN, replace=False)]
    return pd.concat([pos, neg])


def main():
    config.PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.PREPROCESS_SEED)

    print(f"Reading candidates from {config.CANDIDATES_CSV} ...")
    candidates = pd.read_csv(config.CANDIDATES_CSV)
    print(f"  {len(candidates)} candidates "
          f"({int((candidates['class'] == 1).sum())} positive)")

    uid_to_scan = find_scans(config.DATA_DIR)
    print(f"Found {len(uid_to_scan)} scans on disk.")

    cand_by_uid = dict(tuple(candidates.groupby("seriesuid")))

    # Patches are accumulated per subset and saved as one stacked array each.
    subset_patches = {i: [] for i in range(10)}
    index_rows = []
    missing_scans = 0

    for uid, (mhd_path, subset_idx) in tqdm(sorted(uid_to_scan.items()),
                                            desc="Processing scans"):
        scan_cands = cand_by_uid.get(uid)
        if scan_cands is None:
            missing_scans += 1
            continue

        volume, origin, spacing = load_volume(mhd_path)
        selected = select_candidates(scan_cands, subset_idx, rng)

        for _, row in selected.iterrows():
            label = int(row["class"])
            voxel = world_to_voxel([row["coordX"], row["coordY"], row["coordZ"]],
                                   origin, spacing)
            offsets = config.POSITIVE_SLICE_OFFSETS if label == 1 else [0]
            for off in offsets:
                patch = extract_patch(volume, spacing, voxel, z_offset=off)
                if patch is None:
                    continue
                idx = len(subset_patches[subset_idx])
                subset_patches[subset_idx].append(patch)
                index_rows.append({"subset": subset_idx, "idx": idx,
                                   "label": label, "seriesuid": uid})

    if missing_scans:
        print(f"WARNING: {missing_scans} scans had no candidate rows.")

    print("Saving per-subset patch arrays ...")
    for subset_idx, patches in subset_patches.items():
        if not patches:
            continue
        arr = np.stack(patches)
        out = config.PATCHES_DIR / f"subset{subset_idx}_patches.npy"
        np.save(out, arr)
        print(f"  subset{subset_idx}: {arr.shape} -> {out}")

    index = pd.DataFrame(index_rows)
    index.to_csv(config.INDEX_CSV, index=False)
    print(f"Index written to {config.INDEX_CSV} ({len(index)} patches, "
          f"{int((index['label'] == 1).sum())} positive).")


if __name__ == "__main__":
    main()
