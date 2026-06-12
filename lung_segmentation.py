"""Automatic 3D lung segmentation (thesis objective 1).

Classical pipeline, no learning involved:

  1. threshold the volume at -320 HU -> binary "air-like" mask,
  2. 3D connected components; discard the background air that touches the
     image border and tiny components (airways, noise, bowel gas),
  3. keep the lung components (the large air regions inside the body),
  4. per-slice morphological closing with a spacing-aware disk so
     juxta-pleural nodules and vessels are included in the lung field.

The result is a boolean mask with the same (Z, Y, X) shape as the volume.
Optionally compares against the official LUNA16 masks (seg-lungs-LUNA16)
with the Dice coefficient.

"""
import argparse

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

import config

LUNG_THRESHOLD_HU = -320     # air/lung parenchyma is below, soft tissue above
MIN_LUNG_VOLUME_MM3 = 2e5    # discard air pockets smaller than 0.2 liters
CLOSING_RADIUS_MM = 10.0     # closes pleural indentations / includes nodules


def segment_lungs(volume, spacing):
    """Segment the lung field.

    Args:
        volume:  ndarray (Z, Y, X) in HU.
        spacing: ndarray (x, y, z) mm/voxel (SimpleITK order).

    Returns:
        boolean ndarray (Z, Y, X) lung mask.
    """
    binary = volume < LUNG_THRESHOLD_HU

    # Connected components in 3D.
    labels, _ = ndi.label(binary)

    # Background = everything connected to the volume's XY border (outside-
    # the-body air).  Collect the labels present on the four XY border planes.
    border_labels = np.unique(np.concatenate([
        labels[:, 0, :].ravel(), labels[:, -1, :].ravel(),
        labels[:, :, 0].ravel(), labels[:, :, -1].ravel(),
    ]))
    voxel_volume = float(np.prod(spacing))  # mm^3 per voxel
    min_voxels = int(MIN_LUNG_VOLUME_MM3 / voxel_volume)

    counts = np.bincount(labels.ravel())
    keep = np.zeros(len(counts), dtype=bool)
    keep[counts >= min_voxels] = True
    keep[border_labels] = False
    keep[0] = False
    lung_mask = keep[labels]

    # Spacing-aware 2D closing per axial slice (much faster than a 3D ball
    # and sufficient since slices are processed at full in-plane resolution).
    radius_px = max(1, int(round(CLOSING_RADIUS_MM / spacing[0])))
    footprint = morphology.disk(radius_px)
    closed = np.empty_like(lung_mask)
    for z in range(lung_mask.shape[0]):
        closed[z] = morphology.binary_closing(lung_mask[z], footprint)

    # Fill any remaining holes slice-wise (dense nodules inside the lung).
    for z in range(closed.shape[0]):
        closed[z] = ndi.binary_fill_holes(closed[z])
    return closed


def dice_coefficient(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    return 2.0 * inter / (mask_a.sum() + mask_b.sum() + 1e-9)


def load_official_mask(uid):
    """Load the LUNA16-provided lung mask for comparison (labels 3/4 = lungs)."""
    import SimpleITK as sitk
    path = config.DATA_DIR / "seg-lungs-LUNA16" / f"{uid}.mhd"
    if not path.exists():
        return None
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    return arr >= 3  # 3 = left lung, 4 = right lung (1/2 are trachea etc.)


def main():
    from preprocessing import find_scans, load_volume

    parser = argparse.ArgumentParser(description="Lung segmentation demo")
    parser.add_argument("--uid", type=str, default=None,
                        help="seriesuid (default: first scan found)")
    parser.add_argument("--dice", action="store_true",
                        help="compare with the official LUNA16 lung mask")
    args = parser.parse_args()

    scans = find_scans(config.DATA_DIR)
    uid = args.uid or sorted(scans)[0]
    mhd_path, _ = scans[uid]

    print(f"Segmenting {uid} ...")
    volume, _, spacing = load_volume(mhd_path)
    mask = segment_lungs(volume, spacing)

    voxel_volume = float(np.prod(spacing))
    print(f"Volume shape: {volume.shape}, spacing (x,y,z): {spacing}")
    print(f"Lung volume: {mask.sum() * voxel_volume / 1e6:.2f} liters")

    if args.dice:
        official = load_official_mask(uid)
        if official is None:
            print("No official mask found for this scan.")
        else:
            print(f"Dice vs official LUNA16 mask: "
                  f"{dice_coefficient(mask, official):.4f}")


if __name__ == "__main__":
    main()
