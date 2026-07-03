import argparse

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

import config

LUNG_THRESHOLD_HU = -320     # air/lung parenchyma is below, soft tissue above
MIN_LUNG_VOLUME_MM3 = 2e5    # discard air pockets smaller than 0.2 liters
CLOSING_RADIUS_MM = 10.0     # closes pleural indentations / includes nodules


def segment_lungs(volume, spacing):
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

def main():
    from preprocessing import find_scans, load_volume

    parser = argparse.ArgumentParser(description="Lung segmentation demo")
    parser.add_argument("--uid", type=str, default=None,
                        help="seriesuid (default: first scan found)")
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



if __name__ == "__main__":
    main()
