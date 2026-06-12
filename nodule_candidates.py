"""Nodule candidate generation based on shape and intensity (thesis objective 2).

Classical pipeline applied inside the segmented lung field:

  1. intensity threshold: nodules/vessels are denser than lung parenchyma,
  2. morphological opening (spacing-aware) to break thin vessel connections,
  3. Euclidean distance transform (in mm) + watershed with the distance
     peaks as markers -> separates nodules that touch vessels or the pleura,
  4. connected-component analysis: each blob becomes one candidate with
     position, bounding box, volume and shape/intensity features,
  5. size/shape filtering (nodules are 3-30 mm and roughly spherical).

Every candidate is a dict with:
    centroid_voxel (z, y, x), centroid_world (x, y, z) mm, slice z index,
    bbox (zlo, ylo, xlo, zhi, yhi, xhi), volume_mm3, equiv_diameter_mm,
    plus the feature vector used by the SVM / k-NN classifiers.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# Two threshold passes: -500 HU isolates solid nodules; -650 HU captures
# subsolid/ground-glass nodules (which never exceed ~-300 HU).  A single
# threshold cannot do both: too low and solid nodules merge with adjacent
# vessels into elongated blobs that the shape filter rejects.
DENSITY_THRESHOLDS_HU = (-500.0, -650.0)
MERGE_DISTANCE_MM = 5.0       # candidates closer than this are duplicates
OPENING_RADIUS_MM = 1.0       # breaks thin vessel bridges (in-plane only:
                              # opening along z would erase nodules that span
                              # only 2-3 slices on thick-slice scans)
MARKER_MIN_DISTANCE_MM = 4.0  # minimum separation between watershed markers
MIN_DIAMETER_MM = 2.5         # slightly below the 3 mm LUNA16 cutoff: the
                              # threshold captures only a nodule's dense core,
                              # which can be smaller than its true extent
MAX_DIAMETER_MM = 35.0
MAX_ELONGATION = 4.0          # vessels are long and thin; nodules are compact

FEATURE_NAMES = [
    "volume_mm3", "equiv_diameter_mm", "max_inscribed_radius_mm",
    "elongation", "extent", "compactness",
    "mean_hu", "std_hu", "max_hu", "min_hu",
]


def generate_candidates(volume, spacing, lung_mask):
    """Detect nodule candidates inside the lung mask.

    Runs the blob pipeline once per density threshold (solid + subsolid
    pass) and merges the results, dropping duplicates closer than
    MERGE_DISTANCE_MM (the higher-threshold pass wins).

    Args:
        volume:    ndarray (Z, Y, X) in HU.
        spacing:   ndarray (x, y, z) mm/voxel.
        lung_mask: boolean ndarray (Z, Y, X).

    Returns:
        list of candidate dicts (see module docstring).
    """
    spacing_zyx = spacing[::-1]
    merged, kept_positions = [], []
    for threshold in DENSITY_THRESHOLDS_HU:
        for cand in _candidates_at_threshold(volume, spacing, lung_mask,
                                             threshold):
            pos_mm = cand["centroid_voxel"] * spacing_zyx
            if kept_positions and (np.linalg.norm(
                    np.array(kept_positions) - pos_mm, axis=1)
                    < MERGE_DISTANCE_MM).any():
                continue
            merged.append(cand)
            kept_positions.append(pos_mm)
    return merged


def _candidates_at_threshold(volume, spacing, lung_mask, density_threshold):
    """One blob-detection pass at a fixed density threshold."""
    spacing_zyx = spacing[::-1]  # distance computations use array axis order
    voxel_volume = float(np.prod(spacing))

    # 1. dense structures inside the lungs
    dense = lung_mask & (volume > density_threshold)

    # 2. spacing-aware opening, in-plane only (z is left untouched)
    open_px = max(1, int(round(OPENING_RADIUS_MM / spacing[0])))
    structure = np.ones((1, 2 * open_px + 1, 2 * open_px + 1), dtype=bool)
    dense = ndi.binary_opening(dense, structure=structure)

    if not dense.any():
        return []

    # 3. distance transform (mm) + watershed
    distance = ndi.distance_transform_edt(dense, sampling=spacing_zyx)
    min_dist_px = np.maximum(1, np.round(MARKER_MIN_DISTANCE_MM /
                                         spacing_zyx)).astype(int)
    peaks = peak_local_max(distance, footprint=np.ones(2 * min_dist_px + 1),
                           labels=dense, exclude_border=False)
    if len(peaks) == 0:
        return []
    markers = np.zeros_like(distance, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    blobs = watershed(-distance, markers, mask=dense)

    # 4./5. per-blob features + filtering
    candidates = []
    for region in measure.regionprops(blobs, intensity_image=volume):
        vol_mm3 = region.num_pixels * voxel_volume
        equiv_diam = 2.0 * (3.0 * vol_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)
        if not (MIN_DIAMETER_MM <= equiv_diam <= MAX_DIAMETER_MM):
            continue

        # bbox in voxels and physical extents in mm
        zlo, ylo, xlo, zhi, yhi, xhi = region.bbox
        extent_mm = (np.array([zhi - zlo, yhi - ylo, xhi - xlo])
                     * spacing_zyx)
        elongation = float(extent_mm.max() / max(extent_mm.min(), 1e-3))
        if elongation > MAX_ELONGATION:
            continue  # long thin structure -> vessel

        # largest inscribed sphere radius (mm) from the distance map
        blob_slice = tuple(slice(lo, hi) for lo, hi in
                           zip((zlo, ylo, xlo), (zhi, yhi, xhi)))
        blob_mask = blobs[blob_slice] == region.label
        max_radius = float(distance[blob_slice][blob_mask].max())

        # compactness: blob volume vs the sphere spanned by its largest extent
        sphere_vol = (4.0 / 3.0) * np.pi * (extent_mm.max() / 2.0) ** 3
        compactness = float(vol_mm3 / max(sphere_vol, 1e-3))

        hu_values = volume[blob_slice][blob_mask]
        centroid = np.array(region.centroid)  # (z, y, x), float voxels

        candidates.append({
            "centroid_voxel": centroid,
            "slice_z": int(round(centroid[0])),
            "bbox": region.bbox,
            "volume_mm3": float(vol_mm3),
            "equiv_diameter_mm": float(equiv_diam),
            "features": {
                "volume_mm3": float(vol_mm3),
                "equiv_diameter_mm": float(equiv_diam),
                "max_inscribed_radius_mm": max_radius,
                "elongation": elongation,
                "extent": float(region.extent),
                "compactness": compactness,
                "mean_hu": float(hu_values.mean()),
                "std_hu": float(hu_values.std()),
                "max_hu": float(hu_values.max()),
                "min_hu": float(hu_values.min()),
            },
        })
    return candidates


def add_world_coordinates(candidates, origin, spacing):
    """Attach world (mm) coordinates: centroid_world = (x, y, z)."""
    for cand in candidates:
        voxel_xyz = cand["centroid_voxel"][::-1]  # (z,y,x) -> (x,y,z)
        cand["centroid_world"] = voxel_xyz * spacing + origin
    return candidates


def match_to_annotations(candidates, annotations, origin, spacing):
    """Label candidates against ground truth (for training / evaluation).

    A candidate is positive when its center lies within the annotated
    nodule radius (the official LUNA16 hit criterion).

    Args:
        annotations: DataFrame rows for this scan
                     (coordX, coordY, coordZ, diameter_mm).
    Returns:
        the candidates, each with an added "label" key.
    """
    centers = annotations[["coordX", "coordY", "coordZ"]].to_numpy(float) \
        if len(annotations) else np.zeros((0, 3))
    radii = annotations["diameter_mm"].to_numpy(float) / 2.0 \
        if len(annotations) else np.zeros(0)

    for cand in candidates:
        world = cand["centroid_world"]
        label = 0
        if len(centers):
            dists = np.linalg.norm(centers - world, axis=1)
            label = int((dists <= radii).any())
        cand["label"] = label
    return candidates
