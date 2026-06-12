"""End-to-end nodule detection on a CT scan.

Pipeline:  CT volume -> lung segmentation -> classical candidate generation
(threshold + morphology + watershed) -> 2D CNN scores every candidate ->
detections above the probability threshold are reported.

For every detection the script returns / saves:
  * world coordinates [x, y, z] in mm and voxel coordinates,
  * the axial slice index where the nodule center lies,
  * the bounding box of the nodule (from the segmented blob),
  * volume (mm^3) and equivalent diameter (mm),
  * the CNN probability,
  * a PNG of the FULL axial slice (whole lung field visible) taken at the
    nodule's center slice, with the bounding box drawn around the nodule.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch

import config
from lung_segmentation import segment_lungs
from model import NoduleCNN
from nodule_candidates import add_world_coordinates, generate_candidates
from preprocessing import extract_patch, find_scans, load_volume


def load_model(checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = NoduleCNN().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def score_candidates(model, volume, spacing, candidates, device,
                     batch_size=256):
    """Run the CNN on a 2D patch around every candidate center."""
    patches, valid = [], []
    for i, cand in enumerate(candidates):
        voxel_xyz = cand["centroid_voxel"][::-1]  # (z,y,x) -> (x,y,z)
        patch = extract_patch(volume, spacing, voxel_xyz)
        if patch is None:
            cand["probability"] = 0.0
            continue
        patches.append(np.asarray(patch, dtype=np.float32))
        valid.append(i)

    for start in range(0, len(patches), batch_size):
        batch = torch.from_numpy(np.stack(patches[start:start + batch_size]))
        batch = batch.unsqueeze(1).to(device)  # (B, 1, H, W)
        with torch.no_grad():
            probs = torch.sigmoid(model(batch).float()).cpu().numpy()
        for j, p in enumerate(probs):
            candidates[valid[start + j]]["probability"] = float(p)
    return candidates


def save_detection_slice(volume, detection, out_path, ground_truth=None):
    """Save the full axial slice at the detection's center z with the
    nodule bounding box drawn (whole lung field visible)."""
    z = detection["slice_z"]
    axial = np.clip(volume[z], config.HU_MIN, config.HU_MAX)

    zlo, ylo, xlo, zhi, yhi, xhi = detection["bbox"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(axial, cmap="gray")
    ax.add_patch(mpatches.Rectangle((xlo, ylo), xhi - xlo, yhi - ylo,
                                    fill=False, edgecolor="red", linewidth=1.5))
    label = (f"p={detection['probability']:.2f}  "
             f"d={detection['equiv_diameter_mm']:.1f}mm")
    ax.text(xlo, max(ylo - 6, 2), label, color="red", fontsize=9)

    if ground_truth is not None:
        for _, gt in ground_truth.iterrows():
            gz = int(round(gt["voxel_z"]))
            if abs(gz - z) <= max(1, int(gt["radius_px_z"])):
                r_y, r_x = gt["radius_px_y"], gt["radius_px_x"]
                ax.add_patch(mpatches.Rectangle(
                    (gt["voxel_x"] - r_x, gt["voxel_y"] - r_y),
                    2 * r_x, 2 * r_y, fill=False, edgecolor="lime",
                    linewidth=1.2, linestyle="--"))

    world = detection["centroid_world"]
    ax.set_title(f"slice z={z}   world=({world[0]:.1f}, {world[1]:.1f}, "
                 f"{world[2]:.1f}) mm")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def ground_truth_table(uid, origin, spacing):
    """Annotation rows for this scan converted to voxel space (for overlay)."""
    ann = pd.read_csv(config.ANNOTATIONS_CSV)
    ann = ann[ann["seriesuid"] == uid].copy()
    if not len(ann):
        return ann
    world = ann[["coordX", "coordY", "coordZ"]].to_numpy(float)
    voxel = (world - origin) / spacing  # (x, y, z)
    ann["voxel_x"], ann["voxel_y"], ann["voxel_z"] = voxel.T
    for axis, name in enumerate(["x", "y", "z"]):
        ann[f"radius_px_{name}"] = ann["diameter_mm"] / 2.0 / spacing[axis]
    return ann


def detect_scan(mhd_path, uid, model, device, threshold):
    print(f"Loading {uid} ...")
    volume, origin, spacing = load_volume(mhd_path)

    print("Segmenting lungs ...")
    lung_mask = segment_lungs(volume, spacing)

    print("Generating candidates (threshold + watershed) ...")
    candidates = generate_candidates(volume, spacing, lung_mask)
    candidates = add_world_coordinates(candidates, origin, spacing)
    print(f"  {len(candidates)} candidates")

    print("Scoring candidates with the CNN ...")
    candidates = score_candidates(model, volume, spacing, candidates, device)

    detections = [c for c in candidates if c.get("probability", 0) >= threshold]
    detections.sort(key=lambda c: c["probability"], reverse=True)
    return volume, origin, spacing, candidates, detections


def main():
    parser = argparse.ArgumentParser(description="Detect nodules in a CT scan")
    parser.add_argument("--uid", type=str, default=None, help="LUNA16 seriesuid")
    parser.add_argument("--mhd", type=str, default=None, help="path to a .mhd file")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="CNN probability threshold")
    parser.add_argument("--checkpoint", type=str,
                        default=str(config.CHECKPOINT_DIR / "best.pt"))
    parser.add_argument("--gt", action="store_true",
                        help="overlay ground-truth boxes (green, dashed)")
    args = parser.parse_args()

    if args.mhd:
        from pathlib import Path
        mhd_path = Path(args.mhd)
        uid = mhd_path.stem
    else:
        scans = find_scans(config.DATA_DIR)
        uid = args.uid or sorted(scans)[0]
        mhd_path, _ = scans[uid]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    volume, origin, spacing, candidates, detections = detect_scan(
        mhd_path, uid, model, device, args.threshold)

    out_dir = config.RESULTS_DIR / "detections" / uid
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = ground_truth_table(uid, origin, spacing) if args.gt else None

    rows = []
    for i, det in enumerate(detections, start=1):
        world, voxel = det["centroid_world"], det["centroid_voxel"]
        rows.append({
            "rank": i,
            "coordX_mm": round(world[0], 2),
            "coordY_mm": round(world[1], 2),
            "coordZ_mm": round(world[2], 2),
            "slice_z": det["slice_z"],
            "voxel_y": round(voxel[1], 1),
            "voxel_x": round(voxel[2], 1),
            "bbox_zlo,ylo,xlo,zhi,yhi,xhi": str(det["bbox"]),
            "equiv_diameter_mm": round(det["equiv_diameter_mm"], 2),
            "volume_mm3": round(det["volume_mm3"], 1),
            "probability": round(det["probability"], 4),
        })
        save_detection_slice(volume, det,
                             out_dir / f"detection_{i:02d}_slice{det['slice_z']}.png",
                             ground_truth=gt)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "detections.csv", index=False)

    print(f"\n{len(detections)} detections (of {len(candidates)} candidates) "
          f"at threshold {args.threshold}:")
    if len(df):
        print(df.to_string(index=False))
    print(f"\nSlice images + detections.csv saved to {out_dir}")


if __name__ == "__main__":
    main()
