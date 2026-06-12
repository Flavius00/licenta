"""3D visualization of the segmented lungs with detected nodules (PyVista).

Renders the lung surface (marching cubes on the segmentation mask, in
physical mm space) semi-transparent, with one red sphere per detection
sized by the nodule's equivalent diameter.

Reads detections from results/detections/<uid>/detections.csv, so run
detect.py for the scan first.

"""
import argparse

import numpy as np
import pandas as pd
import pyvista as pv
from skimage import measure

import config
from lung_segmentation import segment_lungs
from preprocessing import find_scans, load_volume


def lung_surface_mesh(lung_mask, spacing):
    """Marching-cubes surface of the lung mask in physical (mm) space."""
    spacing_zyx = spacing[::-1]
    verts, faces, _, _ = measure.marching_cubes(
        lung_mask.astype(np.uint8), level=0.5, spacing=spacing_zyx)
    faces_pv = np.column_stack(
        [np.full(len(faces), 3), faces]).ravel()
    return pv.PolyData(verts, faces_pv)


def main():
    parser = argparse.ArgumentParser(description="3D lung + nodule rendering")
    parser.add_argument("--uid", type=str, default=None)
    parser.add_argument("--screenshot", action="store_true",
                        help="render off-screen and save a PNG instead of "
                             "opening a window")
    args = parser.parse_args()

    scans = find_scans(config.DATA_DIR)
    uid = args.uid or sorted(scans)[0]
    mhd_path, _ = scans[uid]

    det_csv = config.RESULTS_DIR / "detections" / uid / "detections.csv"
    detections = pd.read_csv(det_csv) if det_csv.exists() else pd.DataFrame()
    if not det_csv.exists():
        print(f"No detections.csv for this scan - run detect.py --uid {uid} "
              f"first. Rendering lungs only.")

    print(f"Loading and segmenting {uid} ...")
    volume, origin, spacing = load_volume(mhd_path)
    lung_mask = segment_lungs(volume, spacing)
    mesh = lung_surface_mesh(lung_mask, spacing)

    plotter = pv.Plotter(off_screen=args.screenshot, window_size=(1200, 900))
    plotter.add_mesh(mesh, color="lightblue", opacity=0.30, smooth_shading=True,
                     label="lung field")

    # Detections live in world mm; the mesh lives in array-index mm
    # (z, y, x voxels * spacing).  Convert world -> array-mm.
    spacing_zyx = spacing[::-1]
    for _, det in detections.iterrows():
        world = np.array([det["coordX_mm"], det["coordY_mm"], det["coordZ_mm"]])
        voxel_xyz = (world - origin) / spacing
        pos_mm = voxel_xyz[::-1] * spacing_zyx  # (z, y, x) in mm
        radius = max(det["equiv_diameter_mm"] / 2.0, 2.0)
        sphere = pv.Sphere(radius=radius, center=pos_mm)
        plotter.add_mesh(sphere, color="red")
        plotter.add_point_labels(
            [pos_mm], [f"p={det['probability']:.2f}"],
            font_size=12, text_color="red", shape=None, always_visible=True)

    plotter.add_legend()
    plotter.add_text(f"{uid}\n{len(detections)} detections", font_size=10)
    plotter.camera_position = "xz"

    if args.screenshot:
        out_dir = config.RESULTS_DIR / "3d"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{uid}.png"
        plotter.screenshot(out_path)
        print(f"Saved {out_path}")
    else:
        plotter.show()


if __name__ == "__main__":
    main()
