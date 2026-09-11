"""Segment pallets from an RGB-D capture and export their 3D poses with SAM3."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
SAM3_ROOT = PROJECT_ROOT / "sam3"
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

CAMERA_INTRINSICS = {
    "fx": 1113.178,
    "fy": 1112.932,
    "cx": 660.693,
    "cy": 473.587,
    "width": 1280,
    "height": 960,
}
MASK_COLORS = [(255, 0, 0), (0, 200, 255), (0, 255, 80), (255, 165, 0), (180, 0, 255)]


def extract_masks_from_inference(inference_state: dict) -> np.ndarray:
    masks = inference_state.get("masks")
    if masks is None:
        raise ValueError("No masks found in inference_state")
    masks_np = masks.detach().float().cpu().numpy() if isinstance(masks, torch.Tensor) else np.asarray(masks)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        return masks_np[:, 0]
    if masks_np.ndim != 3:
        raise ValueError(f"Unexpected masks shape: {masks_np.shape}")
    return masks_np


def project_to_2d(points_3d: np.ndarray, intrinsics: dict) -> np.ndarray:
    points = np.asarray(points_3d)
    u = (points[:, 0] * intrinsics["fx"] / points[:, 2] + intrinsics["cx"]).astype(int)
    v = (points[:, 1] * intrinsics["fy"] / points[:, 2] + intrinsics["cy"]).astype(int)
    return np.stack([u, v], axis=1)


def extract_corner_coordinates(binary_mask: np.ndarray, epsilon_factor: float) -> np.ndarray:
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return np.empty((0, 2), dtype=int)
    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, epsilon_factor * cv2.arcLength(contour, True), True)
    return approx.reshape(-1, 2)[:, ::-1]


def pixel_to_3d(pixel_coords: np.ndarray, depth_map: np.ndarray, intrinsics: dict) -> np.ndarray:
    points_3d = []
    for y, x in pixel_coords:
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            z = depth_map[y, x]
            if z > 0:
                points_3d.append([(x - intrinsics["cx"]) * z / intrinsics["fx"], (y - intrinsics["cy"]) * z / intrinsics["fy"], z])
    return np.asarray(points_3d).reshape(-1, 3)


def build_corners_from_pose(center: np.ndarray, rotation: np.ndarray, extent: np.ndarray) -> np.ndarray:
    half = 0.5 * np.asarray(extent, dtype=np.float64)
    local_corners = np.array(
        [[-half[0], -half[1], -half[2]], [half[0], -half[1], -half[2]],
         [half[0], half[1], -half[2]], [-half[0], half[1], -half[2]],
         [-half[0], -half[1], half[2]], [half[0], -half[1], half[2]],
         [half[0], half[1], half[2]], [-half[0], half[1], half[2]]],
        dtype=np.float64,
    )
    return local_corners @ np.asarray(rotation, dtype=np.float64).T + np.asarray(center, dtype=np.float64)


def obb_to_pose(obb: o3d.geometry.OrientedBoundingBox) -> dict:
    extent = np.asarray(obb.extent, dtype=np.float64)
    return {
        "center": np.asarray(obb.center, dtype=np.float64),
        "rotation": np.asarray(obb.R, dtype=np.float64),
        "extent": extent,
        "corners_3d": np.asarray(obb.get_box_points(), dtype=np.float64),
        "volume_liters": float(np.prod(extent) / 1e6),
    }


def normalize(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    return vector if length < 1e-9 else vector / length


def canonicalize_pallet_pose(pose: dict) -> dict:
    rotation = np.asarray(pose["rotation"], dtype=np.float64)
    extent = np.asarray(pose["extent"], dtype=np.float64)
    z_idx, y_idx, x_idx = (int(index) for index in np.argsort(extent))
    x_axis, y_axis, z_axis = (normalize(rotation[:, index].copy()) for index in (x_idx, y_idx, z_idx))

    up_hint = np.array([0.0, -1.0, 0.0])
    if np.dot(z_axis, up_hint) < 0.0:
        z_axis = -z_axis
    x_ref = normalize(np.array([1.0, 0.0, 0.0]) - np.dot([1.0, 0.0, 0.0], z_axis) * z_axis)
    if np.dot(x_axis, x_ref) < 0.0:
        x_axis = -x_axis
    y_axis = normalize(np.cross(z_axis, x_axis))
    if np.linalg.norm(y_axis) < 1e-9:
        y_axis = normalize(rotation[:, y_idx].copy())
    x_axis = normalize(np.cross(y_axis, z_axis))
    z_axis = normalize(np.cross(x_axis, y_axis))

    pose["rotation"] = np.stack([x_axis, y_axis, z_axis], axis=1)
    pose["extent"] = np.array([extent[x_idx], extent[y_idx], extent[z_idx]], dtype=np.float64)
    pose["corners_3d"] = build_corners_from_pose(pose["center"], pose["rotation"], pose["extent"])
    return pose


def project_points_to_image(points_3d: np.ndarray, intrinsics: dict) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_3d, dtype=np.float64)
    valid = points[:, 2] > 1e-6
    uv = np.full((len(points), 2), -1, dtype=np.int32)
    if np.any(valid):
        uv[valid] = project_to_2d(points[valid], intrinsics)
    return uv, valid


def select_origin_corner(corners_3d: np.ndarray, tolerance_mm: float = 5.0) -> np.ndarray:
    corners = np.asarray(corners_3d, dtype=np.float64)
    indices = np.arange(len(corners))
    for axis, use_max in ((0, True), (1, False), (2, False)):
        values = corners[indices, axis]
        threshold = values.max() - tolerance_mm if use_max else values.min() + tolerance_mm
        indices = indices[values >= threshold] if use_max else indices[values <= threshold]
        if len(indices) == 1:
            break
    return corners[indices[0]]


def save_points_cloud(points_xyz_mm: np.ndarray, colors_rgb: np.ndarray, output_path: Path) -> bool:
    points = np.asarray(points_xyz_mm, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(colors_rgb, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return False
    if colors.max() > 1.0:
        colors /= 255.0
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return o3d.io.write_point_cloud(str(output_path), cloud)


def build_pose_axes_points(pose: dict) -> dict:
    origin = np.asarray(pose["origin_point"], dtype=np.float64)
    axes = np.asarray(pose["rotation"], dtype=np.float64) * pose["axis_length_mm"]
    return {"origin": origin, "x_tip": origin + axes[:, 0], "y_tip": origin + axes[:, 1], "z_tip": origin + axes[:, 2]}


def draw_pose_axes_on_image(image_bgr: np.ndarray, axes_3d: dict, intrinsics: dict) -> np.ndarray:
    points = np.stack([axes_3d["origin"], axes_3d["x_tip"], axes_3d["y_tip"], axes_3d["z_tip"]])
    uv, valid = project_points_to_image(points, intrinsics)
    height, width = image_bgr.shape[:2]
    if not valid[0] or not (0 <= uv[0, 0] < width and 0 <= uv[0, 1] < height):
        return image_bgr
    origin = tuple(uv[0].tolist())
    for index, color in ((1, (0, 0, 255)), (2, (0, 255, 0)), (3, (255, 0, 0))):
        if valid[index]:
            visible, start, end = cv2.clipLine((0, 0, width, height), origin, tuple(uv[index].tolist()))
            if visible:
                cv2.arrowedLine(image_bgr, start, end, color, 3, tipLength=0.2)
    return image_bgr


def display(image: np.ndarray, title: str, show: bool) -> None:
    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data/picaulo/capture4")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/picaulo")
    parser.add_argument("--prompt", default="pallet")
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    parser.add_argument("--corner-epsilon-factor", type=float, default=0.02)
    parser.add_argument("--show", action="store_true", help="Display the notebook-equivalent visualizations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.input_dir / "rgb.png"
    depth_map_path = args.input_dir / "depth_mm.npy"
    bpe_path = SAM3_ROOT / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    checkpoint_path = PROJECT_ROOT / "sam3.pt"
    output_dir = args.output_dir
    for path, name in ((args.input_dir, "Input folder"), (image_path, "RGB image"), (depth_map_path, "Depth map"), (bpe_path, "BPE tokenizer"), (checkpoint_path, "SAM3 checkpoint")):
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable GPU is required by this SAM3 pipeline.")
    torch.cuda.set_device(0)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading local SAM3 checkpoint: {checkpoint_path}")
    image_model = build_sam3_image_model(bpe_path=str(bpe_path), checkpoint_path=str(checkpoint_path), load_from_HF=False)
    image_processor = Sam3Processor(image_model, confidence_threshold=args.confidence_threshold)

    image = Image.open(image_path).convert("RGB")
    depth_mm = np.load(depth_map_path)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        inference_state = image_processor.set_image(image)
        image_processor.reset_all_prompts(inference_state)
        inference_state = image_processor.set_text_prompt(prompt=args.prompt, state=inference_state)
    plot_results(image, inference_state)
    detection_path = output_dir / "sam3_detections.png"
    plt.gcf().savefig(detection_path, dpi=100, bbox_inches="tight")
    if not args.show:
        plt.close()

    masks = extract_masks_from_inference(inference_state)
    binary_masks = []
    for mask in masks:
        mask_image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
        binary_masks.append(np.asarray(mask_image.resize(image.size, Image.Resampling.NEAREST)) > 0)
    overlay = np.asarray(image).copy()
    for index, mask in enumerate(binary_masks):
        overlay[mask] = (overlay[mask] * 0.55 + np.array(MASK_COLORS[index % len(MASK_COLORS)]) * 0.45).astype(np.uint8)
    mask_overlay_path = output_dir / "sam3_masks_overlay.png"
    Image.fromarray(overlay).save(mask_overlay_path)
    display(overlay, f"SAM3 masks: '{args.prompt}' ({len(binary_masks)} object(s))", args.show)

    obbs, poses = [], []
    for index, mask in enumerate(binary_masks):
        ys, xs = np.where(mask)
        depth = depth_mm[ys, xs].astype(float)
        valid = depth > 0
        ys, xs, depth = ys[valid], xs[valid], depth[valid]
        if len(depth) < 100:
            continue
        points_3d = np.stack([(xs - CAMERA_INTRINSICS["cx"]) * depth / CAMERA_INTRINSICS["fx"], (ys - CAMERA_INTRINSICS["cy"]) * depth / CAMERA_INTRINSICS["fy"], depth], axis=1)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points_3d)
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if len(cloud.points) < 100:
            continue
        cloud_path = output_dir / f"{args.prompt.replace(' ', '_')}_segment_{index:02d}.ply"
        o3d.io.write_point_cloud(str(cloud_path), cloud)
        obb = cloud.get_minimal_oriented_bounding_box(robust=True)
        pose = canonicalize_pallet_pose(obb_to_pose(obb))
        corners_3d = pixel_to_3d(extract_corner_coordinates(mask, args.corner_epsilon_factor), depth_mm, CAMERA_INTRINSICS)
        if len(corners_3d) != 4:
            continue
        corner_uvs = project_to_2d(corners_3d, CAMERA_INTRINSICS)
        if not all(0 <= u < CAMERA_INTRINSICS["width"] and 0 <= v < CAMERA_INTRINSICS["height"] for u, v in corner_uvs):
            continue
        pose["corners_3d"] = corners_3d
        pose["axis_length_mm"] = float(np.clip(0.18 * np.max(pose["extent"]), 80.0, 220.0))
        pose["object_index"] = index
        pose["point_cloud_path"] = str(cloud_path)
        pose["origin_point"] = select_origin_corner(corners_3d)
        pose["origin_type"] = "selected_origin_corner"
        pose["corners_uv"], pose["corners_valid"] = project_points_to_image(corners_3d, CAMERA_INTRINSICS)
        pose["selected_corner_idx"] = int(np.argmin(np.linalg.norm(corners_3d - pose["origin_point"], axis=1)))
        origin_index = pose["selected_corner_idx"]
        edge1 = corners_3d[(origin_index + 1) % 4] - pose["origin_point"]
        edge2 = corners_3d[(origin_index - 1) % 4] - pose["origin_point"]
        if np.linalg.norm(edge1) >= np.linalg.norm(edge2):
            x_axis, y_axis = normalize(edge1), normalize(edge2)
        else:
            x_axis, y_axis = normalize(edge2), normalize(edge1)
        z_axis = normalize(np.cross(x_axis, y_axis))
        if z_axis[2] > 0:
            z_axis = -z_axis
        y_axis = normalize(np.cross(z_axis, x_axis))
        pose["rotation"] = np.stack([x_axis, y_axis, z_axis], axis=1)
        pose_points_path = output_dir / f"{args.prompt.replace(' ', '_')}_pose_points_{index:02d}.ply"
        save_points_cloud(
            np.vstack([corners_3d, pose["center"], pose["origin_point"], corners_3d[origin_index]]),
            np.vstack([np.tile([[220, 220, 220]], (4, 1)), [[0, 255, 0]], [[0, 0, 255]], [[255, 0, 0]]]),
            pose_points_path,
        )
        pose["pose_points_path"] = str(pose_points_path)
        mesh_path = output_dir / f"{args.prompt.replace(' ', '_')}_obb_mesh_{index:02d}.ply"
        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obb)
        mesh.paint_uniform_color([1.0, 1.0, 1.0])
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        pose["mesh_path"] = str(mesh_path)
        obbs.append(obb)
        poses.append(pose)

    corner_image_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    pose_image_bgr = corner_image_bgr.copy()
    for pose in poses:
        color = [int(value) for value in np.array(MASK_COLORS[pose["object_index"] % len(MASK_COLORS)])[::-1]]
        for corner_index, (uv, visible) in enumerate(zip(pose["corners_uv"], pose["corners_valid"])):
            if visible:
                cv2.circle(corner_image_bgr, tuple(uv), 6, color, -1)
                cv2.putText(corner_image_bgr, str(corner_index), tuple(uv + [8, -6]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        origin_uv, origin_valid = project_points_to_image(pose["origin_point"][None, :], CAMERA_INTRINSICS)
        if origin_valid[0]:
            cv2.circle(corner_image_bgr, tuple(origin_uv[0]), 10, (255, 255, 255), -1)
            cv2.circle(corner_image_bgr, tuple(origin_uv[0]), 10, color, 2)
        pose_image_bgr = draw_pose_axes_on_image(pose_image_bgr, build_pose_axes_points(pose), CAMERA_INTRINSICS)

    obb_image_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    for index, obb in enumerate(obbs):
        box_uv, box_valid = project_points_to_image(np.asarray(obb.get_box_points()), CAMERA_INTRINSICS)
        box_points = box_uv[box_valid].reshape(-1, 1, 2).astype(np.int32)
        if len(box_points) >= 3:
            color = [int(value) for value in np.array(MASK_COLORS[index % len(MASK_COLORS)])[::-1]]
            cv2.drawContours(obb_image_bgr, [cv2.convexHull(box_points)], 0, color, 2)

    corner_path = output_dir / f"{args.prompt.replace(' ', '_')}_corners_rgb.png"
    pose_path = output_dir / f"{args.prompt.replace(' ', '_')}_pose_axes_rgb.png"
    cv2.imwrite(str(corner_path), corner_image_bgr)
    cv2.imwrite(str(pose_path), pose_image_bgr)
    display(cv2.cvtColor(corner_image_bgr, cv2.COLOR_BGR2RGB), f"Contour corners + selected origin ({len(poses)} object(s))", args.show)
    display(cv2.cvtColor(obb_image_bgr, cv2.COLOR_BGR2RGB), f"3D OBB projected footprint ({len(obbs)} object(s))", args.show)
    display(cv2.cvtColor(pose_image_bgr, cv2.COLOR_BGR2RGB), f"Pose axes overlay ({len(poses)} object(s))", args.show)

    poses_json = []
    for pose in poses:
        corners_3d, corners_uv = np.asarray(pose["corners_3d"]), np.asarray(pose["corners_uv"])
        selected = pose["selected_corner_idx"]
        poses_json.append({
            "object_index": pose["object_index"], "frame": "camera",
            "translation_center_mm": pose["center"].tolist(), "translation_origin_mm": pose["origin_point"].tolist(),
            "rotation_cam_from_obj": pose["rotation"].tolist(), "extent_mm": pose["extent"].tolist(),
            "volume_liters": pose["volume_liters"], "axis_length_mm": pose["axis_length_mm"], "origin_type": pose["origin_type"],
            "selected_corner_idx": selected, "selected_corner_3d_mm": corners_3d[selected].tolist(),
            "selected_corner_uv_px": corners_uv[selected].tolist(), "corners_3d_mm": corners_3d.tolist(),
            "corners_uv_px": corners_uv.tolist(), "corners_visible": pose["corners_valid"].tolist(),
            "pose_points_path": pose["pose_points_path"], "point_cloud_path": pose["point_cloud_path"], "mesh_path": pose["mesh_path"],
        })
    json_path = output_dir / f"{args.prompt.replace(' ', '_')}_poses.json"
    json_path.write_text(json.dumps({"camera_intrinsics": CAMERA_INTRINSICS, "prompt": args.prompt, "num_objects": len(poses_json), "poses": poses_json}, indent=2), encoding="utf-8")
    print(f"Fitted {len(obbs)} OBB(s); saved pose JSON: {json_path}")


if __name__ == "__main__":
    main()