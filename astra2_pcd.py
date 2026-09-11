#!/usr/bin/env python3
"""
Grab a single aligned RGB-D frame from an Orbbec Astra 2 and save it to disk.

Saves:
    <out>/rgb.png         - color image (BGR, for cv2.imwrite)
    <out>/depth.npy        - raw uint16 depth (in sensor units, NOT mm)
    <out>/depth_mm.npy      - depth converted to mm (float32), 0 = invalid
    <out>/depth_vis.png     - normalized depth visualization (for quick viewing)

Prints color-camera intrinsics (fx, fy, cx, cy) used to align depth->color.

Usage:
    python astra2_rgbd_capture.py
    python astra2_rgbd_capture.py --out capture01 --warmup 15
    python astra2_rgbd_capture.py --frames 5          # save frame_000, frame_001, ...
"""

import os
import argparse
import numpy as np
import cv2
import open3d as o3d

from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBFormat, OBStreamType, AlignFilter,
)

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="rgbd_out")
parser.add_argument("--warmup", type=int, default=10,
                    help="frames to discard before capture (let AE/AWB settle)")
parser.add_argument("--frames", type=int, default=1,
                    help="number of frames to capture/save")
parser.add_argument("--color-res", default="1280x960x30",
                    help="WxHxFPS for color stream")
parser.add_argument("--depth-res", default="1600x1200x30",
                    help="WxHxFPS for depth stream (Y16)")
parser.add_argument("--no-pcd", action="store_true",
                    help="skip point cloud generation/saving")
parser.add_argument("--max-depth-mm", type=float, default=0,
                    help="drop points beyond this depth in mm (0 = no limit)")
parser.add_argument("--voxel", type=float, default=0,
                    help="voxel-downsample size in mm (0 = no downsampling)")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)


def parse_res(s):
    w, h, fps = s.lower().split("x")
    return int(w), int(h), int(fps)


CW, CH, CFPS = parse_res(args.color_res)
DW, DH, DFPS = parse_res(args.depth_res)


def pick(profiles, w, h, fps, fmt):
    for i in range(profiles.get_count()):
        p = profiles.get_stream_profile_by_index(i).as_video_stream_profile()
        if (p.get_width(), p.get_height(), p.get_fps(), p.get_format()) == (w, h, fps, fmt):
            return p
    # helpful error: list what's actually available
    avail = []
    for i in range(profiles.get_count()):
        p = profiles.get_stream_profile_by_index(i).as_video_stream_profile()
        avail.append((p.get_width(), p.get_height(), p.get_fps(), p.get_format()))
    raise RuntimeError(
        f"No {w}x{h}@{fps} {fmt} profile.\nAvailable:\n" +
        "\n".join(str(a) for a in avail)
    )


def depth_to_pcd(depth_mm, rgb, fx, fy, cx, cy, max_depth_mm=0):
    """Backproject the full aligned depth map to a colored point cloud.

    depth_mm: (H,W) float32, 0 = invalid
    rgb: (H,W,3) uint8, RGB order, already aligned to the depth (same frame)
    """
    h, w = depth_mm.shape
    ys, xs = np.where(depth_mm > 0)
    z = depth_mm[ys, xs]
    if max_depth_mm > 0:
        keep = z <= max_depth_mm
        xs, ys, z = xs[keep], ys[keep], z[keep]
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    pts = np.stack([X, Y, z], axis=1).astype(np.float64)
    cols = rgb[ys, xs].astype(np.float64) / 255.0
    return pts, cols


def grab(pipeline, align, timeout_ms=200, retries=30):
    for _ in range(retries):
        frames = pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            continue
        frames = align.process(frames)
        if frames is None:
            continue
        frames = frames.as_frame_set()
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if color is None or depth is None:
            continue

        cw, ch = color.get_width(), color.get_height()
        rgb = np.frombuffer(color.get_data(), np.uint8).reshape(ch, cw, 3)

        dw, dh = depth.get_width(), depth.get_height()
        draw = np.frombuffer(depth.get_data(), np.uint16)
        if draw.size != dw * dh:
            continue
        draw = draw.reshape(dh, dw)

        return rgb.copy(), draw.copy(), depth.get_depth_scale()
    return None


def main():
    pipeline = Pipeline()
    config = Config()
    dp = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    cp = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)

    config.enable_stream(pick(dp, DW, DH, DFPS, OBFormat.Y16))
    config.enable_stream(pick(cp, CW, CH, CFPS, OBFormat.RGB))
    align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    pipeline.start(config)

    intr = pipeline.get_camera_param().rgb_intrinsic
    print(f"[intr] fx={intr.fx:.3f} fy={intr.fy:.3f} "
          f"cx={intr.cx:.3f} cy={intr.cy:.3f} (calib {intr.width}x{intr.height})")

    print(f"[cam] warming up ({args.warmup} frames)...")
    for _ in range(args.warmup):
        grab(pipeline, align)

    for i in range(args.frames):
        got = grab(pipeline, align)
        if got is None:
            print(f"[cam] frame {i}: capture failed")
            continue

        rgb, depth_raw, depth_scale = got
        depth_mm = depth_raw.astype(np.float32) * depth_scale  # scale -> mm

        if args.frames == 1:
            prefix = args.out
            tag = ""
        else:
            prefix = args.out
            tag = f"_{i:03d}"

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(prefix, f"rgb{tag}.png"), bgr)
        np.save(os.path.join(prefix, f"depth{tag}.npy"), depth_raw)
        np.save(os.path.join(prefix, f"depth_mm{tag}.npy"), depth_mm)

        valid = depth_mm[depth_mm > 0]
        vis = np.zeros(depth_mm.shape, np.uint8)
        if valid.size:
            lo, hi = np.percentile(valid, [1, 99])
            vis = np.clip((depth_mm - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
            vis[depth_mm <= 0] = 0
        vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(prefix, f"depth_vis{tag}.png"), vis_color)

        pcd_note = ""
        if not args.no_pcd:
            pts, cols = depth_to_pcd(depth_mm, rgb, intr.fx, intr.fy, intr.cx, intr.cy,
                                     max_depth_mm=args.max_depth_mm)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols)
            if args.voxel > 0:
                pcd = pcd.voxel_down_sample(args.voxel)
            ply_path = os.path.join(prefix, f"cloud{tag}.ply")
            o3d.io.write_point_cloud(ply_path, pcd)
            pcd_note = f" | pcd={len(pcd.points)} pts -> {ply_path}"

        print(f"[frame {i}] rgb={rgb.shape} depth={depth_raw.shape} "
              f"depth_scale={depth_scale:.6f} "
              f"valid_depth_median={float(np.median(valid)) if valid.size else 0:.0f}mm "
              f"-> saved to {prefix}/*{tag}{pcd_note}")

    pipeline.stop()
    print(f"\ndone. output in ./{args.out}/")


if __name__ == "__main__":
    main()