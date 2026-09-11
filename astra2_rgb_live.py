#!/usr/bin/env python3
"""
Live RGB preview for Orbbec Astra 2.

Usage:
  python astra2_rgb_live.py
  python astra2_rgb_live.py --color-res 1280x960x30

Press Q or ESC to quit.
"""

import argparse
import time
import cv2
import numpy as np

from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat


def parse_res(spec: str):
    w, h, fps = spec.lower().split("x")
    return int(w), int(h), int(fps)


def pick_color_profile(profiles, w: int, h: int, fps: int, fmt):
    for i in range(profiles.get_count()):
        p = profiles.get_stream_profile_by_index(i).as_video_stream_profile()
        if (p.get_width(), p.get_height(), p.get_fps(), p.get_format()) == (w, h, fps, fmt):
            return p

    available = []
    for i in range(profiles.get_count()):
        p = profiles.get_stream_profile_by_index(i).as_video_stream_profile()
        available.append((p.get_width(), p.get_height(), p.get_fps(), p.get_format()))

    raise RuntimeError(
        f"No {w}x{h}@{fps} {fmt} color profile.\nAvailable:\n"
        + "\n".join(str(x) for x in available)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--color-res", default="1280x960x30", help="WxHxFPS for RGB stream")
    parser.add_argument("--window", default="Astra2 RGB Live", help="OpenCV window title")
    args = parser.parse_args()

    cw, ch, cfps = parse_res(args.color_res)

    pipeline = Pipeline()
    config = Config()

    cp = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    config.enable_stream(pick_color_profile(cp, cw, ch, cfps, OBFormat.RGB))

    pipeline.start(config)

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    print("[live] Streaming started. Press Q or ESC to quit.")

    prev_t = time.time()
    fps_ema = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames(200)
            if frames is None:
                continue

            color = frames.get_color_frame()
            if color is None:
                continue

            w, h = color.get_width(), color.get_height()
            rgb = np.frombuffer(color.get_data(), np.uint8).reshape(h, w, 3)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            fps_ema = inst_fps if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * inst_fps

            cv2.putText(
                bgr,
                f"{w}x{h}  FPS:{fps_ema:5.1f}",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(args.window, bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[live] Stopped.")


if __name__ == "__main__":
    main()
