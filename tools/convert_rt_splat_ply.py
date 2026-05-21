#!/usr/bin/env python3
"""Convert an RT-Splatting checkpoint PLY into WebGPU splat assets."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
from plyfile import PlyData

SH_C0 = 0.28209479177387814
MAGIC_LITE = b"RTSPGS1\0"
MAGIC_FULL = b"RTSPGS2\0"
HEADER_SIZE = 64
STRIDE_LITE_FLOATS = 16
STRIDE_FULL_FLOATS = 72


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    norm = np.maximum(norm, 1.0e-8)
    return q / norm


def quaternion_to_axes(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = normalize_quaternion(q)
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    axis0 = np.stack((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)), axis=1)
    axis1 = np.stack((2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)), axis=1)
    return axis0, axis1


def get_columns(vertex: np.ndarray, prefix: str) -> np.ndarray:
    names = sorted((name for name in vertex.dtype.names or () if name.startswith(prefix)), key=lambda name: int(name.rsplit("_", 1)[-1]))
    if not names:
        raise ValueError(f"Missing PLY properties with prefix {prefix!r}")
    return np.stack([np.asarray(vertex[name], dtype=np.float32) for name in names], axis=1)


def get_optional_columns(vertex: np.ndarray, prefix: str, width: int, fill: float = 0.0) -> np.ndarray:
    names = sorted((name for name in vertex.dtype.names or () if name.startswith(prefix)), key=lambda name: int(name.rsplit("_", 1)[-1]))
    if not names:
        return np.full((vertex.shape[0], width), fill, dtype=np.float32)
    cols = np.stack([np.asarray(vertex[name], dtype=np.float32) for name in names], axis=1)
    if cols.shape[1] < width:
        out = np.full((cols.shape[0], width), fill, dtype=np.float32)
        out[:, : cols.shape[1]] = cols
        return out
    return cols[:, :width]


def read_xyz(vertex: np.ndarray) -> np.ndarray:
    return np.stack((np.asarray(vertex["x"], dtype=np.float32), np.asarray(vertex["y"], dtype=np.float32), np.asarray(vertex["z"], dtype=np.float32)), axis=1)


def read_sh(vertex: np.ndarray) -> np.ndarray:
    f_dc = np.stack((np.asarray(vertex["f_dc_0"], dtype=np.float32), np.asarray(vertex["f_dc_1"], dtype=np.float32), np.asarray(vertex["f_dc_2"], dtype=np.float32)), axis=1)
    f_rest = get_columns(vertex, "f_rest_")
    if f_rest.shape[1] != 45:
        raise ValueError(f"Expected 45 f_rest coefficients for SH degree 3, got {f_rest.shape[1]}")
    f_rest = f_rest.reshape((f_rest.shape[0], 3, 15))
    sh = np.zeros((f_rest.shape[0], 3, 16), dtype=np.float32)
    sh[:, :, 0] = f_dc
    sh[:, :, 1:] = f_rest
    return sh


def make_selection(xyz, axis0, axis1, occupancy, opacity, transmissivity, inside, args):
    crop_mask = np.ones(xyz.shape[0], dtype=bool)
    if args.crop_center is not None and args.crop_radius > 0:
        crop_center = np.asarray(args.crop_center, dtype=np.float32)
        crop_mask = np.linalg.norm(xyz - crop_center[None, :], axis=1) <= args.crop_radius
    effective_trans = transmissivity * inside
    visible_alpha = occupancy * ((1.0 - effective_trans) + effective_trans * opacity)
    scale_area = np.linalg.norm(axis0, axis=1) * np.linalg.norm(axis1, axis=1)
    importance = visible_alpha * np.sqrt(np.maximum(scale_area, 1.0e-12))
    keep = np.isfinite(importance) & (visible_alpha >= args.min_alpha) & crop_mask
    if args.max_splats > 0 and int(np.count_nonzero(keep)) > args.max_splats:
        candidate = np.flatnonzero(keep)
        top = np.argpartition(importance[candidate], -args.max_splats)[-args.max_splats:]
        selected = candidate[top]
        selected = selected[np.argsort(importance[selected])[::-1]]
    else:
        selected = np.flatnonzero(keep)
        selected = selected[np.argsort(importance[selected])[::-1]]
    if selected.size == 0:
        raise ValueError("No splats survived filtering/cropping")
    return selected, importance


def write_header(f, magic, version, count, stride, center, radius, sh_degree):
    header = bytearray(HEADER_SIZE)
    struct.pack_into("<8sIII4fI", header, 0, magic, version, count, stride, float(center[0]), float(center[1]), float(center[2]), float(radius), sh_degree)
    f.write(header)


def convert(args: argparse.Namespace) -> None:
    vertex = PlyData.read(Path(args.input))["vertex"].data
    xyz = read_xyz(vertex)
    sh = read_sh(vertex)
    scales = np.exp(get_columns(vertex, "scale_"))
    rots = get_columns(vertex, "rot_")
    axis0, axis1 = quaternion_to_axes(rots)
    axis0 = axis0 * scales[:, 0:1]
    axis1 = axis1 * scales[:, 1:2]
    occupancy = sigmoid(np.asarray(vertex["occupancy"], dtype=np.float32))
    opacity = sigmoid(np.asarray(vertex["opacity"], dtype=np.float32))
    transmissivity = sigmoid(np.asarray(vertex["transmissivity"], dtype=np.float32))
    roughness = sigmoid(np.asarray(vertex["roughness_0"], dtype=np.float32))
    reflectance = sigmoid(np.asarray(vertex["reflectance_0"], dtype=np.float32))
    feature = np.tanh(get_optional_columns(vertex, "feature_", 4))
    if args.env_center is not None and args.env_radius > 0:
        env_center = np.asarray(args.env_center, dtype=np.float32)
        inside = (np.linalg.norm(xyz - env_center[None, :], axis=1) <= args.env_radius).astype(np.float32)
    else:
        inside = np.ones(xyz.shape[0], dtype=np.float32)
    selected, importance = make_selection(xyz, axis0, axis1, occupancy, opacity, transmissivity, inside, args)
    xyz = xyz[selected]
    axis0 = axis0[selected] * args.radius_scale
    axis1 = axis1[selected] * args.radius_scale
    sh = sh[selected]
    occupancy = occupancy[selected]
    opacity = opacity[selected]
    transmissivity = transmissivity[selected]
    roughness = roughness[selected]
    reflectance = reflectance[selected]
    feature = feature[selected]
    inside = inside[selected]
    importance = importance[selected]
    if args.crop_center is not None and args.crop_radius > 0:
        center = np.asarray(args.crop_center, dtype=np.float32)
        radius = float(args.crop_radius)
    else:
        center = xyz.mean(axis=0).astype(np.float32)
        radius = float(np.percentile(np.linalg.norm(xyz - center, axis=1), 98.0))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        if args.asset_format == "lite":
            colors = np.clip(0.5 + SH_C0 * sh[:, :, 0], 0.0, 1.0)
            packed = np.zeros((xyz.shape[0], STRIDE_LITE_FLOATS), dtype=np.float32)
            packed[:, 0:3] = xyz
            packed[:, 4:7] = axis0
            packed[:, 8:11] = axis1
            packed[:, 12:15] = colors
            packed[:, 15] = occupancy * opacity
            write_header(f, MAGIC_LITE, 1, xyz.shape[0], STRIDE_LITE_FLOATS, center, radius, 0)
        else:
            effective_trans = transmissivity * inside
            alpha_hint = occupancy * ((1.0 - effective_trans) + effective_trans * opacity)
            packed = np.zeros((xyz.shape[0], STRIDE_FULL_FLOATS), dtype=np.float32)
            packed[:, 0:3] = xyz
            packed[:, 4:7] = axis0
            packed[:, 8:11] = axis1
            packed[:, 12] = occupancy
            packed[:, 13] = opacity
            packed[:, 14] = transmissivity
            packed[:, 15] = alpha_hint
            packed[:, 16] = roughness
            packed[:, 17] = reflectance
            packed[:, 18] = inside
            packed[:, 19] = importance
            packed[:, 20:24] = feature
            packed[:, 24:40] = sh[:, 0, :]
            packed[:, 40:56] = sh[:, 1, :]
            packed[:, 56:72] = sh[:, 2, :]
            write_header(f, MAGIC_FULL, 2, xyz.shape[0], STRIDE_FULL_FLOATS, center, radius, 3)
        f.write(packed.tobytes(order="C"))
    mb = out_path.stat().st_size / (1024.0 * 1024.0)
    print(f"Wrote {out_path}")
    print(f"format={args.asset_format} splats={xyz.shape[0]} size={mb:.1f} MiB center={center.tolist()} radius={radius:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asset-format", choices=["full", "lite"], default="full")
    parser.add_argument("--max-splats", type=int, default=180_000)
    parser.add_argument("--min-alpha", type=float, default=0.01)
    parser.add_argument("--radius-scale", type=float, default=2.0)
    parser.add_argument("--crop-center", nargs=3, type=float, default=None)
    parser.add_argument("--crop-radius", type=float, default=0.0)
    parser.add_argument("--env-center", nargs=3, type=float, default=None)
    parser.add_argument("--env-radius", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
