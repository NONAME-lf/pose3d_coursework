"""
joint_angles.py
================
Базові функції обчислення кутів у суглобах за 3D-координатами ключових точок.

Усі функції приймають масиви numpy форми (..., 3), що представляють
тривимірні координати (x, y, z). Підтримується робота як з окремим кадром,
так і з послідовністю кадрів (T, 3).

Формула: кут між двома векторами обчислюється через скалярний добуток:
    cos(θ) = (v1 · v2) / (|v1| * |v2|)
    θ = arccos(cos(θ))

Результат завжди у градусах, у діапазоні [0°, 180°].
"""

import numpy as np


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    eps = 1e-8
    v1n = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + eps)
    v2n = v2 / (np.linalg.norm(v2, axis=-1, keepdims=True) + eps)
    cos_theta = np.clip(np.sum(v1n * v2n, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def joint_angle(p_proximal: np.ndarray, p_joint: np.ndarray,
                p_distal: np.ndarray) -> np.ndarray:
    v1 = p_proximal - p_joint
    v2 = p_distal - p_joint
    return angle_between_vectors(v1, v2)


def angle_to_vertical(p_lower: np.ndarray, p_upper: np.ndarray,
                      vertical_axis: int = 1) -> np.ndarray:
    segment = p_upper - p_lower
    vertical = np.zeros(3)
    vertical[vertical_axis] = 1.0
    return angle_between_vectors(segment, vertical)


def angle_in_horizontal_plane(p1: np.ndarray, p2: np.ndarray,
                               vertical_axis: int = 1) -> np.ndarray:
    v = p2 - p1
    # Take the two axes that are not vertical_axis
    axes = [i for i in range(3) if i != vertical_axis]
    a, b = axes
    return np.degrees(np.arctan2(v[..., b], v[..., a]))


def find_peak_frame(angle_series: np.ndarray, mode: str = "max") -> int:
    if mode == "max":
        return int(np.argmax(angle_series))
    elif mode == "min":
        return int(np.argmin(angle_series))
    raise ValueError(f"Невідомий mode: {mode!r}, очікується 'max' або 'min'")


def smooth_series(series: np.ndarray, window: int = 5) -> np.ndarray:
    if len(series) <= window or window <= 1:
        return np.asarray(series, dtype=float).copy()
    pad = window // 2
    padded = np.pad(series, pad, mode='edge')
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode='valid')


def median_in_window(series: np.ndarray, center: int,
                     window: int = 5) -> float:
    half = window // 2
    lo = max(0, center - half)
    hi = min(len(series), center + half + 1)
    return float(np.median(series[lo:hi]))
