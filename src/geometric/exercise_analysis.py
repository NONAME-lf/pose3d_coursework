from dataclasses import dataclass, field
from typing import Callable

import numpy as np

try:
    from .joint_angles import (
        joint_angle, angle_to_vertical, find_peak_frame, angle_between_vectors,
        smooth_series, median_in_window,
    )
    from .kimore_joints import get_joint
except ImportError:
    from joint_angles import (
        joint_angle, angle_to_vertical, find_peak_frame, angle_between_vectors,
        smooth_series, median_in_window,
    )
    from kimore_joints import get_joint


@dataclass
class AngleSpec:
    name: str
    compute: Callable[[np.ndarray], np.ndarray]
    normal_range: tuple = None  # (min_deg, max_deg) - filled later based on healthy subjects
    description: str = ""


@dataclass
class ExerciseSpec:
    """Специфікація однієї вправи KIMORE."""
    name: str
    description: str
    # Кут, за яким визначаємо піковий кадр вправи.
    peak_angle: str
    peak_mode: str = "max"  # "max" або "min"
    angles: list = field(default_factory=list)


# Ex1 — Lifting of the arms

def _ex1_shoulder_right(skel):
    return joint_angle(
        get_joint(skel, 'SpineBase'),
        get_joint(skel, 'ShoulderRight'),
        get_joint(skel, 'ElbowRight'),
    )

def _ex1_shoulder_left(skel):
    return joint_angle(
        get_joint(skel, 'SpineBase'),
        get_joint(skel, 'ShoulderLeft'),
        get_joint(skel, 'ElbowLeft'),
    )

def _ex1_elbow_right(skel):
    return joint_angle(
        get_joint(skel, 'ShoulderRight'),
        get_joint(skel, 'ElbowRight'),
        get_joint(skel, 'WristRight'),
    )

def _ex1_elbow_left(skel):
    return joint_angle(
        get_joint(skel, 'ShoulderLeft'),
        get_joint(skel, 'ElbowLeft'),
        get_joint(skel, 'WristLeft'),
    )

def _trunk_lean(skel):
    return angle_to_vertical(
        get_joint(skel, 'SpineBase'),
        get_joint(skel, 'SpineShoulder'),
    )

EX1 = ExerciseSpec(
    name="Ex1",
    description="Lifting of the arms — піднімання обох рук вгору",
    peak_angle="shoulder_right",
    peak_mode="max",
    angles=[
        AngleSpec("shoulder_right", _ex1_shoulder_right,
                  description="підняття правої руки (180° = повністю вгору)"),
        AngleSpec("shoulder_left", _ex1_shoulder_left,
                  description="підняття лівої руки"),
        AngleSpec("elbow_right", _ex1_elbow_right,
                  description="випрямлення правого ліктя (180° = пряма рука)"),
        AngleSpec("elbow_left", _ex1_elbow_left,
                  description="випрямлення лівого ліктя"),
        AngleSpec("trunk_lean", _trunk_lean,
                  description="нахил тулуба від вертикалі (0° = рівно)"),
    ],
)


# Ex2 — Lateral tilt of the trunk with arms in extension

def _ex2_arm_horizontal_right(skel):
    """Кут між тулубом і правою рукою — має бути ~90° (рука в сторону)."""
    return joint_angle(
        get_joint(skel, 'SpineBase'),
        get_joint(skel, 'ShoulderRight'),
        get_joint(skel, 'WristRight'),
    )

def _ex2_arm_horizontal_left(skel):
    return joint_angle(
        get_joint(skel, 'SpineBase'),
        get_joint(skel, 'ShoulderLeft'),
        get_joint(skel, 'WristLeft'),
    )

EX2 = ExerciseSpec(
    name="Ex2",
    description="Lateral tilt of the trunk — бічний нахил з витягнутими руками",
    peak_angle="trunk_lean",
    peak_mode="max",
    angles=[
        AngleSpec("trunk_lean", _trunk_lean,
                  description="нахил тулуба вбік (має бути значним, 30-45°)"),
        AngleSpec("arm_extension_right", _ex2_arm_horizontal_right,
                  description="витягнутість правої руки (90° = перпендикулярно тулубу)"),
        AngleSpec("arm_extension_left", _ex2_arm_horizontal_left,
                  description="витягнутість лівої руки"),
        AngleSpec("elbow_right", _ex1_elbow_right,
                  description="права рука пряма (180°)"),
        AngleSpec("elbow_left", _ex1_elbow_left,
                  description="ліва рука пряма (180°)"),
    ],
)


# Ex3 — Trunk rotation

def _ex3_shoulder_axis_to_hip_axis(skel):
    """Кут між віссю плечей і віссю стегон у горизонтальній площині.

    Коли людина стоїть рівно — обидві осі паралельні (0°). Під час обертання
    тулуба плечі повертаються відносно стегон, кут зростає.
    """
    sl = get_joint(skel, 'ShoulderLeft')
    sr = get_joint(skel, 'ShoulderRight')
    hl = get_joint(skel, 'HipLeft')
    hr = get_joint(skel, 'HipRight')
    # Proections on horizontal plane (ignore vertical Y axis)
    shoulder_vec = sr - sl
    hip_vec = hr - hl
    shoulder_vec[..., 1] = 0
    hip_vec[..., 1] = 0
    return angle_between_vectors(shoulder_vec, hip_vec)

EX3 = ExerciseSpec(
    name="Ex3",
    description="Trunk rotation — обертання тулуба",
    peak_angle="trunk_twist",
    peak_mode="max",
    angles=[
        AngleSpec("trunk_twist", _ex3_shoulder_axis_to_hip_axis,
                  description="ротація плечей відносно тазу (0° = рівно, 30-60° на піку)"),
        AngleSpec("trunk_lean", _trunk_lean,
                  description="вертикальність тулуба (має лишатись 0°)"),
    ],
)


# Ex4 — Pelvis rotations
def _ex4_hip_rotation(skel):
    hl = get_joint(skel, 'HipLeft')
    hr = get_joint(skel, 'HipRight')
    hip_vec = hr - hl
    hip_vec[..., 1] = 0  # proection on horizontal plane
    # Take the initial hip vector as reference (assuming subject starts facing forward)
    initial = hip_vec[0]
    return angle_between_vectors(hip_vec, initial[None, :])

EX4 = ExerciseSpec(
    name="Ex4",
    description="Pelvis rotations — обертання тазу",
    peak_angle="pelvis_rotation",
    peak_mode="max",
    angles=[
        AngleSpec("pelvis_rotation", _ex4_hip_rotation,
                  description="ротація тазу від початкового положення"),
        AngleSpec("trunk_lean", _trunk_lean,
                  description="вертикальність тулуба (має лишатись 0°)"),
    ],
)


# Ex5 — Squatting

def _knee_angle_right(skel):
    return joint_angle(
        get_joint(skel, 'HipRight'),
        get_joint(skel, 'KneeRight'),
        get_joint(skel, 'AnkleRight'),
    )

def _knee_angle_left(skel):
    return joint_angle(
        get_joint(skel, 'HipLeft'),
        get_joint(skel, 'KneeLeft'),
        get_joint(skel, 'AnkleLeft'),
    )

def _hip_angle_right(skel):
    return joint_angle(
        get_joint(skel, 'SpineBase'),
        get_joint(skel, 'HipRight'),
        get_joint(skel, 'KneeRight'),
    )

EX5 = ExerciseSpec(
    name="Ex5",
    description="Squatting — присідання",
    peak_angle="knee_right",
    peak_mode="min",  # on the bottom of the squat, knee angle is minimalq
    angles=[
        AngleSpec("knee_right", _knee_angle_right,
                  description="згинання правого коліна (180° = стоячи, ~90° на дні)"),
        AngleSpec("knee_left", _knee_angle_left,
                  description="згинання лівого коліна"),
        AngleSpec("hip_right", _hip_angle_right,
                  description="згинання правого стегна"),
        AngleSpec("trunk_lean", _trunk_lean,
                  description="нахил тулуба (має лишатись помірним, до 30°)"),
    ],
)


EXERCISES = {
    "Ex1": EX1,
    "Ex2": EX2,
    "Ex3": EX3,
    "Ex4": EX4,
    "Ex5": EX5,
}


# Analysis functions and report generation
@dataclass
class AngleResult:
    name: str
    description: str
    measured: float
    normal_range: tuple
    in_norm: bool
    deviation: float  # how much the measured value deviates from the nearest bound of the normal range


@dataclass
class ExerciseReport:
    exercise: str
    peak_frame: int
    total_frames: int
    angles: list  # list[AngleResult]
    overall_correct: bool


def analyze_exercise(skeleton: np.ndarray,
                     spec: ExerciseSpec,
                     reference_ranges: dict = None,
                     smooth_window: int = 5,
                     peak_window: int = 5) -> ExerciseReport:
    # Compute all angle series for the exercise.
    angle_series = {}
    for ang in spec.angles:
        angle_series[ang.name] = ang.compute(skeleton)

    # Find the peak frame on the smoothed main angle — this makes the search
    # robust to individual noisy frames from Kinect
    main_smooth = smooth_series(angle_series[spec.peak_angle], smooth_window)
    peak_idx = find_peak_frame(main_smooth, spec.peak_mode)

    # Collect results at the peak frame — median in a window instead of
    # raw value at the single peak frame, for robustness to noise
    results = []
    all_in_norm = True
    for ang in spec.angles:
        measured = median_in_window(angle_series[ang.name], peak_idx, peak_window)

        if reference_ranges and ang.name in reference_ranges:
            rng = reference_ranges[ang.name]
        else:
            rng = ang.normal_range

        if rng is None:
            # Without a reference range, skip norm check but still report the measured value
            results.append(AngleResult(
                ang.name, ang.description, measured, None, True, 0.0
            ))
            continue

        lo, hi = rng
        in_norm = (lo <= measured <= hi)
        if measured < lo:
            deviation = lo - measured
        elif measured > hi:
            deviation = measured - hi
        else:
            deviation = 0.0

        if not in_norm:
            all_in_norm = False

        results.append(AngleResult(
            ang.name, ang.description, measured, rng, in_norm, deviation
        ))

    return ExerciseReport(
        exercise=spec.name,
        peak_frame=peak_idx,
        total_frames=len(skeleton),
        angles=results,
        overall_correct=all_in_norm,
    )


def format_report(report: ExerciseReport) -> str:
    lines = []
    lines.append(f"Вправа: {report.exercise}")
    lines.append(f"Пік досягнуто на кадрі {report.peak_frame}/{report.total_frames}")
    lines.append("")
    lines.append(f"{'Кут':<22} {'Виміряно':>10} {'Норма':>16} {'Статус':>10}")
    lines.append("-" * 64)

    for r in report.angles:
        if r.normal_range is None:
            norm_str = "—"
            status = "?"
        else:
            lo, hi = r.normal_range
            norm_str = f"{lo:.0f}°–{hi:.0f}°"
            if r.in_norm:
                status = "OK"
            else:
                status = f"−{r.deviation:.0f}°"
        lines.append(f"{r.name:<22} {r.measured:>9.1f}° {norm_str:>16} {status:>10}")

    lines.append("")
    if report.overall_correct:
        lines.append("Висновок: вправу виконано правильно.")
    else:
        problems = [r for r in report.angles
                    if r.normal_range is not None and not r.in_norm]
        lines.append("Висновок: вправу виконано з порушеннями:")
        for r in problems:
            lines.append(f"  - {r.description} (відхилення {r.deviation:.0f}°)")

    return "\n".join(lines)


def compute_reference_ranges(healthy_skeletons: list,
                              spec: ExerciseSpec,
                              sigma_multiplier: float = 2.0,
                              smooth_window: int = 5,
                              peak_window: int = 5,
                              trim_outliers: bool = True) -> dict:
    peak_values = {ang.name: [] for ang in spec.angles}

    for skel in healthy_skeletons:
        # Compute all angle series for the exercise
        series = {ang.name: ang.compute(skel) for ang in spec.angles}
        # Look for the peak frame on the smoothed main angle
        main_smooth = smooth_series(series[spec.peak_angle], smooth_window)
        peak_idx = find_peak_frame(main_smooth, spec.peak_mode)
        # Values — median in a window around the peak.
        for ang in spec.angles:
            peak_values[ang.name].append(
                median_in_window(series[ang.name], peak_idx, peak_window)
            )

    # mean +- k·delta for each angle, optionally with outlier removal.
    ranges = {}
    for name, values in peak_values.items():
        arr = np.array(values)
        if trim_outliers and len(arr) >= 6:
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            mask = (arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)
            if mask.sum() >= 4:  # need at least 4 values to compute mean/std
                arr = arr[mask]
        mean = arr.mean()
        std = arr.std()
        ranges[name] = (mean - sigma_multiplier * std,
                        mean + sigma_multiplier * std)
    return ranges
