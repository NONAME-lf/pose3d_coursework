import pickle
from pathlib import Path

import numpy as np

# Lowercase order of joints prepare_kimore.py
LOWERCASE_JOINT_NAMES = [
    'spinebase', 'spinemid', 'neck', 'head',
    'shoulderleft', 'elbowleft', 'wristleft', 'handleft',
    'shoulderright', 'elbowright', 'wristright', 'handright',
    'hipleft', 'kneeleft', 'ankleleft', 'footleft',
    'hipright', 'kneeright', 'ankleright', 'footright',
    'spineshoulder', 'handtipleft', 'thumbleft', 'handtipright', 'thumbright',
]


def _extract_positions(joint_data) -> np.ndarray:
    arr = np.asarray(joint_data)
    return arr[:, -3:]


def load_subject_skeleton(df, idx: int) -> np.ndarray | None:
    positions_per_joint = []
    n_frames = None

    for joint in LOWERCASE_JOINT_NAMES:
        if joint not in df.columns:
            return None
        try:
            pos = _extract_positions(df.iloc[idx][joint])
        except Exception:
            return None
        if n_frames is None:
            n_frames = pos.shape[0]
        elif pos.shape[0] != n_frames:
            # Inconsistent frame count across joints, skip this subject
            return None
        positions_per_joint.append(pos)

    if n_frames is None or n_frames < 10:
        return None

    # Stack into shape (T, 25, 3).
    skeleton = np.stack(positions_per_joint, axis=1)
    return skeleton


def load_exercise(pkl_path: str | Path,
                  exercise: str) -> tuple[list, list]:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    key = exercise.lower()
    if key not in data:
        raise KeyError(f"Вправа {exercise!r} не знайдена в pkl. "
                       f"Доступні: {list(data.keys())}")

    df = data[key]
    skeletons = []
    scores = []
    skipped = 0

    for idx in range(len(df)):
        skel = load_subject_skeleton(df, idx)
        if skel is None:
            skipped += 1
            continue
        try:
            score = float(df.iloc[idx]['cTS'])
        except (KeyError, ValueError, TypeError):
            score = float('nan')
        skeletons.append(skel)
        scores.append(score)

    print(f"Завантажено {len(skeletons)} записів вправи {exercise} "
          f"(пропущено {skipped}).")
    return skeletons, scores


def split_healthy_patient(skeletons: list, scores: list,
                          healthy_threshold: float = 0.85
                          ) -> tuple[list, list, list, list]:
    healthy_skels, healthy_scores = [], []
    patient_skels, patient_scores = [], []
    for skel, score in zip(skeletons, scores):
        if np.isnan(score):
            continue
        if score >= healthy_threshold:
            healthy_skels.append(skel)
            healthy_scores.append(score)
        else:
            patient_skels.append(skel)
            patient_scores.append(score)
    return healthy_skels, healthy_scores, patient_skels, patient_scores
