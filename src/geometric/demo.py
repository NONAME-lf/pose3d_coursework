import numpy as np

try:
    from .exercise_analysis import (
        EXERCISES, analyze_exercise, format_report, compute_reference_ranges,
    )
    from .kimore_joints import JOINTS, NUM_JOINTS
except ImportError:
    from exercise_analysis import (
        EXERCISES, analyze_exercise, format_report, compute_reference_ranges,
    )
    from kimore_joints import JOINTS, NUM_JOINTS


def make_base_skeleton() -> np.ndarray:
    skel = np.zeros((NUM_JOINTS, 3))
    # Body
    skel[JOINTS['SpineBase']]     = [0.0,  0.00, 0.0]
    skel[JOINTS['SpineMid']]      = [0.0,  0.30, 0.0]
    skel[JOINTS['SpineShoulder']] = [0.0,  0.55, 0.0]
    skel[JOINTS['Neck']]          = [0.0,  0.62, 0.0]
    skel[JOINTS['Head']]          = [0.0,  0.78, 0.0]
    # Shoulders and arms
    skel[JOINTS['ShoulderLeft']]  = [-0.18, 0.55, 0.0]
    skel[JOINTS['ShoulderRight']] = [ 0.18, 0.55, 0.0]
    skel[JOINTS['ElbowLeft']]     = [-0.18, 0.25, 0.0]
    skel[JOINTS['ElbowRight']]    = [ 0.18, 0.25, 0.0]
    skel[JOINTS['WristLeft']]     = [-0.18, 0.00, 0.0]
    skel[JOINTS['WristRight']]    = [ 0.18, 0.00, 0.0]
    skel[JOINTS['HandLeft']]      = [-0.18, -0.05, 0.0]
    skel[JOINTS['HandRight']]     = [ 0.18, -0.05, 0.0]
    skel[JOINTS['HandTipLeft']]   = [-0.18, -0.10, 0.0]
    skel[JOINTS['HandTipRight']]  = [ 0.18, -0.10, 0.0]
    skel[JOINTS['ThumbLeft']]     = [-0.20, -0.07, 0.0]
    skel[JOINTS['ThumbRight']]    = [ 0.20, -0.07, 0.0]
    # Legs
    skel[JOINTS['HipLeft']]       = [-0.10, 0.00, 0.0]
    skel[JOINTS['HipRight']]      = [ 0.10, 0.00, 0.0]
    skel[JOINTS['KneeLeft']]      = [-0.10, -0.45, 0.0]
    skel[JOINTS['KneeRight']]     = [ 0.10, -0.45, 0.0]
    skel[JOINTS['AnkleLeft']]     = [-0.10, -0.85, 0.0]
    skel[JOINTS['AnkleRight']]    = [ 0.10, -0.85, 0.0]
    skel[JOINTS['FootLeft']]      = [-0.10, -0.92, 0.10]
    skel[JOINTS['FootRight']]     = [ 0.10, -0.92, 0.10]
    return skel


def synth_ex1_correct(num_frames: int = 100,
                      noise_std: float = 0.005,
                      seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = make_base_skeleton()
    sequence = np.tile(base, (num_frames, 1, 1))

    upper_len = 0.30   # shoulder -> elbow
    forearm_len = 0.25  # elbow -> wrist
    hand_len = 0.05     # wrist -> hand

    # alpha from 0 to 1 and back to 0, controlling the lifting phase (up and down)
    half = num_frames // 2
    alpha_seq = np.concatenate([np.linspace(0, 1, half),
                                 np.linspace(1, 0, num_frames - half)])

    for i, alpha in enumerate(alpha_seq):
        # Angle of lifting: 0 -> π. If 0 hand down, pi/2 — to the side, π — up.
        theta = np.pi * alpha

        # Rind hand: movement in frontal area XY.
        sr = base[JOINTS['ShoulderRight']]
        # Direction shoulder->elbow: down=(0,-1,0), right=(+1,0,0), up=(0,+1,0).
        dir_right = np.array([np.sin(theta), -np.cos(theta), 0.0])
        sequence[i, JOINTS['ElbowRight']] = sr + upper_len * dir_right
        sequence[i, JOINTS['WristRight']] = sr + (upper_len + forearm_len) * dir_right
        sequence[i, JOINTS['HandRight']]  = sr + (upper_len + forearm_len + hand_len) * dir_right

        # Left hand: normal lifting for contrast
        sl = base[JOINTS['ShoulderLeft']]
        dir_left = np.array([-np.sin(theta), -np.cos(theta), 0.0])
        sequence[i, JOINTS['ElbowLeft']] = sl + upper_len * dir_left
        sequence[i, JOINTS['WristLeft']] = sl + (upper_len + forearm_len) * dir_left
        sequence[i, JOINTS['HandLeft']]  = sl + (upper_len + forearm_len + hand_len) * dir_left

    sequence += rng.normal(0, noise_std, sequence.shape)
    return sequence


def synth_ex1_incorrect(num_frames: int = 100, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = make_base_skeleton()

    # Tilt to the left on 12° around SpineBase.
    tilt_deg = 12
    tilt = np.deg2rad(tilt_deg)
    cos_t, sin_t = np.cos(tilt), np.sin(tilt)
    R = np.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0,      0,     1],
    ])
    pivot = base[JOINTS['SpineBase']]
    base_tilted = (base - pivot) @ R.T + pivot

    sequence = np.tile(base_tilted, (num_frames, 1, 1))

    upper_len = 0.30
    forearm_len = 0.25
    hand_len = 0.05

    half = num_frames // 2
    alpha_seq = np.concatenate([np.linspace(0, 1, half),
                                 np.linspace(1, 0, num_frames - half)])

    # Errors params
    max_lift_right = 0.65   # right hand only lifted to 65% of π
    elbow_bend_deg = 50     # deviation of forearm from shoulder line

    for i, alpha in enumerate(alpha_seq):
        # Right hand with errors.
        theta_right = np.pi * alpha * max_lift_right
        sr = base_tilted[JOINTS['ShoulderRight']]
        upper_dir = np.array([np.sin(theta_right), -np.cos(theta_right), 0.0])
        elbow_pos = sr + upper_len * upper_dir
        # The forearm is not in line with the shoulder—it is bent at a 50° angle.
        bend = np.deg2rad(elbow_bend_deg)
        # Rotation of `upper_dir` about the bend in the XY plane (elbow flexion in the frontal plane).
        forearm_dir = np.array([
            np.sin(theta_right - bend),
            -np.cos(theta_right - bend),
            0.0,
        ])
        wrist_pos = elbow_pos + forearm_len * forearm_dir
        hand_pos  = elbow_pos + (forearm_len + hand_len) * forearm_dir
        sequence[i, JOINTS['ElbowRight']] = elbow_pos
        sequence[i, JOINTS['WristRight']] = wrist_pos
        sequence[i, JOINTS['HandRight']]  = hand_pos

        # Left hand — normally (for contrast).
        theta_left = np.pi * alpha
        sl = base_tilted[JOINTS['ShoulderLeft']]
        dir_left = np.array([-np.sin(theta_left), -np.cos(theta_left), 0.0])
        sequence[i, JOINTS['ElbowLeft']] = sl + upper_len * dir_left
        sequence[i, JOINTS['WristLeft']] = sl + (upper_len + forearm_len) * dir_left
        sequence[i, JOINTS['HandLeft']]  = sl + (upper_len + forearm_len + hand_len) * dir_left

    sequence += rng.normal(0, 0.005, sequence.shape)
    return sequence


def main():
    print("=" * 70)
    print("ДЕМОНСТРАЦІЯ ГЕОМЕТРИЧНОГО АНАЛІЗУ ВИКОНАННЯ ВПРАВИ")
    print("=" * 70)

    # 1. We create 5 synthetic “healthy” runs to calculate the norms.    print("\nКрок 1: генерація синтетичних здорових виконань Ex1...")
    healthy_examples = [synth_ex1_correct(seed=i) for i in range(5)]

    #2. Calculate the reference ranges
    print("Крок 2: обчислення референсних діапазонів зі здорових...")
    ref_ranges = compute_reference_ranges(healthy_examples, EXERCISES["Ex1"])
    print("\nРеферентні діапазони (mean ± 2σ для здорових):")
    for name, (lo, hi) in ref_ranges.items():
        print(f"  {name:<22} {lo:>6.1f}° – {hi:.1f}°")

    #3. Analyze one correct solution.
    print("\n" + "=" * 70)
    print("ПРИКЛАД 1: правильне виконання вправи")
    print("=" * 70)
    correct_test = synth_ex1_correct(seed=42)
    report = analyze_exercise(correct_test, EXERCISES["Ex1"], ref_ranges)
    print(format_report(report))

    #4. Analyze one incorrect execution.
    print("\n" + "=" * 70)
    print("ПРИКЛАД 2: неправильне виконання вправи")
    print("=" * 70)
    incorrect_test = synth_ex1_incorrect(seed=99)
    report = analyze_exercise(incorrect_test, EXERCISES["Ex1"], ref_ranges)
    print(format_report(report))


if __name__ == "__main__":
    main()
