import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parent))
    from exercise_analysis import (
        EXERCISES, analyze_exercise, format_report, compute_reference_ranges
    )
    from kimore_loader import load_exercise, split_healthy_patient
else:
    from .exercise_analysis import (
        EXERCISES, analyze_exercise, format_report, compute_reference_ranges
    )
    from .kimore_loader import load_exercise, split_healthy_patient


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exercise", choices=list(EXERCISES.keys()),
                   help="Яку вправу аналізувати (Ex1..Ex5)")
    p.add_argument("--pkl", default="data/kimore/kimore_exercise_dataset.pkl",
                   help="Шлях до KIMORE pkl-файлу")
    p.add_argument("--healthy-threshold", type=float, default=0.85,
                   help="Поріг cTS для розрізнення здорових і пацієнтів (за замовч. 0.85)")
    p.add_argument("--max-patient-reports", type=int, default=10,
                   help="Скільки звітів пацієнтів вивести (за замовч. 10)")
    p.add_argument("--output", type=str, default=None,
                   help="Файл для збереження виводу (за замовч. — лише на екран)")
    p.add_argument("--sigma", type=float, default=2.0,
                   help="Множник σ для границь норми (2.0=м'яко, 1.5=рекомендовано, 1.0=жорстко)")
    p.add_argument("--smooth-window", type=int, default=5,
                   help="Вікно згладжування головного кута перед пошуком піка")
    p.add_argument("--peak-window", type=int, default=5,
                   help="Вікно для медіани значень кутів навколо піка")
    p.add_argument("--no-trim-outliers", action="store_true",
                   help="НЕ викидати викиди зі здорових при обчисленні референсів")
    return p.parse_args()


def main():
    args = parse_args()
    spec = EXERCISES[args.exercise]

    # download KIMORE data
    print(f"Завантаження KIMORE з {args.pkl}...")
    skeletons, scores = load_exercise(args.pkl, args.exercise.lower())
    if not skeletons:
        print("Дані не завантажено. Перевір шлях до pkl та структуру датасету.")
        return

    # 2. divide healthy and patient subjects by the cTS threshold
    healthy_skels, healthy_scores, patient_skels, patient_scores = (
        split_healthy_patient(skeletons, scores, args.healthy_threshold)
    )
    print(f"\nЗдорових (cTS ≥ {args.healthy_threshold}): {len(healthy_skels)}")
    print(f"Пацієнтів (cTS < {args.healthy_threshold}): {len(patient_skels)}")

    if len(healthy_skels) < 5:
        print("\nУвага: занадто мало здорових суб'єктів для надійних референсних "
              "діапазонів. Рекомендується щонайменше 10 суб'єктів.")

    # 3. Compute reference ranges from healthy subjects and analyze patients
    print(f"\nОбчислення референсних діапазонів для {args.exercise} "
          f"за {len(healthy_skels)} здоровими суб'єктами "
          f"(σ-множник={args.sigma}, smooth={args.smooth_window}, "
          f"peak_window={args.peak_window})...")
    ref_ranges = compute_reference_ranges(
        healthy_skels, spec,
        sigma_multiplier=args.sigma,
        smooth_window=args.smooth_window,
        peak_window=args.peak_window,
        trim_outliers=not args.no_trim_outliers,
    )

    output_lines = []

    output_lines.append(f"Геометричний аналіз вправи {args.exercise}: {spec.description}")
    output_lines.append("=" * 70)
    output_lines.append("")
    output_lines.append("РЕФЕРЕНСНІ ДІАПАЗОНИ (mean ± 2σ для здорових)")
    output_lines.append("-" * 70)
    output_lines.append(f"{'Кут':<25} {'Min':>10} {'Max':>10}")
    for name, (lo, hi) in ref_ranges.items():
        output_lines.append(f"{name:<25} {lo:>9.1f}° {hi:>9.1f}°")
    output_lines.append("")

    # 4. Analyze patients and generate reports
    output_lines.append(f"АНАЛІЗ ПАЦІЄНТІВ ({len(patient_skels)} суб'єктів)")
    output_lines.append("=" * 70)

    classification_correct = 0
    classification_incorrect = 0
    deviations_by_subject = []

    n_to_show = min(args.max_patient_reports, len(patient_skels))
    for i, (skel, score) in enumerate(zip(patient_skels, patient_scores)):
        report = analyze_exercise(
            skel, spec, ref_ranges,
            smooth_window=args.smooth_window,
            peak_window=args.peak_window,
        )
        # Count total deviation for further correlation with cTS
        total_dev = sum(r.deviation for r in report.angles
                        if r.normal_range is not None)
        deviations_by_subject.append((score, total_dev, report.overall_correct))

        if report.overall_correct:
            classification_correct += 1
        else:
            classification_incorrect += 1

        if i < n_to_show:
            output_lines.append("")
            output_lines.append(f"--- Пацієнт {i+1}/{len(patient_skels)} "
                                f"(cTS={score:.3f}) ---")
            output_lines.append(format_report(report))

    if len(patient_skels) > n_to_show:
        output_lines.append("")
        output_lines.append(f"... ({len(patient_skels) - n_to_show} додаткових "
                            f"звітів пропущено; --max-patient-reports={n_to_show})")

    # 5. Summary statistics
    output_lines.append("")
    output_lines.append("ЗВЕДЕНА СТАТИСТИКА")
    output_lines.append("=" * 70)
    output_lines.append(f"Класифіковано як 'правильно':   {classification_correct} / "
                        f"{len(patient_skels)} пацієнтів")
    output_lines.append(f"Класифіковано як 'з помилками': {classification_incorrect} / "
                        f"{len(patient_skels)} пацієнтів")

    # Corelation between cTS and total geometric deviation
    if len(deviations_by_subject) >= 5:
        from scipy.stats import spearmanr
        cts = np.array([d[0] for d in deviations_by_subject])
        devs = np.array([d[1] for d in deviations_by_subject])
        rho, p = spearmanr(cts, devs)
        output_lines.append("")
        output_lines.append(f"Кореляція Спірмена (cTS vs сумарне геометричне відхилення):")
        output_lines.append(f"  ρ = {rho:.3f},  p-value = {p:.4f}")
        output_lines.append("Очікується від'ємна кореляція: чим нижча cTS, тим більше "
                            "відхилення.")

    full_output = "\n".join(output_lines)
    print()
    print(full_output)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(full_output, encoding="utf-8")
        print(f"\nЗвіт збережено у {args.output}")


if __name__ == "__main__":
    main()
