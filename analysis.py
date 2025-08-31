import torch
import numpy as np
from tqdm import tqdm
from config import Config


def analyze_thresholds(risk_level_a=0.3):
    try:
        output_path = Config.get_output_path()
        data = torch.load(output_path, map_location='cpu')
        print(f"Successfully loaded data from '{output_path}'.")
    except FileNotFoundError:
        print(f"Error: Detection result file '{output_path}' not found.")
        print("Please run 'detection.py' first to generate the result file.")
        return

    pred_masks = data['pred_masks']
    true_masks = data['true_masks']
    num_classes = data['num_classes']
    dataset_type = data.get('dataset_type', 'unknown')
    num_samples = len(pred_masks)

    print(f"Starting analysis for {num_samples} images...")
    print(f"Dataset type: {dataset_type}, Number of classes: {num_classes}")

    n = num_samples

    if risk_level_a * (n + 1) <= 1:
        print(f"Warning: Risk level a={risk_level_a} is too low to calculate a valid FNR threshold.")
        print(f"For {n} samples, please choose a value greater than {1 / (n + 1):.3f}.")
        return

    fnr_risk_threshold = (risk_level_a * (n + 1) - 1) / n
    print(f"User-specified risk level a = {risk_level_a}")


    thresholds = np.linspace(0.00, 1.00, 101)
    best_threshold = None
    min_avg_fnr = float('inf')
    best_metrics = {}

    for threshold in tqdm(thresholds, desc="Testing thresholds"):
        stats = {
            'tp': np.zeros(num_classes), 'fp': np.zeros(num_classes),
            'tn': np.zeros(num_classes), 'fn': np.zeros(num_classes)
        }

        for pred_score, true_mask in zip(pred_masks, true_masks):
            for c in range(num_classes):
                pred_binary = (pred_score[c] > threshold).float()
                true_binary = (true_mask[c] > 0).float()

                tp = ((pred_binary == 1) & (true_binary == 1)).sum().item()
                fp = ((pred_binary == 1) & (true_binary == 0)).sum().item()
                tn = ((pred_binary == 0) & (true_binary == 0)).sum().item()
                fn = ((pred_binary == 0) & (true_binary == 1)).sum().item()

                stats['tp'][c] += tp
                stats['fp'][c] += fp
                stats['tn'][c] += tn
                stats['fn'][c] += fn

        class_fnr = np.zeros(num_classes)
        for c in range(num_classes):
            tp_fn = stats['tp'][c] + stats['fn'][c]
            class_fnr[c] = stats['fn'][c] / tp_fn if tp_fn > 0 else 0.0

        avg_fnr = np.mean(class_fnr)

        if avg_fnr < min_avg_fnr:
            min_avg_fnr = avg_fnr
            best_threshold = threshold
            best_metrics = {
                'class_fnr': class_fnr,
                'stats': stats
            }

    print("\n--- Analysis Complete ---")

    if best_threshold is not None and min_avg_fnr < fnr_risk_threshold:
        print(f"Best threshold found under FNR < {fnr_risk_threshold:.4f} constraint: {best_threshold:.2f}")
        print(f"Corresponding lowest average False Negative Rate (FNR): {min_avg_fnr:.4f}")

        print("\nDetailed metrics at the best threshold:")
        stats = best_metrics['stats']
        for c in range(num_classes):
            class_name = "Defect" if dataset_type == "KSDD2" else f"Class {c + 1}"
            print(f"  {class_name}:")
            print(f"    False Negative Rate (FNR): {best_metrics['class_fnr'][c]:.4f}")
            print(f"    TP: {int(stats['tp'][c])}, FN: {int(stats['fn'][c])}, FP: {int(stats['fp'][c])}, TN: {int(stats['tn'][c])}")
    else:
        print(f"No threshold found that satisfies the average FNR < {fnr_risk_threshold:.4f} constraint.")
        if best_threshold is not None:
            print(f"The lowest average FNR found was {min_avg_fnr:.4f} (at threshold {best_threshold:.2f}), but this does not meet the constraint.")
        print("Please try relaxing the risk level 'a'.")


if __name__ == "__main__":
    user_risk_level_a = Config.USER_RISK_LEVEL_A
    analyze_thresholds(risk_level_a=user_risk_level_a)