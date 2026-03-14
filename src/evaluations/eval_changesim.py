import os
import cv2
import csv
import argparse
from tqdm import tqdm
import torch
import torchmetrics

# 初始化 TorchMetrics
miou_metric = torchmetrics.JaccardIndex(task='binary')
f1_metric = torchmetrics.F1Score(task='binary')


def calculate_metrics_torch(ground_truth, predicted):

    ground_truth_tensor = torch.tensor(ground_truth).unsqueeze(0)
    predicted_tensor = torch.tensor(predicted).unsqueeze(0)

    miou = miou_metric(predicted_tensor, ground_truth_tensor).item()
    f1 = f1_metric(predicted_tensor, ground_truth_tensor).item()

    return miou, f1


def evaluate_changesim_all(gt_root, pred_root, stride=5, align=False):

    miou_scores = []
    f1_scores = []

    for warehouse in sorted(os.listdir(gt_root)):

        warehouse_gt_path = os.path.join(gt_root, warehouse)

        if not os.path.isdir(warehouse_gt_path):
            continue

        Seqs = ["Seq_0", "Seq_1"]

        for seq in Seqs:

            if not align:
                gt_dir = os.path.join(
                    warehouse_gt_path,
                    seq,
                    "change_segmentation_processed"
                )
            else:
                gt_dir = os.path.join(
                    warehouse_gt_path,
                    seq,
                    "change_segmentation"
                )

            pred_dir = os.path.join(pred_root, warehouse, seq)

            if not os.path.isdir(gt_dir) or not os.path.isdir(pred_dir):
                continue

            pred_files = [
                f for f in os.listdir(pred_dir)
                if f.endswith('.png') and "vis" not in f
            ]

            pred_files = sorted(
                pred_files,
                key=lambda x: int(x.split('.')[0])
            )

            for pred_file_name in tqdm(
                pred_files,
                desc=f"Evaluating {warehouse}/{seq}",
                unit="mask",
                leave=False
            ):

                predicted_binary_path = os.path.join(
                    pred_dir,
                    pred_file_name
                )

                predicted_binary = cv2.imread(
                    predicted_binary_path,
                    cv2.IMREAD_GRAYSCALE
                )

                if predicted_binary is None:
                    continue

                current_index = int(pred_file_name.split('.')[0])

                if align:
                    gt_file_name = pred_file_name
                else:
                    next_index = current_index + stride
                    gt_file_name = f"{current_index}_{next_index}.png"

                ground_truth_path = os.path.join(
                    gt_dir,
                    gt_file_name
                )

                if not os.path.exists(ground_truth_path):
                    continue

                ground_truth = cv2.imread(
                    ground_truth_path,
                    cv2.IMREAD_GRAYSCALE
                )

                _, ground_truth = cv2.threshold(
                    ground_truth,
                    0,
                    255,
                    cv2.THRESH_BINARY
                )

                ground_truth = ground_truth // 255

                if ground_truth.shape != predicted_binary.shape:

                    predicted_binary = cv2.resize(
                        predicted_binary,
                        (ground_truth.shape[1], ground_truth.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                _, predicted_binary = cv2.threshold(
                    predicted_binary,
                    127,
                    255,
                    cv2.THRESH_BINARY
                )

                predicted_binary = predicted_binary // 255

                miou, f1 = calculate_metrics_torch(
                    ground_truth,
                    predicted_binary
                )

                miou_scores.append(miou)
                f1_scores.append(f1)

    if miou_scores:

        mean_miou = torch.tensor(miou_scores).mean().item()
        mean_f1 = torch.tensor(f1_scores).mean().item()

    else:

        mean_miou = 0.0
        mean_f1 = 0.0

    return mean_miou, mean_f1


def evaluate_all_subsets(gt_root, results_root, output_csv):

    subsets = {
        "stride5": {"stride": 5, "align": False},
        "stride10": {"stride": 10, "align": False},
        "stride15": {"stride": 15, "align": False},
        "align": {"stride": 5, "align": True},
    }

    results = []

    for subset, config in subsets.items():

        pred_root = os.path.join(results_root, subset)

        if not os.path.isdir(pred_root):
            print(f"Skip {subset}, not found")
            continue

        print(f"\n===== Evaluating {subset} =====")

        miou, f1 = evaluate_changesim_all(
            gt_root,
            pred_root,
            stride=config["stride"],
            align=config["align"]
        )

        print(f"{subset}: mIoU={miou:.4f}, F1={f1:.4f}")

        results.append((subset, miou, f1))

    # 计算平均
    if results:

        avg_miou = sum(r[1] for r in results) / len(results)
        avg_f1 = sum(r[2] for r in results) / len(results)

        results.append(("average", avg_miou, avg_f1))
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    # 写CSV
    with open(output_csv, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["subset", "miou", "f1"])

        for row in results:
            writer.writerow(row)

    print(f"\nSaved to {output_csv}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--gt-root',
        required=True
    )

    parser.add_argument(
        '--results-root',
        required=True,
        help="contains stride5 stride10 stride15 align"
    )

    parser.add_argument(
        '--output-csv',
        default="../metrics/changesim_summary.csv"
    )
    if not os.path.exists(os.path.dirname(args.output_csv)):
        os.makedirs(os.path.dirname(args.output_csv))

    args = parser.parse_args()

    evaluate_all_subsets(
        args.gt_root,
        args.results_root,
        args.output_csv
    )


if __name__ == "__main__":
    main()
