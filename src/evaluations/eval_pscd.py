import os
import torch
import cv2
import argparse
import csv
from tqdm import tqdm
import torchmetrics

# Initialize TorchMetrics
miou_metric = torchmetrics.JaccardIndex(task='binary')
f1_metric = torchmetrics.F1Score(task='binary')


def calculate_metrics_torch(ground_truth, predicted):

    ground_truth_tensor = torch.tensor(ground_truth).unsqueeze(0)
    predicted_tensor = torch.tensor(predicted).unsqueeze(0)

    miou = miou_metric(predicted_tensor, ground_truth_tensor).item()
    f1 = f1_metric(predicted_tensor, ground_truth_tensor).item()

    return miou, f1


def evaluate_segmentation(gt_dir, pred_dir, stride=1, align=False):

    miou_scores = []
    f1_scores = []

    pred_files = sorted(os.listdir(pred_dir))

    step = (1024 - 224) // (15 - 1)

    for pred_file_name in tqdm(
        pred_files,
        desc=f"Evaluating {os.path.basename(pred_dir)}",
        unit="mask"
    ):

        if "vis" in pred_file_name:
            continue

        pred_path = os.path.join(pred_dir, pred_file_name)

        gt_file_name = pred_file_name
        gt_path = os.path.join(gt_dir, gt_file_name)

        if not os.path.exists(gt_path):
            continue

        # Read GT
        ground_truth = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        _, ground_truth = cv2.threshold(
            ground_truth,
            0,
            255,
            cv2.THRESH_BINARY
        )

        ground_truth = ground_truth // 255

        if not align:
            ground_truth[:, : step * stride] = 0

        # Read prediction
        predicted_binary = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if predicted_binary is None:
            continue

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

    print(f"{os.path.basename(pred_dir)} → mIoU={mean_miou:.4f}, F1={mean_f1:.4f}")

    return mean_miou, mean_f1


def evaluate_all_subsets(gt_root, pred_root, output_csv):

    subsets = {
        "stride1": {"stride": 1, "align": False},
        "stride2": {"stride": 2, "align": False},
        "align": {"stride": 1, "align": True},
    }

    results = []

    for subset, config in subsets.items():

        pred_dir = os.path.join(pred_root, subset)

        if not os.path.isdir(pred_dir):
            print(f"Skip {subset}, not found")
            continue

        print(f"\n===== Evaluating {subset} =====")

        miou, f1 = evaluate_segmentation(
            gt_root,
            pred_dir,
            stride=config["stride"],
            align=config["align"]
        )

        results.append((subset, miou, f1))

    # compute average
    if results:

        avg_miou = sum(x[1] for x in results) / len(results)
        avg_f1 = sum(x[2] for x in results) / len(results)

        results.append(("average", avg_miou, avg_f1))

    # save CSV
    with open(output_csv, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["subset", "miou", "f1"])

        for row in results:
            writer.writerow(row)

    print(f"\nSaved CSV to: {output_csv}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gt-root",
        required=True,
        help="ground truth root"
    )

    parser.add_argument(
        "--results-root",
        required=True,
        help="contains stride1 stride2 align"
    )

    parser.add_argument(
        "--output-csv",
        default="metrics_summary.csv"
    )

    args = parser.parse_args()
    if not os.path.exists(os.path.dirname(args.output_csv)):
        os.makedirs(os.path.dirname(args.output_csv))

    evaluate_all_subsets(
        args.gt_root,
        args.results_root,
        args.output_csv
    )


if __name__ == "__main__":
    main()
