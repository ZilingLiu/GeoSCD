import os
import cv2
import argparse
import csv
from tqdm import tqdm
import torch
import torchmetrics
import json


# TorchMetrics
miou_metric = torchmetrics.JaccardIndex(task='binary')
f1_metric = torchmetrics.F1Score(task='binary')


def calculate_metrics_torch(ground_truth, predicted):

    ground_truth_tensor = torch.tensor(ground_truth).unsqueeze(0)
    predicted_tensor = torch.tensor(predicted).unsqueeze(0)

    miou = miou_metric(predicted_tensor, ground_truth_tensor).item()
    f1 = f1_metric(predicted_tensor, ground_truth_tensor).item()

    return miou, f1


def evaluate_changesim_stride(gt_root, pred_root, stride):

    miou_scores = []
    f1_scores = []

    for warehouse in sorted(os.listdir(gt_root)):

        warehouse_gt_path = os.path.join(gt_root, warehouse)

        if not os.path.isdir(warehouse_gt_path):
            continue

        Seqs = ["Instance_1", "Instance_2"]

        for seq in Seqs:

            gt_dir = os.path.join(
                warehouse_gt_path,
                seq,
                f"gt_{stride}"
            )

            json_path = os.path.join(
                warehouse_gt_path,
                seq,
                f"pairs_{stride}.json"
            )

            if not os.path.exists(json_path):
                continue

            with open(json_path, 'r', encoding='utf-8') as f:
                img_pairs_json = json.load(f)

            img_pairs_list = img_pairs_json["pairs"]

            image_pairs = {
                pair["img0"]: pair["img1"]
                for pair in img_pairs_list
            }

            pred_dir = os.path.join(pred_root, warehouse, seq)

            if not os.path.isdir(gt_dir) or not os.path.isdir(pred_dir):
                continue

            pred_files = [
                f for f in os.listdir(pred_dir)
                if "vis" not in f
            ]

            pred_files = sorted(pred_files)

            for pred_file_name in tqdm(
                pred_files,
                desc=f"Stride {stride} | {warehouse}/{seq}",
                leave=False
            ):

                pred_path = os.path.join(pred_dir, pred_file_name)

                predicted_binary = cv2.imread(
                    pred_path,
                    cv2.IMREAD_GRAYSCALE
                )

                if predicted_binary is None:
                    continue

                img_name = pred_file_name.split('.')[0]

                if img_name not in image_pairs:
                    continue

                ref_file_name = image_pairs[img_name]

                gt_file_name = f"{img_name}_vs_{ref_file_name}.png"

                gt_path = os.path.join(gt_dir, gt_file_name)

                if not os.path.exists(gt_path):
                    continue

                ground_truth = cv2.imread(
                    gt_path,
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
                        (ground_truth.shape[1],
                         ground_truth.shape[0]),
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

    print(f"Stride {stride}: mIoU={mean_miou:.4f}, F1={mean_f1:.4f}")

    return mean_miou, mean_f1


def evaluate_all(gt_root, results_root, output_csv):

    strides = [5, 10, 15]

    results = []

    for stride in strides:

        print(f"\n===== Evaluating stride{stride} =====")

        pred_root = os.path.join(
            results_root,
            f"stride{stride}"
        )

        if not os.path.isdir(pred_root):
            print(f"Skip stride{stride}, not found")
            continue

        miou, f1 = evaluate_changesim_stride(
            gt_root,
            pred_root,
            stride
        )

        results.append(
            (f"stride{stride}", miou, f1)
        )

    # average
    if results:

        avg_miou = sum(x[1] for x in results) / len(results)
        avg_f1 = sum(x[2] for x in results) / len(results)

        results.append(
            ("average", avg_miou, avg_f1)
        )

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
        '--gt-root',
        required=True
    )

    parser.add_argument(
        '--results-root',
        required=True
    )

    parser.add_argument(
        '--output-csv',
        default="metrics/metrics_summary.csv"
    )

    args = parser.parse_args()
    if not os.path.exists(os.path.dirname(args.output_csv)):
        os.makedirs(os.path.dirname(args.output_csv))
        
    evaluate_all(
        args.gt_root,
        args.results_root,
        args.output_csv
    )


if __name__ == "__main__":
    main()
