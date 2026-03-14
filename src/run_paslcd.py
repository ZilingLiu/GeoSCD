import os
import torch
from argparse import ArgumentParser, Namespace

from framework import GeSCF
from vggt.models.vggt import VGGT

from unaligned_cd_dir import run_unaligned_cd_pairs   

parser = ArgumentParser()
parser.add_argument("--data-root", type=str, required=True,
                    help="root dir")
parser.add_argument("--save-root", type=str, required=True)

parser.add_argument("--mode", type=str, default="occupy",
                    choices=["initial", "occupy"])
parser.add_argument("--model-path", type=str, default="./pretrained/model.pt")
parser.add_argument("--stride", type=int, default=5,
                   )
    
parser.add_argument('--align', action="store_true")
parser.add_argument('--light', action="store_true")
parser.add_argument('--transfer', action="store_true", default=False)
parser.add_argument('--bright_thresh', default=25.0, type=float)
parser.add_argument('--color_thresh', default=0.2, type=float)
parser.add_argument('--iou_thresh', default=0.35, type=float)
parser.add_argument('--sem_filter', type=float, default=None)
parser.add_argument('--dilate',action="store_true", default=False)
args_cli = parser.parse_args()

instances = ["Instance_1", "Instance_2"]
# strides = [5, 10, 15]

args = Namespace(
            output_size=512,
            test_dataset="Random",
            feature_facet="key",
            feature_layer=17,
            embedding_layer=32,
            sam_backbone="vit_h",
            points_per_side=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.7,
            pseudo_backbone="vit_h",
            dataset="paslcd",
            stride=args_cli.stride,
            mode=args_cli.mode,
            iou_thresh = args_cli.iou_thresh,
            sem_filter = args_cli.sem_filter,
            align=False
        )
seg_model = GeSCF(args)
seg_model.eval()
print("SAM/GeSCF model loaded!")

device = "cuda"
vggt_model = VGGT()
vggt_model.load_state_dict(torch.load(args_cli.model_path, map_location=device), strict=True)
vggt_model.eval().to(device)
print("VGGT model loaded!")

# -------------------- 遍历场景 --------------------
scenes = [d for d in os.listdir(args_cli.data_root)
          if os.path.isdir(os.path.join(args_cli.data_root, d))]

for scene in scenes:
    for instance in instances:
        instance_path = os.path.join(args_cli.data_root, scene, instance)
        if not os.path.exists(instance_path):
            continue

        # for stride in strides:
        save_dir = os.path.join(args_cli.save_root, scene, instance)
        os.makedirs(save_dir, exist_ok=True)

        # 构造 args，传给 run_unaligned_cd_pairs
        args = Namespace(
            scene_dir=instance_path,
            save_dir=save_dir,
            output_size=512,
            test_dataset="Random",
            feature_facet="key",
            feature_layer=17,
            embedding_layer=32,
            sam_backbone="vit_h",
            points_per_side=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.7,
            pseudo_backbone="vit_h",
            dataset="paslcd",
            stride=args_cli.stride,
            mode=args_cli.mode,
            iou_thresh = args_cli.iou_thresh,
            sem_filter = args_cli.sem_filter,
            align=False
        )
        

        print(f"Processing scene {scene}, {instance}, stride {args_cli.stride}")
        run_unaligned_cd_pairs(args, seg_model, vggt_model, device)
