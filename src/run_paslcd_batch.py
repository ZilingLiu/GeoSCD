import os
import torch
import torch.multiprocessing as mp
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
import sys
import numpy as np
import cv2

# 引入你的自定义模块
from framework import GeSCF
from vggt.models.vggt import VGGT
from unaligned_cd_dir import run_unaligned_cd_pairs 

def get_cli_args():
    parser = ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True,
                        help="root dir")
    parser.add_argument("--save-root", type=str, required=True,
                        help="save results to")
    parser.add_argument("--mode", type=str, default="occupy",
                        choices=["initial", "occupy"])
    parser.add_argument("--model-path", type=str, default="./pretrained/model.pt",
                        help="vggt path")
    parser.add_argument("--stride", type=int, default=5)
    
    # 其他参数保持默认或从命令行获取
    parser.add_argument('--align', action="store_true")
    parser.add_argument('--light', action="store_true")
    parser.add_argument('--transfer', action="store_true", default=False)
    parser.add_argument('--bright_thresh', default=25.0, type=float)
    parser.add_argument('--color_thresh', default=0.2, type=float)
    parser.add_argument('--iou_thresh', default=0.35, type=float)
    parser.add_argument('--sem_filter', type=float, default=None)
    parser.add_argument(
    '--gpus',
    type=str,
    default=None,
    help='Comma separated gpu ids, e.g. 0,1,2'
)
    
    return parser.parse_args()

def worker(rank, world_size, task_list, args_cli):
    """
    每个GPU的工作函数
    """

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    print(f"[GPU {rank}] Starting... Assigned {len(task_list[rank::world_size])} tasks.")


    base_model_args = Namespace(
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
        align=False,
        light=args_cli.light,
        bright_thresh=args_cli.bright_thresh,
        color_thresh=args_cli.color_thresh,
        transfer=args_cli.transfer,
        iou_thresh=args_cli.iou_thresh,
        sem_filter=args_cli.sem_filter
    )

    try:
        seg_model = GeSCF(base_model_args)
        if hasattr(seg_model, 'to'):
            seg_model = seg_model.to(device)
        # print(f"[GPU {rank}] SAM/GeSCF model loaded!")

        vggt_model = VGGT()
        vggt_model.load_state_dict(torch.load(args_cli.model_path, map_location=device), strict=True)
        vggt_model.eval().to(device)
        # print(f"[GPU {rank}] VGGT model loaded!")
    except Exception as e:
        print(f"[GPU {rank}] Error loading models: {e}")
        return

    my_tasks = task_list[rank::world_size]

    for task in my_tasks:
        scene_name, instance_name, instance_path = task
        
        save_dir = os.path.join(args_cli.save_root, scene_name, instance_name)
        os.makedirs(save_dir, exist_ok=True)


        current_args = Namespace(**vars(base_model_args))
        current_args.scene_dir = instance_path
        current_args.save_dir = save_dir
        
        print(f"[GPU {rank}] Processing: {scene_name}/{instance_name}")
        
        try:
          
            run_unaligned_cd_pairs(current_args, seg_model, vggt_model, device)
        except Exception as e:
            print(f"[GPU {rank}] Error processing {scene_name}/{instance_name}: {e}")

    print(f"[GPU {rank}] All tasks finished.")

if __name__ == "__main__":
    # 1. 获取命令行参数
    args_cli = get_cli_args()
    
    # 2. 扫描文件系统，生成任务列表
    print("Scanning dataset structure...")
    instances = ["Instance_1", "Instance_2"]
    
    # 确保 data_root 存在
    if not os.path.exists(args_cli.data_root):
        print(f"Error: Data root {args_cli.data_root} does not exist.")
        sys.exit(1)

    scenes = [d for d in os.listdir(args_cli.data_root)
              if os.path.isdir(os.path.join(args_cli.data_root, d))]
    
    # 收集所有有效的任务 (Scene, Instance, Path)
    all_tasks = []
    for scene in sorted(scenes): # 排序以保证每次运行顺序一致
        for instance in instances:
            instance_path = os.path.join(args_cli.data_root, scene, instance)
            if os.path.exists(instance_path):
                all_tasks.append((scene, instance, instance_path))
    
    total_tasks = len(all_tasks)
    if total_tasks == 0:
        print("No valid instances found. Please check data-root and directory structure.")
        sys.exit(0)

    print(f"Found {total_tasks} tasks (Scene/Instance pairs).")
    
    # 3. 确定 GPU 数量并启动多进程
    # world_size = torch.cuda.device_count()
    # if world_size == 0:
    #     print("No CUDA devices found!")
    #     sys.exit(1)
    if args_cli.gpus is None:
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = [int(x) for x in args_cli.gpus.split(',')]

    world_size = len(gpu_ids)
        
    print(f"Spawning {world_size} workers to process data...")
    
    mp.spawn(
        worker,
        args=(world_size, all_tasks, args_cli),
        nprocs=world_size,
        join=True
    )
    
    print("Main process execution complete.")