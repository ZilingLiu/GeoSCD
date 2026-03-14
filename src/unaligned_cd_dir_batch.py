import os 
import numpy as np
import cv2
from tqdm import tqdm
import logging
import argparse
import matplotlib.pyplot as plt
import math

# 多进程相关库
import torch
import torch.multiprocessing as mp

from framework import GeSCF
from utils import calculate_metric, visualize_results_custom
from vggt.models.vggt import VGGT

from pixel_match import run_dense_match
from datasets.load_dataset import load_dataset

# 日志设置
logging.basicConfig(
    level=logging.INFO,               
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 支持的图像后缀名
IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.JPG', '.PNG']

def find_matching_file(basename, directory):
    """在指定目录下根据文件名（忽略扩展名）寻找匹配文件"""
    for ext in IMG_EXTENSIONS:
        candidate = os.path.join(directory, basename + ext)
        if os.path.exists(candidate):
            return candidate
    return None

def map_right_mask_to_left(right2left_match: np.ndarray, right_mask: np.ndarray):
    """
    将 Right 图像的 mask 映射到 Left 图像上。
    """
    H, W = right_mask.shape
    left_mask = np.zeros((H, W), dtype=bool)

    right_x = right2left_match[..., 0].astype(np.int64)
    right_y = right2left_match[..., 1].astype(np.int64)

    valid_x = (right_x >= 0) & (right_x < W)
    valid_y = (right_y >= 0) & (right_y < H)
    valid = valid_x & valid_y

    left_mask[valid] = right_mask[right_y[valid], right_x[valid]]

    return left_mask

def gpu_worker(rank, world_size, args, img_pairs_list, query_dir, ref_dir):

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    total_len = len(img_pairs_list)
    chunk_size = math.ceil(total_len / world_size)
    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, total_len)
    
    local_pairs = img_pairs_list[start_idx:end_idx]
    
    if len(local_pairs) == 0:
        print(f"[GPU {rank}] No data to process, exiting.")
        return

    print(f"[GPU {rank}] Processing {len(local_pairs)} pairs (Index {start_idx} to {end_idx})...")

  
    # --- Load VGGT ---
    vggt_model = VGGT()
    _URL = "./pretrained/model.pt" # 保持你的硬编码路径
    
    vggt_model.load_state_dict(torch.load(_URL, map_location=device), strict=True) 
    vggt_model.eval()
    vggt_model = vggt_model.to(device)
    
    # --- Load GeSCF ---
  
    seg_model = GeSCF(args)

    if hasattr(seg_model, 'to'):
        seg_model = seg_model.to(device)
    
    print(f"[GPU {rank}] Models loaded successfully.")

    # 4. 确保保存目录存在 (多进程可能会竞争创建目录，exist_ok=True 很重要)
    os.makedirs(args.save_dir, exist_ok=True)

    # 5. 推理循环
    light = args.light
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(rank)[0] >= 8 else torch.float16
    
    # 只有 Rank 0 显示进度条，避免控制台混乱，或者每个进程都显示但带上前缀
    iterator = local_pairs
    if rank == 0: 
        iterator = tqdm(local_pairs, desc=f"Main Process (GPU 0)", total=len(local_pairs))
    else:
        # 其他GPU可以选择不显示进度条，或者简单打印
        # iterator = tqdm(local_pairs, desc=f"GPU {rank}", position=rank) 
        pass 

    for i, pair in enumerate(iterator):
        extension = os.listdir(query_dir)[0].split('.')[-1]
        if args.align:
            img0_name =  pair["img0"]
            img1_name =  pair["img0"]
        else:    
            img0_name =  pair["img0"]
            img1_name =  pair["img1"]
            
        if "." not in img0_name:
            img0_name += f".{extension}"
        if "." not in img1_name:
            img1_name += f".{extension}"
        path_t0 = os.path.join(query_dir, img0_name)
        path_t1 = os.path.join(ref_dir, img1_name)
        
        # 读取图片
        img_t0 = cv2.imread(path_t0)
        img_t1 = cv2.imread(path_t1)
        if img_t0 is None or img_t1 is None:
            print(f"[GPU {rank}] Error reading {img0_name} or {img1_name}, skipping.")
            continue

        rgb_img_t0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2RGB)
        rgb_img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB)
        
        # --- Pixel Match T0 -> T1 ---
        img_path_list = [path_t0, path_t1]
        
        right2left_match, ignore_left_mask, left2right_depth,  left2right_match, ignore_right_mask, right2left_depth, extrin1 = run_dense_match(img_path_list, vggt_model, device, dtype, light=light)
            #                                             
        W = right2left_match.shape[0]
        H = right2left_match.shape[1]
        
        project_in_range = (right2left_match[..., 0] >= 0) & (right2left_match[..., 0] < W) & \
        (right2left_match[..., 1] >= 0) & (right2left_match[..., 1] < H)
        valid_count1 = project_in_range[project_in_range == True].shape[0]
        
        ###### VGGT 反向推理 ###########
        img_path_list = [path_t1, path_t0]
        
        
        left2right_match_v2, ignore_right_mask_v2, right2left_depth_v2,  right2left_match_v2, ignore_left_mask_v2, left2right_depth_v2, extrin2 = run_dense_match(img_path_list, vggt_model, device, dtype,light=light) # t0 pixel match t1 pixel
        
        project_in_range2 = (right2left_match_v2[..., 0] >= 0) & (right2left_match_v2[..., 0] < W) & \
        (right2left_match_v2[..., 1] >= 0) & (right2left_match_v2[..., 1] < H)
        valid_count2 = project_in_range2[project_in_range2 == True].shape[0]
        
        if valid_count2 > valid_count1: # 由于VGGT的不对称性质
            right2left_match = right2left_match_v2
            ignore_left_mask = ignore_left_mask_v2
            left2right_depth = left2right_depth_v2
            left2right_match = left2right_match_v2
            ignore_right_mask = ignore_right_mask_v2
            right2left_depth = right2left_depth_v2
        
        ### right2left depth,用于筛选被遮挡区域，这个map里depth为0的区域是右图中看不到的，筛选掉，不参与sim计算
        
            # 推理 左边mask
        left_mask = seg_model(path_t0, path_t1, args, right2left_match, right2left_depth, ignore_left = ignore_left_mask)
        
        
        # 右边mask
        right_mask = seg_model(path_t1, path_t0, args, left2right_match, left2right_depth, ignore_left = ignore_right_mask)
        
        # right_mask[ignore_left_mask] = 0
        
        right2left_mask = map_right_mask_to_left(right2left_match.cpu().numpy(), right_mask)
        # 融合
        final_change_mask = left_mask | right2left_mask
        # right2left_match, ignore_left_mask, left2right_depth, _, _, _, _ = run_dense_match(
        #     img_path_list, vggt_model, device, dtype, light=light
        # ) 

        # # --- Pixel Match T1 -> T0 ---
        # img_path_list_rev = [path_t1, path_t0]
        # left2right_match, ignore_right_mask, right2left_depth, _, _, _, _ = run_dense_match(
        #     img_path_list_rev, vggt_model, device, dtype, light=light
        # )
        
        # # --- Segmentation ---

        # left_mask = seg_model(path_t0, path_t1, args, right2left_match, right2left_depth, ignore_left=ignore_left_mask)
        # right_mask = seg_model(path_t1, path_t0, args, left2right_match, left2right_depth, ignore_left=ignore_right_mask)
        
    
        # right2left_mask = map_right_mask_to_left(right2left_match.cpu().numpy(), right_mask)
        # final_change_mask = left_mask | right2left_mask
         

        final_height = min(img_t0.shape[0], img_t1.shape[0])
        final_width = min(img_t0.shape[1], img_t1.shape[1])
        final_change_mask = cv2.resize(final_change_mask, (final_width, final_height), interpolation=cv2.INTER_NEAREST)
        
        # Resize
        final_height = min(img_t0.shape[0], img_t1.shape[0])
        final_width = min(img_t0.shape[1], img_t1.shape[1])
        final_change_mask = cv2.resize(final_change_mask.astype(np.uint8), (final_width, final_height), interpolation=cv2.INTER_NEAREST).astype(bool)

        save_path = os.path.join(args.save_dir , f"{img0_name}_{img1_name}_vis.png")
        mask_path = os.path.join(args.save_dir, f"{img0_name}")
        
        cv2.imwrite(mask_path, final_change_mask * 255)

        visualize_results_custom(rgb_img_t0, rgb_img_t1, final_change_mask.astype(np.uint8), resize_hw=(final_height, final_width), save_path=save_path)
    
    print(f"[GPU {rank}] Finished.")

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Generalizable Scene Change Detection Framework (GeSCF)', add_help=add_help)

    ### Dataset / IO
    parser.add_argument("--scene-dir", type = str,  default= '../data/unaligned_changesim/Warehouse_6/Seq_0')
    parser.add_argument('--save-dir', default='../results/changesim/Warehouse_6/Seq_0', help='Directory to save prediction masks')
    parser.add_argument('--output-size', default=512, type=int, help='resize size for network input')

    ### Model config
    parser.add_argument('--test-dataset', default='Random', help='Dataset name')
    parser.add_argument('--feature-facet', default='key', help='feature-facet to intercept')
    parser.add_argument('--feature-layer', default=17, type=int, help='ViT layer to intercept featire-facet')
    parser.add_argument('--embedding-layer', default=32, type=int, help='ViT layer to intercept image-embedding & mask-embedding')

    ### SAM
    parser.add_argument('--sam-backbone', default='vit_h', help='SAM backbone')
    parser.add_argument('--points-per-side', default=32, type=int, help='SAM density')
    parser.add_argument('--pred-iou-thresh', default=0.7, type=float)
    parser.add_argument('--stability-score-thresh', default=0.7, type=float)

    ### Pseudo mask generator
    parser.add_argument('--pseudo-backbone', default='vit_h', help='Backbone for pseudo mask generation')
    
    # eval
    parser.add_argument('--dataset', 
                   type=str,
                   choices=['changesim', 'mv3dcd', 'cmu', 'pscd', 'paslcd'],
                   help='Dataset name')
    
    # custom
    parser.add_argument('--stride', default=5, help='stride')
    
    parser.add_argument('--mode', 
                   type=str,
                   default='occupy',
                   choices=['initial', 'occupy'],
                   help='Dataset name')
    
    parser.add_argument('--align', action="store_true")
    parser.add_argument('--light', action="store_true")
    parser.add_argument('--transfer', action="store_true", default=False)
    parser.add_argument('--bright_thresh', default=25.0, type=float)
    parser.add_argument('--color_thresh', default=0.2, type=float)
    parser.add_argument('--iou_thresh', default=0.65, type=float)
    parser.add_argument('--sem_filter', type=float, default=None)
    parser.add_argument('--dilate',action="store_true", default=False)
    parser.add_argument(
    '--gpus',
    type=str,
    default=None,
    help='Comma separated gpu ids, e.g. 0,1,2'
)
    return parser

if __name__ == '__main__':

    args = get_args_parser().parse_args()
    

    print("Loading dataset...")
    if args.light:
        print("Light mode: Retinex normalization enabled.")
    else:
        print("Normal mode: Retinex normalization disabled.")

    query_dir, ref_dir, img_pairs_list = load_dataset(args.dataset, args.scene_dir, args.stride)
    print(f"Total pairs to process: {len(img_pairs_list)}")
    
    # world_size = torch.cuda.device_count()
    # if world_size == 0:
    #     raise RuntimeError("No CUDA devices found!")
    if args.gpus is None:
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = [int(x) for x in args.gpus.split(',')]

    world_size = len(gpu_ids)

    if world_size == 0:
        raise RuntimeError("No CUDA devices specified or available!")
    
    print(f"Spawning processes for {world_size} GPUs...")
    
    mp.spawn(
        gpu_worker,
        args=(world_size, args, img_pairs_list, query_dir, ref_dir),
        nprocs=world_size,
        join=True
    )
    
    print("All processes completed.")