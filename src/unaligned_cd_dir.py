import os 
import numpy as np
import cv2
from tqdm import tqdm
import logging
import argparse
import matplotlib.pyplot as plt

from framework import GeSCF
from utils import calculate_metric, visualize_results_custom
import torch
from vggt.models.vggt import VGGT

from pixel_match import run_dense_match
from datasets.load_dataset import load_dataset
import time

logging.basicConfig(
    level=logging.INFO,               
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


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

    Args:
        right2left_match (np.ndarray): (H, W, 2)，表示 Left 上每个像素在 Right 上的对应坐标 (x, y)
        right_mask (np.ndarray): (H, W)，Right 图像上的 bool mask 或 0/1 mask

    Returns:
        left_mask (np.ndarray): (H, W)，Left 图像上的 bool mask
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


def run_unaligned_cd_pairs(args, seg_model, vggt_model, device, light = False):
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        os.makedirs(args.save_dir, exist_ok=True)
        if light:
            print("Light mode: Retinex normalization enabled.")
        else:
            print("Normal mode: Retinex normalization disabled.")

        query_dir, ref_dir, img_pairs_list = load_dataset(args.dataset, args.scene_dir, args.stride)
        
        for i, pair in tqdm(enumerate(img_pairs_list), total=len(img_pairs_list), desc="Processing image pairs"):
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
            
            img_t0 = cv2.imread(path_t0)
            img_t1 = cv2.imread(path_t1)
            rgb_img_t0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2RGB)
            rgb_img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB)
                                
            # mask merging project right mask to left mask
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
          
            img_path_list = [path_t0, path_t1]
            right2left_match, ignore_left_mask, left2right_depth, _, _, _, _ = run_dense_match(img_path_list, vggt_model, device, dtype, 
                                                                             light=light) # t0 pixel match t1 pixel

            img_path_list = [path_t1, path_t0]
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            
            left2right_match, ignore_right_mask, right2left_depth, _, _, _, _ = run_dense_match(img_path_list, vggt_model, device, dtype, 
                                                                                    light=light) # t0 pixel match t1 pixel
            left_mask = seg_model(path_t0, path_t1, args, right2left_match, right2left_depth, ignore_left = ignore_left_mask)


      
            right_mask = seg_model(path_t1, path_t0, args,left2right_match, left2right_depth, ignore_left = ignore_right_mask)
            
            right2left_mask = map_right_mask_to_left(right2left_match.cpu().numpy(), right_mask)
  
            final_change_mask = left_mask | right2left_mask
          
            final_height = min(img_t0.shape[0], img_t1.shape[0])
            final_width = min(img_t0.shape[1], img_t1.shape[1])
         

            
            save_path = os.path.join(args.save_dir , f"{img0_name}_{img1_name}_vis.png")
            mask_path = os.path.join(args.save_dir, f"{img0_name}")
           
            
            cv2.imwrite(mask_path, final_change_mask * 255)
            visualize_results_custom(rgb_img_t0, rgb_img_t1, final_change_mask, resize_hw = (final_height, final_width), save_path=save_path)
            
   

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Generalizable Scene Change Detection Framework (GeSCF)', add_help=add_help)

    ### Dataset / IO
    parser.add_argument("--scene-dir", type = str,  default= "")
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save prediction masks')
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
   
    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    # evaluate_image_directory(args)
    seg_model = GeSCF(args)
    print("Sam Model loaded!")
    device = "cuda"
    vggt_model = VGGT()
    _URL = "pretrained/model.pt"
    vggt_model.load_state_dict(torch.load(_URL, map_location=device), strict=True)
    
    vggt_model.eval()
    vggt_model = vggt_model.to(device)
    
    print(f"VGGT Model loaded!")
    
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    run_unaligned_cd_pairs(args, seg_model, vggt_model, device, light = args.light)
