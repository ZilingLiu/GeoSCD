import os 
import numpy as np
import cv2
from tqdm import tqdm
import logging
import argparse
from framework import GeSCF
from utils import calculate_metric, visualize_results_custom
import torch
from sea_raft.core.raft import RAFT
from sea_raft.core.utils.utils import load_ckpt
from flow_match import run_flow_match
from flow_cd import json_to_args, parse_flow

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

    # 构造索引并赋值
    left_mask[valid] = right_mask[right_y[valid], right_x[valid]]

    return left_mask


def run_unaligned_cd_pairs(args, seg_model, flow_model, device):
        os.makedirs(args.save_dir, exist_ok=True)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        query_dir, ref_dir, img_pairs_list = load_dataset(args.dataset, args.scene_dir, args.stride)
        # data loader，不同的数据集不同的load结构
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
            
            # 读取图像并转换颜色空间
            img_t0 = cv2.imread(path_t0)
            img_t1 = cv2.imread(path_t1)
            rgb_img_t0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2RGB)
            rgb_img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB)
            right2left_match = run_flow_match(rgb_img_t0, rgb_img_t1, flow_model, device, dtype) # t0 pixel match t1 pixel
            ignore_left_mask = None
            left2right_depth = None
            
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            
            left2right_match = run_flow_match(rgb_img_t1, rgb_img_t0, flow_model, device, dtype) # t0 pixel match t1 pixel
            ignore_right_mask = None
            right2left_depth = None
        
            ### right2left depth,用于筛选被遮挡区域，这个map里depth为0的区域是右图中看不到的，筛选掉，不参与sim计算
        
            left_mask = seg_model(path_t0, path_t1, args, right2left_match, right2left_depth, ignore_left = ignore_left_mask)
            
            
            if args.dataset == "cmu" or args.dataset == "pscd":  # pscd 只用预测左边mask，然后用t0 mask评估
                final_change_mask = left_mask
            else:
                # 右边mask
                right_mask = seg_model(path_t1, path_t0, args, left2right_match, left2right_depth, ignore_left = ignore_right_mask)
                
                right2left_mask = map_right_mask_to_left(right2left_match.cpu().numpy(), right_mask)
                # 融合
                final_change_mask = left_mask | right2left_mask
            
            final_height = min(img_t0.shape[0], img_t1.shape[0])
            final_width = min(img_t0.shape[1], img_t1.shape[1])
            final_change_mask = cv2.resize(final_change_mask, (final_width, final_height), interpolation=cv2.INTER_NEAREST)
            
    
            save_path = os.path.join(args.save_dir , f"{img0_name}_{img1_name}_vis.png")
            mask_path = os.path.join(args.save_dir, f"{img0_name}")
           
            if args.dataset == "cmu":
                black_mask = np.all(rgb_img_t0 == [0, 0, 0], axis=-1) 
                final_change_mask[black_mask] = 0
            
            cv2.imwrite(mask_path, final_change_mask * 255)
            visualize_results_custom(rgb_img_t0, rgb_img_t1, final_change_mask, resize_hw = (final_height, final_width), save_path=save_path)
            
       

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Generalizable Scene Change Detection Framework (GeSCF)', add_help=add_help)

    ### Dataset / IO
    parser.add_argument("--scene-dir", type = str,  default= '../data/unaligned_changesim/Warehouse_6/Seq_0')
    parser.add_argument('--save-dir', default='results/bi_direct/changesim/Warehouse_6/Seq_0', help='Directory to save prediction masks')
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
    parser.add_argument('--stride', default=5, help='stride')
    
    parser.add_argument('--mode', 
                   type=str,
                   default='initial',
                   choices=['initial', 'occupy'],
                   help='Dataset name')
    
    parser.add_argument('--align', action="store_true")
    
    parser.add_argument('--cfg', help='experiment configure file name', type=str,  default="sea_raft/config/eval/spring-M.json")
    parser.add_argument('--flow_path', help='checkpoint path', type=str, default="sea_raft/models/Tartan-C-T-TSKH-spring540x960-M.pth")

    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    # evaluate_image_directory(args)
    seg_model = GeSCF(args)
    print("Sam Model loaded!")
    device = "cuda"
    flow_args = parse_flow(args.cfg)
    model = RAFT(flow_args)
    load_ckpt(model, args.flow_path)    
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()
    
    print(f"Optical Flow Model loaded!")
    
    
    run_unaligned_cd_pairs(args, seg_model, model, device)
