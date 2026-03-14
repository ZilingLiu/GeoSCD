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
import sys

from sea_raft.core.raft import RAFT
from sea_raft.core.utils.utils import load_ckpt

from pixel_match import run_dense_match
from flow_match import run_flow_match
import torch.nn.functional as F
from framework import overlay_mask_on_image
import json
# 日志设置
logging.basicConfig(
    level=logging.INFO,               
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.JPG', '.PNG']


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



def find_matching_file(basename, directory):
    """在指定目录下根据文件名（忽略扩展名）寻找匹配文件"""
    for ext in IMG_EXTENSIONS:
        candidate = os.path.join(directory, basename + ext)
        if os.path.exists(candidate):
            return candidate
    return None

def run_unaligned_cd(args, seg_model, flow_model, device):
        path_t0 = args.path_t0
        path_t1 = args.path_t1
      
        # 读取图像并转换颜色空间
        img_t0 = cv2.imread(path_t0)
        img_t1 = cv2.imread(path_t1)
        rgb_img_t0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2RGB)
        rgb_img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB)
        
        
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        os.makedirs(args.save_dir, exist_ok=True)
        right2left_match = run_flow_match(rgb_img_t0, rgb_img_t1, flow_model, device, dtype, debug=args.debug, mask_path = args.mask_path) # t0 pixel match t1 pixel
        ignore_left_mask = None
        left2right_depth = None
       
        # 推理

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        os.makedirs(args.save_dir+"_right", exist_ok=True)
        left2right_match = run_flow_match(rgb_img_t1, rgb_img_t0, flow_model, device, dtype, debug=args.debug) # t0 pixel match t1 pixel
        ignore_right_mask = None
        right2left_depth = None
      
        ### right2left depth,用于筛选被遮挡区域，这个map里depth为0的区域是右图中看不到的，筛选掉，不参与sim计算
    
        left_mask = seg_model(path_t0, path_t1, args, right2left_match, right2left_depth, ignore_left = ignore_left_mask, 
                              save_dir = args.save_dir, debug = args.debug)
        if args.dataset == "cmu" or args.dataset == "pscd":  # pscd 只用预测左边mask，然后用t0 mask评估
            final_change_mask = left_mask
        else:
            # 右边mask
            right_mask = seg_model(path_t1, path_t0, args, left2right_match, left2right_depth, ignore_left = ignore_right_mask ,save_dir = args.save_dir + "_right", debug = args.debug)
            
            right2left_mask = map_right_mask_to_left(right2left_match.cpu().numpy(), right_mask)
            # 融合
            final_change_mask = left_mask | right2left_mask
        
        final_height = min(img_t0.shape[0], img_t1.shape[0])
        final_width = min(img_t0.shape[1], img_t1.shape[1])
        final_change_mask = cv2.resize(final_change_mask, (final_width, final_height), interpolation=cv2.INTER_NEAREST)

        if args.path_gt is not None:
            gt = cv2.imread(args.path_gt, 0) / 255.
            gt = cv2.resize(gt, (final_width, final_height))
          
            precision, recall = calculate_metric(gt, final_change_mask)
            f1score = 2 * (precision * recall) / (precision + recall + 1e-9)
            logging.info(f"mode_{args.mode}, f1: {f1score}")
            if args.debug:
                visualize_results_custom(rgb_img_t0, rgb_img_t1, final_change_mask, save_path= os.path.join(args.save_dir, 'visualization.png'))
        # mask merging project right mask to left mask
        
        
      
def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Generalizable Scene Change Detection Framework (GeSCF)', add_help=add_help)

    ### Dataset / IO
    parser.add_argument('--path-t0', default='../data/Cantina/images/Inst_1_test_IMG_2865.jpg', help='Path to t0 image directory')
    parser.add_argument('--path-t1', default='../data/Cantina/images/Inst_1_IMG_3026.jpg', help='Path to t1 image directory')
    parser.add_argument('--path-gt', default=None, help='Path to ground-truth mask directory (optional)')
    parser.add_argument('--save-dir', default='predictions/Cantina', help='Directory to save prediction masks')
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
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    parser.add_argument('--dataset', 
                   type=str,
                   choices=['changesim', 'mv3dcd', 'cmu', 'pscd', 'paslcd'],
                   help='Dataset name')
    
    parser.add_argument('--mode', 
                   type=str,
                   choices=['initial', 'occupy'],
                   help='Dataset name')
    parser.add_argument('--mask_path', type = str, help='')
    
    parser.add_argument('--light', action='store_true', help='use light aug')
    
    # flow model
    parser.add_argument('--cfg', help='experiment configure file name', type=str,  default="sea_raft/config/eval/spring-M.json")
    parser.add_argument('--flow_path', help='checkpoint path', type=str, default="sea_raft/models/Tartan-C-T-TSKH-spring540x960-M.pth")
    
    return parser

def json_to_args(json_path):
    # return a argparse.Namespace object
    with open(json_path, 'r') as f:
        data = json.load(f)
    args = argparse.Namespace()
    args_dict = args.__dict__
    for key, value in data.items():
        args_dict[key] = value
    return args

def parse_flow(json_path):
    args = json_to_args(json_path)
    return args

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
    
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    run_unaligned_cd(args, seg_model, model, device)
