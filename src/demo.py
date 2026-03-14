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
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from pixel_match import run_dense_match
import torch.nn.functional as F
from framework import overlay_mask_on_image

logging.basicConfig(
    level=logging.INFO,               
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.JPG', '.PNG']

def simple_retinex(img, sigma_list=[15, 80, 250]):
    img = img.astype(np.float32) + 1.0
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        retinex += np.log(img) - np.log(blur + 1.0)
    retinex = retinex / len(sigma_list)
    retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex))
    retinex = (retinex * 255).astype(np.uint8)
    return retinex

def map_right_mask_to_left(right2left_match: np.ndarray, right_mask: np.ndarray):
    
    H, W = right_mask.shape
    left_mask = np.zeros((H, W), dtype=bool)

    right_x = right2left_match[..., 0].astype(np.int64)
    right_y = right2left_match[..., 1].astype(np.int64)

    valid_x = (right_x >= 0) & (right_x < W)
    valid_y = (right_y >= 0) & (right_y < H)
    valid = valid_x & valid_y


    left_mask[valid] = right_mask[right_y[valid], right_x[valid]]

    return left_mask

def mutual_consistency_mask(left2right, right2left, tol=1.0):
   
    H, W, _ = left2right.shape
    left2right = left2right.float()
    right2left = right2left.float()
    
   
    y_L, x_L = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    coords_L = torch.stack([x_L, y_L], dim=-1).float().to(left2right.device)  # (H, W, 2)

  
    coords_R = left2right  # (H, W, 2)

   
    coords_R_norm = coords_R.clone()
    coords_R_norm[..., 0] = coords_R_norm[..., 0] / (W - 1) * 2 - 1
    coords_R_norm[..., 1] = coords_R_norm[..., 1] / (H - 1) * 2 - 1

    right2left_reshaped = right2left.permute(2, 0, 1).unsqueeze(0)  # (1, 2, H, W)
    coords_L2 = F.grid_sample(
        right2left_reshaped, coords_R_norm.unsqueeze(0), 
        align_corners=True, mode='bilinear'
    ).squeeze(0).permute(1, 2, 0)  # (H, W, 2)

    error = torch.norm(coords_L2 - coords_L, dim=-1)  # (H, W) float

    error_norm = (error - error.min()) / (error.max() - error.min() + 1e-8)
    error_vis = (error_norm * 255).cpu().numpy().astype('uint8')  # uint8 灰度图


    error_vis_color = cv2.applyColorMap(error_vis, cv2.COLORMAP_JET)

    mask = error <= tol

    return mask, error, error_vis_color


def find_matching_file(basename, directory):
  
    for ext in IMG_EXTENSIONS:
        candidate = os.path.join(directory, basename + ext)
        if os.path.exists(candidate):
            return candidate
    return None

def run_unaligned_cd(args, seg_model, vggt_model, device, light = False):
        path_t0 = args.path_t0
        path_t1 = args.path_t1
      
      
        img_t0 = cv2.imread(path_t0)
        img_t1 = cv2.imread(path_t1)
        rgb_img_t0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2RGB)
        rgb_img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB)
        
        img_path_list = [path_t0, path_t1]
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        os.makedirs(args.save_dir, exist_ok=True)
        right2left_match, ignore_left_mask, left2right_depth, _, _, _, _ = run_dense_match(img_path_list, vggt_model, device, dtype, 
                                                                             light=light) # t0 pixel match t1 pixel

        img_path_list = [path_t1, path_t0]
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        os.makedirs(args.save_dir+"_right", exist_ok=True)
        left2right_match, ignore_right_mask, right2left_depth, _, _, _, _ = run_dense_match(img_path_list, vggt_model, device, dtype, 
                                                                                light=light) # t0 pixel match t1 pixel
        
        left_mask = seg_model(path_t0, path_t1, args, right2left_match, right2left_depth, ignore_left = ignore_left_mask, 
                              save_dir = args.save_dir, debug = args.debug)
        if args.dataset == "cmu":
            final_change_mask = left_mask
        else:
          
            right_mask = seg_model(path_t1, path_t0, args,left2right_match, left2right_depth, ignore_left = ignore_right_mask ,save_dir = args.save_dir + "_right", debug = args.debug)
            
            right2left_mask = map_right_mask_to_left(right2left_match.cpu().numpy(), right_mask)
       
            final_change_mask = left_mask | right2left_mask
        
        final_height = min(img_t0.shape[0], img_t1.shape[0])
        final_width = min(img_t0.shape[1], img_t1.shape[1])
        final_change_mask = cv2.resize(final_change_mask, (final_width, final_height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(args.save_dir , f"final_change_mask.png"), final_change_mask * 255)
        if args.path_gt is not None:
            gt = cv2.imread(args.path_gt, 0) / 255.
            if args.dataset == "pscd":
                step = (1024 - 224) // (15 - 1)
                gt[:, : step] = 0
            gt = cv2.resize(gt, (final_width, final_height))
          
            precision, recall = calculate_metric(gt, final_change_mask)
            f1score = 2 * (precision * recall) / (precision + recall + 1e-9)
            logging.info(f"mode_{args.mode}, f1: {f1score}")
            if args.debug:
                visualize_results_custom(rgb_img_t0, rgb_img_t1, final_change_mask, save_path= os.path.join(args.save_dir, 'visualization.png'))
        
        
      
def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Generalizable Scene Change Detection Framework (GeSCF)', add_help=add_help)

    ### Dataset / IO
    parser.add_argument('--path-t0', default='../data/PASLCD/Printing_area/Instance_1/images/Inst_1_test_IMG_2211.jpg', help='Path to t0 image directory')
    parser.add_argument('--path-t1', default='../data/PASLCD/Printing_area/Instance_1/images/Inst_1_IMG_2270.jpg', help='Path to t1 image directory')
    parser.add_argument('--path-gt', default="../data/PASLCD/Printing_area/Instance_1/gt_mask/Inst_1_test_IMG_2211.png", help='Path to ground-truth mask directory (optional)')
    parser.add_argument('--save-dir', default='../demo_exp/Cantina', help='Directory to save prediction masks')
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
    parser.add_argument('--bright_thresh', default=25.0, type=float)
    parser.add_argument('--color_thresh', default=0.2, type=float)
    parser.add_argument('--iou_thresh', default=0.65, type=float)
    parser.add_argument('--sem_filter', type=float, default=None)
    parser.add_argument('--dilate',action="store_true", default=False)
    
    
    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    # evaluate_image_directory(args)
    seg_model = GeSCF(args)
    print("Sam Model loaded!")
    device = "cuda"
    vggt_model = VGGT()
    _URL = "./pretrained/model.pt" 
    vggt_model.load_state_dict(torch.load(_URL, map_location=device), strict=True)
    
    vggt_model.eval()
    vggt_model = vggt_model.to(device)
    
    print(f"VGGT Model loaded!")
    
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    run_unaligned_cd(args, seg_model, vggt_model, device, args.light)
