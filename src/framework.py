"""
Generalizable Scene Change Detection Framework (GeSCF)
"""

import logging
import cv2
import numpy as np
from scipy.stats import skew

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import os
# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Internal utilities
from utils import calculate_iou, sanity_check_args, load_backbones

# Modules
from segment_anything_model import SamAutomaticMaskGenerator
from pseudo_generator import PseudoGenerator
from registration import coarse_transform
import torch.nn.functional as F
from PIL import Image

def overlay_mask_on_image(mask: np.ndarray, rgb_path: str, output_path: str = "output.png",
                          color=(255, 192, 203), alpha=0.5):
    """
    将一个二值mask resize到RGB图像大小，叠加显示并保存结果图像。
    """
    from PIL import Image
    import numpy as np

    # 加载原图像
    rgb_img = Image.open(rgb_path).convert("RGB")
    rgb_w, rgb_h = rgb_img.size

    # 将 mask 转为 uint8 并 resize
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
    mask_img = mask_img.resize((rgb_w, rgb_h), resample=Image.NEAREST)
    mask_resized = np.array(mask_img) > 0

    # 创建输出数组
    rgb_np = np.array(rgb_img).astype(np.float32)
    blended = rgb_np.copy()

    # 只在 mask 区域进行混合
    blended[mask_resized] = (
        (1 - alpha) * rgb_np[mask_resized] + alpha * np.array(color, dtype=np.float32)
    )

    # 保存结果
    blended_img = Image.fromarray(blended.astype(np.uint8))
    blended_img.save(output_path)
    print(f"保存完成：{output_path}")
    
    
# 保存全分割图片
def show_anns_on_image(anns, image, save_path):
    """
    在给定图像上绘制分割掩码，并保存为文件

    Args:
        anns (list): 掩码列表，格式为 [{'segmentation': np.ndarray, 'area': int}, ...]
        image (np.ndarray): 原图，形状为 (H, W, 3)
        save_path (str): 保存路径，支持 .png/.jpg 等格式
    """
    if len(anns) == 0:
        print("[Warning] No annotations to draw.")
        return

    # 按面积从大到小排序，确保大的 mask 不被小的盖住
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)

    # 创建 RGBA 图像用于绘制 mask
    overlay = np.ones((*image.shape[:2], 4), dtype=np.float32)
    overlay[:, :, 3] = 0  # 初始化 alpha 通道为透明

    for ann in sorted_anns:
        m = ann['segmentation']  # 二值 mask，形状 (H, W)
        color_mask = np.concatenate([np.random.rand(3), [0.35]])  # RGBA
        overlay[m] = color_mask  # 将 mask 区域涂上颜色

    H, W = image.shape[:2]
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)  # 或者 dpi=300
    ax.imshow(image)
    ax.imshow(overlay)
    ax.axis('off')
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
   
    
def match_embed(embed_t0, embed_t1, matching):
    """
    给定 t0 -> t1 的 matching 坐标，计算 t0 的每个像素和其在 t1 中对应特征的 cosine similarity。
    同时输出 mask 过滤掉那些越界坐标。

    Args:
        embed_t0 (Tensor): [H, W, C] 特征图
        embed_t1 (Tensor): [H, W, C] 特征图
        matching (Tensor): [H, W, 2] 每个 pixel 在 t1 中的对应坐标 (x, y)

    Returns:
        cos_sim_map: [H, W] cosine similarity map
        valid_mask: [H, W] 超出边界的为 False，其余为 True
    """
    H, W, C = embed_t0.shape

    # 将 matching 拆成坐标
    x_map = matching[..., 0]
    y_map = matching[..., 1]

    # mask: 超出边界的地方为 False
    valid_mask = (
        (x_map >= 0) & (x_map < W - 1) &  # 为双线性采样预留边界
        (y_map >= 0) & (y_map < H - 1)
    )

    # 归一化到 [-1, 1]，用于 grid_sample
    norm_x = (x_map / (W - 1)) * 2 - 1
    norm_y = (y_map / (H - 1)) * 2 - 1
    grid = torch.stack((norm_x, norm_y), dim=-1)  # [H, W, 2]
    grid = grid.unsqueeze(0)  # [1, H, W, 2]

    # 特征转为 [1, C, H, W]
    embed_t1 = embed_t1.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

    # 采样出 matching 对应位置的特征
    sampled_feat = F.grid_sample(embed_t1, grid, mode='bilinear', align_corners=True)[0]  # [C, H, W]
    sampled_feat = sampled_feat.permute(1, 2, 0)  # [H, W, C]

    # 原始特征也 reshape
    embed_t0 = embed_t0.reshape(H * W, C)
    sampled_feat = sampled_feat.reshape(H * W, C)

    # cosine similarity
    cos_sim = F.cosine_similarity(embed_t0, sampled_feat, dim=1)
    cos_sim_map = cos_sim.reshape(H, W)

    return cos_sim_map, valid_mask

def match_multihead_key_avg(key_t0, key_t1, matching):
    """
    对 multi-head key 特征，计算每个 head 上的 cosine similarity，然后取平均。

    Args:
        key_t0 (Tensor): [B, H, W, C] t0 的 multi-head key（B 是 head 数）
        key_t1 (Tensor): [B, H, W, C] t1 的 multi-head key
        matching (Tensor): [H, W, 2] 每个像素在 t1 中的匹配坐标 (x, y)

    Returns:
        cos_sim_map_avg: [H, W] 所有 head 平均 cosine similarity
        valid_mask: [H, W] 有效像素 mask
    """
    B, H, W, C = key_t0.shape

    # 1. 构造 valid mask
    x_map = matching[..., 0]
    y_map = matching[..., 1]
    valid_mask = (
        (x_map >= 0) & (x_map < W - 1) &
        (y_map >= 0) & (y_map < H - 1)
    )

    # 2. 构造 grid_sample 所需的归一化 grid
    norm_x = (x_map / (W - 1)) * 2 - 1
    norm_y = (y_map / (H - 1)) * 2 - 1
    grid = torch.stack((norm_x, norm_y), dim=-1)  # [H, W, 2]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)    # [B, H, W, 2]

    # 3. grid_sample 采样 t1 中的匹配位置特征
    key_t1 = key_t1.permute(0, 3, 1, 2)  # [B, C, H, W]
    sampled_feat = F.grid_sample(key_t1, grid, mode='bilinear', align_corners=True)  # [B, C, H, W]
    sampled_feat = sampled_feat.permute(0, 2, 3, 1)  # [B, H, W, C]

    # 4. cosine similarity per head
    key_t0_flat = key_t0.reshape(B, -1, C)          # [B, H*W, C]
    sampled_flat = sampled_feat.reshape(B, -1, C)   # [B, H*W, C]
    cos_sim = F.cosine_similarity(key_t0_flat, sampled_flat, dim=-1)  # [B, H*W]

    cos_sim_map = cos_sim.view(B, H, W)             # [B, H, W]

    # 5. 取所有 head 的平均值
    cos_sim_map_avg = cos_sim_map.mean(dim=0)       # [H, W]

    return cos_sim_map_avg, valid_mask

def visualize_sim_map(cos_sim_map: torch.Tensor, save_path: str = "sim_map.png", cmap: str = "plasma"):
    """
    可视化范围为 [0, 1] 的 similarity map。
    
    Args:
        cos_sim_map (torch.Tensor): [H, W] tensor，范围在 [0, 1]，可以在 GPU 上。
        save_path (str): 保存路径。
        cmap (str): matplotlib 的 colormap（如 'plasma', 'viridis', 'jet'）。
    """
    # 如果在 CUDA 上，转到 CPU
    if cos_sim_map.is_cuda:
        cos_sim_map = cos_sim_map.cpu()

    # 转为 numpy
    sim_map_np = cos_sim_map.numpy()

    # 显示并保存
    plt.figure(figsize=(6, 6))
    plt.imshow(sim_map_np, cmap=cmap, vmin=0.0, vmax=1.0)
    plt.colorbar(label="Similarity")
    plt.title("Cosine Similarity Map")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

class GeSCF(nn.Module):
    def __init__(self, args):
        sanity_check_args(args)
        super(GeSCF, self).__init__()

        # Dataset settings
        self.dataset = args.test_dataset
     
        self.img_size = (512, 512)
        self.output_size = (args.output_size, args.output_size)
        
        logging.info(f'dataset name: {self.dataset}')

        # Feature extraction settings
        self.feature_facet = args.feature_facet
        self.feature_layer = args.feature_layer
        self.embedding_layer = args.embedding_layer

        # Default hyperparameters
        self.z_value = -0.52
        self.Ni = -0.2
        self.Nj = 0.2
        self.alpha_t = 0.65
        self.cosine_thr = 0.6
        
        # Build SAM and pseudo backbone
        self.sam_backbone, self.pseudo_backbone = load_backbones(args.sam_backbone, args.pseudo_backbone)
    
        # Build automatic mask generator
        self.automatic_mask_generator = SamAutomaticMaskGenerator(
            model=self.sam_backbone,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=0
        )

        # Build pseudo generator
        self.pseudo_generator = PseudoGenerator(
            feature_layer=self.feature_layer,
            embedding_layer=self.embedding_layer,
            img_size=self.img_size,
            backbone=self.pseudo_backbone
        )
        
    
    def load_img(self, img_path):
        """Load and preprocess an image (RGB and grayscale)."""
        
        # Load and resize RGB image
        bgr_img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, self.img_size)
        rgb_img = np.array(rgb_img)

        # Load and resize grayscale image
        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gray_img = cv2.resize(gray_img, self.img_size) / 255.
        gray_img = np.array(gray_img)

        # Transform RGB image to tensor
        input_tensor = self.transform()(rgb_img).unsqueeze(0)

        return rgb_img, gray_img, input_tensor


    def transform(self):
        """Return composed torchvision transform for RGB image preprocessing."""
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self.img_size),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])
        
        
    def get_skewness_and_type(self, key, query, value, img_t1, flag):
        """Calculate skewness, type, and moderate region mask based on feature similarity map."""

        # Select similarity feature map
        if self.feature_facet == 'key':
            sim_map = key
        elif self.feature_facet == 'query':
            sim_map = query
        else:
            sim_map = value

        sim_map = sim_map.detach().cpu().numpy()[0]

        # Flatten relevant regions based on flag
        if flag:
            oov_idx = np.where(np.any(img_t1 != [0, 0, 0], axis=-1))
            flat_sim_map = sim_map[oov_idx].flatten()
        else:
            oov_idx = None
            flat_sim_map = sim_map.flatten()

        # Compute skewness
        skewness = skew(flat_sim_map)

        # Determine skewness type
        if skewness >= self.Nj:
            skew_type = 'Right-skewed'
        elif skewness <= self.Ni:
            skew_type = 'Left-skewed'
        else:
            skew_type = 'Moderate'

        # Generate moderate region mask using z-score thresholding
        mean = np.mean(sim_map)
        std = np.std(sim_map)
        z_score = (sim_map - mean) / std
        moderate_mask = (z_score < self.z_value).astype(np.float32)

        return skewness, skew_type, sim_map, flat_sim_map, moderate_mask, oov_idx

    
    def get_skewness_and_type_from_mask(self, sim_map, valid_mask):
        """
        Calculate skewness, type, and moderate region mask from a similarity map,
        using only the valid pixels provided in the valid_mask.

        Args:
            sim_map (torch.Tensor or np.ndarray): Similarity map of shape (H, W) or (1, H, W)
            valid_mask (torch.Tensor or np.ndarray): Boolean or binary mask of same spatial size as sim_map.

        Returns:
            skewness (float): Skewness of valid similarity values.
            skew_type (str): 'Right-skewed', 'Left-skewed', or 'Moderate'.
            sim_map (np.ndarray): Full similarity map as numpy array.
            flat_sim_map (np.ndarray): Flattened valid similarity values.
            moderate_mask (np.ndarray): Binary mask of moderate region (same shape as sim_map).
        """
        # Convert to numpy if needed
        if isinstance(sim_map, torch.Tensor):
            sim_map = sim_map.detach().cpu().numpy()
        if isinstance(valid_mask, torch.Tensor):
            valid_mask = valid_mask.detach().cpu().numpy()

        # Handle shape (1, H, W)
        if sim_map.ndim == 3:
            sim_map = sim_map[0]

        # Select valid pixels
        flat_sim_map = sim_map[valid_mask > 0].flatten()

        # Compute skewness
        skewness = skew(flat_sim_map)

        # Determine skewness type
        if skewness >= self.Nj:
            skew_type = 'Right-skewed'
        elif skewness <= self.Ni:
            skew_type = 'Left-skewed'
        else:
            skew_type = 'Moderate'

        # Z-score thresholding to get moderate mask
        mean = np.mean(sim_map)
        std = np.std(sim_map)
        z_score = (sim_map - mean) / std
        # moderate_mask = (z_score < self.z_value).astype(np.float32)
        moderate_mask = (z_score < 0.1).astype(np.float32)
        moderate_mask[valid_mask == False] = 0
        return skewness, skew_type, sim_map, flat_sim_map, moderate_mask

    def threshold(self, skewness):
        """Calculate dynamic threshold based on image size and skewness type."""
        h = self.img_size[0]
        w = self.img_size[1]
        
        # Threshold parameters
        b_left = 0.5
        b_right = 0.05
        s_left = 1.0
        s_right = 0.1
        mu = 2.5e5 / (h * w)
        c = 1.0 / mu**3

        # Threshold calculation based on skewness
        if skewness >= self.Nj:
            threshold = b_right + s_right * skewness * c
        elif skewness <= self.Ni:
            threshold = b_left - s_left * skewness * c
        else:
            threshold = 0.0

        return threshold
    
    
    def adaptive_threshold_function(self, flat_sim_map, skewness):
        """Detect outliers in the similarity map using MAD-based adaptive thresholding."""

        # Compute Median Absolute Deviation (MAD)
        median = np.median(flat_sim_map)
        mad = np.median(np.abs(flat_sim_map - median))

        # Compute modified z-scores
        modified_z_scores = 0.6745 * (flat_sim_map - median) / mad

        # Determine threshold from skewness
        threshold = self.threshold(skewness)

        # Identify outliers
        outliers = modified_z_scores < (-1.0 * threshold)

        return outliers
    

    
    def forward(self, img_t0_path, img_t1_path, args, pixel_match = None, 
                right2left_depth = None, ignore_left = None,  save_dir = None, debug = False):
        '''Generate final change mask from a pair of input images.'''
    
        # Load and preprocess input images, 直接resize到(512, 512)
        img_t0, gray_img_t0, input_t0 = self.load_img(img_t0_path)
        img_t1, gray_img_t1, input_t1 = self.load_img(img_t1_path)
        H, W = img_t1.shape[:2]
        H = int(H)
        W = int(W)
        # Generate class-agnostic object proposals
        masks_t0 = self.automatic_mask_generator.generate(img_t0)
        masks_t1 = self.automatic_mask_generator.generate(img_t1)

        if debug:
            show_anns_on_image(masks_t0, img_t0, os.path.join(save_dir, "mask1.png"))
            show_anns_on_image(masks_t1, img_t1, os.path.join(save_dir, "mask2.png"))
        

        ####################################################
        # Initial Pseudo Mask Generation
        ####################################################
        
        # Feature embedding and similarity
        inputs = torch.cat([input_t0, input_t1], dim=1).to(device='cuda')
        embed_t0, embed_t1, key_t0, key_t1 = self.pseudo_generator(inputs, save_key = True)
        
        ##根据pixel match 投影embedding，再和t0算similarity，pixel match需要和sam的rensize对应上
        # embed_sim, valid_mask = match_embed(embed_t0, embed_t1, pixel_match)
        embed_sim, valid_mask = match_multihead_key_avg(key_t0, key_t1, pixel_match)
        ### analyze sim distribution
        skewness, type, sim_map, flat_sim_map, moderate_mask = self.get_skewness_and_type_from_mask(embed_sim, valid_mask)
        outliers = self.adaptive_threshold_function(flat_sim_map, skewness)
        binary_mask_outliers = np.zeros_like(sim_map, dtype=np.uint8)

        # 将有效区域内的 outliers 映射回 sim_map 的空间
        binary_mask_outliers[valid_mask.cpu().numpy()] = outliers.astype(np.uint8)
        
        # outliers_reshaped = outliers.reshape(self.img_size)
        # binary_mask_outliers[outliers_reshaped] = 1  # 已经筛选出了 coarse，后面还需一些图像学处理
        
        # padding_size = 10
        # kernel_size = 2 * padding_size + 1
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # warped_oov_mask_uint8 = warped_oov_mask.astype(np.uint8) * 255
        # dilated_mask = cv2.dilate(warped_oov_mask_uint8, kernel, iterations=1).astype(bool)

        # binary_mask_outliers[dilated_mask] = 0
        # moderate_mask[dilated_mask] = 0
        
        if type in ['Left-skewed', 'Right-skewed']:
            initial_pseudo_mask = binary_mask_outliers
        else:
            initial_pseudo_mask = moderate_mask
        
        # Refine initial pseudo mask (small region removal and morphological smoothing)
        initial_pseudo_mask = initial_pseudo_mask.astype(np.uint8)
        
        # before occupy
        if debug:
            os.makedirs(save_dir, exist_ok=True)
            overlay_mask_on_image(initial_pseudo_mask, img_t0_path, os.path.join(save_dir, "before_occupy.png"))
        
        ### filter the occupied area
        if right2left_depth is not None:
            occupy_mask = right2left_depth == 0
            occupy_mask = occupy_mask.cpu().numpy()
        
        # initial_pseudo_mask[occupy_mask]  = 0
        
        if args.mode == "occupy":
            # print(f"occ")
            initial_pseudo_mask[ignore_left] = 0
            
        if debug:
            overlay_mask_on_image(initial_pseudo_mask, img_t0_path, os.path.join(save_dir, "before_graph.png"))
       
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 小核避免过度膨胀
        initial_pseudo_mask = cv2.morphologyEx(initial_pseudo_mask, cv2.MORPH_DILATE, kernel, iterations=1)
        
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        # initial_pseudo_mask = cv2.morphologyEx(initial_pseudo_mask, cv2.MORPH_OPEN, kernel)
        if debug:
            overlay_mask_on_image(initial_pseudo_mask, img_t0_path, os.path.join(save_dir, "after_graph.png"))
        # cv2.imwrite(os.path.join(save_dir, "after_graph.png"), initial_pseudo_mask * 255)

        contours, _ = cv2.findContours(initial_pseudo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  # filter out small area
                cv2.drawContours(initial_pseudo_mask, [cnt], -1, (0, 0, 0), thickness=cv2.FILLED)
        if debug:
            overlay_mask_on_image(initial_pseudo_mask, img_t0_path, os.path.join(save_dir, "after_contours_filter.png"))        
     
        
        mask_idx_t0 = []
        all_iou = []
        
        for i in range(len(masks_t0)):
            iou, overlap_mask = calculate_iou(initial_pseudo_mask, masks_t0[i]['segmentation'])
            if float(iou) > 0:
                all_iou.append(float(iou))
                
            
            if iou >= args.iou_thresh:  
                overlap_mask = overlap_mask.astype(bool)  # NumPy 转 bool
                overlap_mask = torch.from_numpy(overlap_mask).bool().to(embed_t0.device)  # 显式转换
                mask_embedding_t0 = embed_t0[overlap_mask].mean(axis=0)
                mask_embedding_t1 = embed_t1[overlap_mask].mean(axis=0)
                
                matching_right_xy = pixel_match[overlap_mask]
                right_x = matching_right_xy[:, 0]
                right_y = matching_right_xy[:, 1]
                
                # 构建合法索引 mask（要求 x, y 都在图像范围内）
                valid_mask = (right_x >= 0) & (right_x < W) & (right_y >= 0) & (right_y < H)

                # 过滤非法点
                matching_right_xy = matching_right_xy[valid_mask]
                right_x = right_x[valid_mask]
                right_y = right_y[valid_mask]
                
                mask_right_embed_t1 = embed_t1[right_y, right_x].mean(axis=0)
                
                cosine_similarity = torch.nn.functional.cosine_similarity(mask_embedding_t0, mask_right_embed_t1, dim=0)
                
                if args.sem_filter is not None:
                    # print(f"do sem filter")
                    cos = F.cosine_similarity(embed_t0[overlap_mask], embed_t1[overlap_mask], dim=1)
                    cosine_similarity = cos.mean()
                    if cosine_similarity < args.sem_filter:  # 对应的sam 的last feature的cos sim，还需要把t1时刻投影到t0时刻，先不投直接用iou筛选看看效果。
                        mask_idx_t0.append(i)
                else:
                    mask_idx_t0.append(i)

        x = np.zeros_like(initial_pseudo_mask)
        for j in mask_idx_t0:
            x = np.logical_or(x, masks_t0[j]['segmentation'])
        
        final_change_mask = x
            
        # Resize final_change_mask to output-size and ensure binary output
        final_change_mask = final_change_mask.astype(np.uint8) * 255  # ensure dtype and scale for cv2
        final_change_mask = cv2.resize(final_change_mask, self.output_size, interpolation=cv2.INTER_LINEAR)
        final_change_mask = (final_change_mask > 127).astype(np.uint8)  # threshold to binary
        
        return final_change_mask
        