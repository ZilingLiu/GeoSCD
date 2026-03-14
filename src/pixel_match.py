import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square, load_and_resize_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF

def save_depth_map(depth_tensor, save_path, cmap='viridis', normalize=True, output_size=None):
    """
    保存深度图
    Args:
        depth_tensor: torch.Tensor, 形状 [H, W, 1] 或 [H, W]
        save_path: str, 保存路径 (如 'depth.png')
        cmap: matplotlib colormap (默认 'viridis')
        normalize: 是否归一化到 [0, 1]
        output_size: tuple (H, W)，指定输出分辨率。如果为 None，则保持原大小
    """
    if not isinstance(depth_tensor, torch.Tensor):
        raise TypeError("输入必须是 torch.Tensor")

    # 去掉最后一维 [H, W, 1] -> [H, W]
    if depth_tensor.ndim == 3 and depth_tensor.shape[-1] == 1:
        depth_tensor = depth_tensor.squeeze(-1)

    depth_np = depth_tensor.detach().cpu().numpy()

    # 归一化到 [0, 1]
    if normalize:
        min_val, max_val = depth_np.min(), depth_np.max()
        if max_val > min_val:
            depth_np = (depth_np - min_val) / (max_val - min_val)

    # resize 到目标分辨率
    if output_size is not None:
        depth_np = cv2.resize(depth_np, (output_size[1], output_size[0]), interpolation=cv2.INTER_LINEAR)

    # 用 colormap 映射
    plt.figure(figsize=(6, 6))
    plt.imshow(depth_np, cmap=cmap)
    plt.axis("off")
    plt.colorbar(label="Depth")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
    print(f"✅ 深度图已保存到 {save_path}, 分辨率: {depth_np.shape[::-1]}")

def visualize_confidence_map(conf_map: torch.Tensor, save_path=None):
    """
    可视化置信度图像，值越大颜色越红。
    
    Args:
        conf_map (torch.Tensor): [H, W] 形状的置信度图像。
        save_path (str): 如果不为 None，则保存图片到此路径。
        show (bool): 是否显示图像。
    """
    assert conf_map.ndim == 2, "conf_map 应该是 [H, W] 形状"
    
    # 归一化到 [0, 255]
    conf_np = conf_map.cpu().numpy()
    conf_norm = (conf_np - conf_np.min()) / (conf_np.max() - conf_np.min() + 1e-8)
    conf_uint8 = (conf_norm * 255).astype(np.uint8)

    # 使用 OpenCV 生成彩色热力图（colormap=JET）
    heatmap = cv2.applyColorMap(conf_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # 转为 RGB 以便 matplotlib 显示

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))  # OpenCV保存需要BGR

    return heatmap


def overlay_mask_on_image(mask: np.ndarray, rgb_path: str, output_path: str = "output.png",
                          color=(255, 0, 0), alpha=0.5):
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

def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        
        return extrinsic, intrinsic, depth_map, depth_conf

def uv_to_pixel_indices(uv: torch.Tensor, intrinsic: torch.Tensor, mode: str = 'round') -> torch.Tensor:
    """
    将以光心为原点的 (u, v) 浮点坐标转换为以左上角为原点的图像像素索引坐标。

    Args:
        uv:         torch.Tensor of shape (..., 2)，(u, v) 坐标，**以光心为原点**
        intrinsic:  torch.Tensor of shape (3, 3)，相机内参，用于获取主点 (cx, cy)
        mode:       取整方式：['round', 'floor', 'ceil', 'trunc']

    Returns:
        torch.Tensor of shape (..., 2)，(u_pixel, v_pixel) 整数像素索引坐标（左上角为原点）
    """
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]  # 主点位置
    uv_shifted = uv + torch.tensor([cx, cy], device=uv.device, dtype=uv.dtype)  # 平移到左上角为原点

    if mode == 'round':
        pixel_indices = torch.round(uv_shifted)
    elif mode == 'floor':
        pixel_indices = torch.floor(uv_shifted)
    elif mode == 'ceil':
        pixel_indices = torch.ceil(uv_shifted)
    elif mode == 'trunc':
        pixel_indices = torch.trunc(uv_shifted)
    else:
        raise ValueError(f"Unsupported rounding mode: {mode}")

    return pixel_indices.to(torch.int64)

def matching_two_imgs(extrinsic: torch.Tensor,
                      intrinsic: torch.Tensor,
                      depth: torch.Tensor, right_intrinsic: torch.Tensor) -> torch.Tensor:
    device = depth.device
    dtype = depth.dtype
    H, W = depth.shape[:2]

    # 1. 左图像像素网格
    ys = torch.arange(0, H, device=device, dtype=dtype)
    xs = torch.arange(0, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    ones = torch.ones_like(grid_x)
    pixel_homo = torch.stack([grid_x, grid_y, ones], dim=-1)  # (H, W, 3)

    # 2. 反投影到左相机坐标系
    K_inv = torch.inverse(intrinsic)
    camera_dirs = pixel_homo @ K_inv.T
    pts_left = camera_dirs * depth  # (H, W, 3)

    # 3. 左相机坐标 → 右相机坐标（先求右相机外参的逆）
    extrinsic_4x4 = torch.eye(4, device=device, dtype=dtype)
    extrinsic_4x4[:3, :4] = extrinsic
    # left_to_right = torch.linalg.inv(extrinsic_4x4)
    left_to_right = extrinsic_4x4

    ones_col = torch.ones((H, W, 1), device=device, dtype=dtype)
    pts_left_homo = torch.cat([pts_left, ones_col], dim=-1)  # (H, W, 4)
    pts_right_homo = pts_left_homo @ left_to_right.T
    pts_right = pts_right_homo[..., :3]

    # 4. 投影到右图像平面
    # proj = pts_right @ intrinsic.T
    proj = pts_right @ right_intrinsic.T
    u_proj = proj[..., 0] / proj[..., 2]
    v_proj = proj[..., 1] / proj[..., 2]

    matching = torch.stack([u_proj, v_proj], dim=-1)  # (H, W, 2)
    pixel_match = torch.round(matching).to(torch.int64)
    # pixel_match = uv_to_pixel_indices(matching, intrinsic)
    return pixel_match

def matching_and_project(extrinsic: torch.Tensor,
                                intrinsic: torch.Tensor,
                                depth: torch.Tensor, left_intrinsic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = depth.device
    dtype = depth.dtype
    H, W = depth.shape[:2]

    # 1. 右图像素网格
    ys = torch.arange(0, H, device=device, dtype=dtype)
    xs = torch.arange(0, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    ones = torch.ones_like(grid_x)
    pixel_homo = torch.stack([grid_x, grid_y, ones], dim=-1)  # (H, W, 3)

    # 2. 反投影到右相机坐标系
    K_inv = torch.inverse(intrinsic)
    camera_dirs = pixel_homo @ K_inv.T
    pts_right = camera_dirs * depth # (H, W, 3)

    # 3. 右 → 左 相机坐标变换
    extrinsic_4x4 = torch.eye(4, device=device, dtype=dtype)
    extrinsic_4x4[:3, :4] = extrinsic
    # right_to_left = torch.inverse(extrinsic_4x4)

    ones_col = torch.ones((H, W, 1), device=device, dtype=dtype)
    pts_right_homo = torch.cat([pts_right, ones_col], dim=-1)  # (H, W, 4)
    # pts_left_homo = pts_right_homo @ right_to_left.T
    pts_left_homo = pts_right_homo @ extrinsic_4x4.T
    pts_left = pts_left_homo[..., :3]

    # 4. 投影到左图像素
    # proj = pts_left @ intrinsic.T
    proj = pts_left @ left_intrinsic.T
    u_proj = proj[..., 0] / (proj[..., 2] + 1e-8)
    v_proj = proj[..., 1] / (proj[..., 2] + 1e-8)

    matching = torch.stack([u_proj, v_proj], dim=-1)  # (H, W, 2)
    pixel_match = torch.round(matching).to(torch.int64)

    # 5. 生成左图视角下的depth map（稀疏）
    depth_map = torch.zeros((H, W), dtype=dtype, device=device)
    z_vals = pts_left[..., 2]  # 投影到左图的深度

    # 只保留投影到图像范围内的像素点
    u_round = pixel_match[..., 0]
    v_round = pixel_match[..., 1]
    valid_mask = (u_round >= 0) & (u_round < W) & (v_round >= 0) & (v_round < H)

    # 展平处理
    u_valid = u_round[valid_mask]
    v_valid = v_round[valid_mask]
    z_valid = z_vals[valid_mask]

    # 生成一个index扁平版：depth_map[v, u] = z
    depth_map[v_valid, u_valid] = z_valid

    return pixel_match, depth_map, valid_mask, z_vals

def matching_right_to_left(extrinsic: torch.Tensor,
                          intrinsic: torch.Tensor,
                          depth_right: torch.Tensor, 
                          left_intrinsic: torch.Tensor) -> torch.Tensor:
    """
    将右图的深度投影到左图的像素平面，返回左图上对应的像素坐标
    
    参数：
        extrinsic: 右相机的世界到相机变换（w2c），形状 [3, 4]
        intrinsic: 右相机的内参矩阵，形状 [3, 3]
        depth_right: 右图的深度图，形状 [H, W]
        left_intrinsic: 左相机的内参矩阵，形状 [3, 3]
    
    返回：
        pixel_match: 左图上对应的像素坐标 [H, W, 2] (整数)
    """
    device = depth_right.device
    dtype = depth_right.dtype
    H, W = depth_right.shape[:2]


    ys = torch.arange(0, H, device=device, dtype=dtype)
    xs = torch.arange(0, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    ones = torch.ones_like(grid_x)
    pixel_homo_right = torch.stack([grid_x, grid_y, ones], dim=-1)  # (H, W, 3)


    K_right_inv = torch.inverse(intrinsic)
    camera_dirs_right = pixel_homo_right @ K_right_inv.T
    pts_right = camera_dirs_right * depth_right  # (H, W, 3)

    extrinsic_4x4 = torch.eye(4, device=device, dtype=dtype)
    extrinsic_4x4[:3, :4] = extrinsic
    right_to_left = torch.inverse(extrinsic_4x4) 
    ones_col = torch.ones((H, W, 1), device=device, dtype=dtype)
    pts_right_homo = torch.cat([pts_right, ones_col], dim=-1)  # (H, W, 4)
    pts_left_homo = pts_right_homo @ right_to_left.T  
    pts_left = pts_left_homo[..., :3]

    proj_left = pts_left @ left_intrinsic.T
    u_proj = proj_left[..., 0] / proj_left[..., 2]
    v_proj = proj_left[..., 1] / proj_left[..., 2]

    matching = torch.stack([u_proj, v_proj], dim=-1)  # (H, W, 2)
    pixel_match = torch.round(matching).to(torch.int64)
    return pixel_match

def save_depth_map(depth_map, save_path, cmap="viridis", normalize=True, output_size=None):
    """
    保存深度图为干净图片 (无标题、无坐标轴).
    
    Args:
        depth_map (torch.Tensor): 深度图 [H, W] 或 [H, W, 1]
        save_path (str): 保存路径
        cmap (str): Matplotlib colormap (默认 "viridis")
        normalize (bool): 是否归一化到 [0, 1]
        output_size (tuple): (H, W)，输出分辨率，None 则保持原始分辨率
    """
    if not isinstance(depth_map, torch.Tensor):
        raise TypeError("输入必须是 torch.Tensor")

    if depth_map.ndim == 3 and depth_map.shape[-1] == 1:
        depth_map = depth_map.squeeze(-1)

    depth_np = depth_map.detach().cpu().numpy()

    # 归一化
    if normalize:
        min_val, max_val = depth_np.min(), depth_np.max()
        if max_val > min_val:
            depth_np = (depth_np - min_val) / (max_val - min_val + 1e-8)

    # resize
    if output_size is not None:
        depth_np = cv2.resize(depth_np, (output_size[1], output_size[0]), interpolation=cv2.INTER_LINEAR)

    # 画图 (无坐标轴/标题/colorbar)
    plt.figure(figsize=(6, 6))
    plt.imshow(depth_np, cmap=cmap)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()
    print(f"✅ 已保存: {save_path}, 分辨率: {depth_np.shape[::-1]}")

def compute_occlusion_mask(
    match_coords: torch.Tensor,    # 匹配坐标 [H, W, 2] (x, y)
    target_depth: torch.Tensor,    # 目标视图的深度图 [H, W]
    projected_zvals: torch.Tensor, # 投影到目标视图的深度值 [H, W]
    k: float = 2.5,                # MAD 系数
    geom_ratio: float = 0.03       # 几何比例下限
) -> torch.Tensor:
    """
    根据几何一致性和统计阈值，计算需要忽略的遮挡 mask。

    Args:
        match_coords (torch.Tensor): 匹配到目标视图的坐标 [H, W, 2] (x, y)。
        target_depth (torch.Tensor): 目标视图深度图 [H, W]。
        projected_zvals (torch.Tensor): 源点投影到目标视图的深度值 [H, W]。
        k (float): MAD 的放大系数，默认 2.5。
        geom_ratio (float): 几何比例下限，默认 0.03。

    Returns:
        ignore_mask (torch.BoolTensor): [H, W]，True 表示需要忽略的点。
    """
    H, W = match_coords.shape[:2]

    # 判断是否在图像范围内
    x_in = (match_coords[..., 0] > 0) & (match_coords[..., 0] < W)
    y_in = (match_coords[..., 1] > 0) & (match_coords[..., 1] < H)
    in_frustum_mask = x_in & y_in

    # 归一化坐标 [-1, 1]，用于 grid_sample
    coords_norm = match_coords.clone().to(torch.float32)
    coords_norm[..., 0] = coords_norm[..., 0] / (W - 1) * 2 - 1
    coords_norm[..., 1] = coords_norm[..., 1] / (H - 1) * 2 - 1

    # 采样目标深度
    target_depth_map = target_depth.permute(2, 0, 1).unsqueeze(0)  # [1,1,H,W]
    sampled_depth = F.grid_sample(
        target_depth_map, coords_norm.unsqueeze(0), align_corners=True, mode='bilinear'
    ).squeeze(0).squeeze(0)  # [H,W]

    # 仅保留有效点
    valid_mask = in_frustum_mask & (sampled_depth > 0)

    # 计算几何下限
    if valid_mask.any():
        median_depth = torch.median(projected_zvals[valid_mask])
        tau_geom = geom_ratio * median_depth.item()
    else:
        tau_geom = 0.0

    # 计算统计下限 (median + k * MAD)
    valid_diff = (projected_zvals - sampled_depth)[valid_mask]
    if valid_diff.numel() > 0:
        median_diff = torch.median(valid_diff)
        mad = torch.median(torch.abs(valid_diff - median_diff)) + 1e-6
        tau_stat = median_diff + k * mad
    else:
        tau_stat = 0.0

    # 取两者的最大值
    tau_z = max(tau_geom, tau_stat)

    # 遮挡判定
    ignore_mask = valid_mask & (projected_zvals - sampled_depth > tau_z)

    return ignore_mask

def visualize_pixel_matches(image1: np.ndarray,
                            image2: np.ndarray,
                            pixel_match: torch.Tensor,
                            num_points: int = 100,
                            random_seed: int = 42,
                            save_path: str = None):
    """
    可视化左右图之间的像素匹配关系。

    Args:
        image1 (np.ndarray): 左图，形状为 (H, W, 3)
        image2 (np.ndarray): 右图，形状为 (H, W, 3)
        pixel_match (torch.Tensor): 左图每个像素对应右图坐标，(H, W, 2)
        num_points (int): 随机采样的点数
        random_seed (int): 随机种子，保证可复现
        save_path (str or None): 如果设置则保存图片到该路径
    """
    assert image1.shape == image2.shape, "Image shapes must match"
    H, W, _ = image1.shape
    pixel_match = pixel_match.cpu().numpy()
    
    # 拼接图像
    combined_img = np.concatenate([image1, image2], axis=1)

    # 随机采样一些像素点
    np.random.seed(random_seed)
    ys = np.random.randint(0, H, size=num_points)
    xs = np.random.randint(0, W, size=num_points)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(combined_img.astype(np.uint8))

    for x, y in zip(xs, ys):
        u2, v2 = pixel_match[y, x]  # 右图中的匹配点坐标
        x2, y2 = int(u2), int(v2)
        if 0 <= x2 < W and 0 <= y2 < H:
            # 在左图的点
            ax.plot(x, y, 'ro', markersize=2)
            # 在右图的点（注意 x 方向需要偏移 W）
            ax.plot(x2 + W, y2, 'go', markersize=2)
            # 画连线
            ax.plot([x, x2 + W], [y, y2], 'y-', linewidth=0.5)

    ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved to {save_path}")
    else:
        plt.show()

def bgr2rgb(img):
    return img[..., ::-1]  # BGR -> RGB        
        
def visualize_pixel_matches_beta(image1: np.ndarray,
                            image2: np.ndarray,
                            pixel_match: torch.Tensor,
                            num_points: int = 100,
                            random_seed: int = 42,
                            save_path: str = None,
                            mask_path: str = None):
    """
    可视化左右图之间的像素匹配关系（支持 mask 过滤）。

    Args:
        image1 (np.ndarray): 左图，形状为 (H, W, 3)
        image2 (np.ndarray): 右图，形状为 (H, W, 3)
        pixel_match (torch.Tensor): 左图每个像素对应右图坐标，(H, W, 2)
        num_points (int): 随机采样的点数
        random_seed (int): 随机种子
        save_path (str or None): 如果设置则保存图片到该路径
        mask_path (str or None): 如果设置，只在 mask>0 的位置采样
    """
    assert image1.shape == image2.shape, "Image shapes must match"
    H, W, _ = image1.shape
    pixel_match = pixel_match.cpu().numpy()

    # 读取 mask 并 resize 到 image 分辨率
    if mask_path is not None:
        mask_img = Image.open(mask_path).convert("L")
        mask_img = mask_img.resize((W, H), Image.NEAREST)  # 最近邻避免灰度模糊
        mask = np.array(mask_img) > 0
    else:
        mask = np.ones((H, W), dtype=bool)

    # 拼接图像
    combined_img = np.concatenate([image1, image2], axis=1)
    # combined_img = bgr2rgb(combined_img).astype(np.uint8)
  

    # 在 mask 内随机采样
    ys_all, xs_all = np.where(mask)
    np.random.seed(random_seed)
    if len(xs_all) > num_points:
        idx = np.random.choice(len(xs_all), size=num_points, replace=False)
        xs, ys = xs_all[idx], ys_all[idx]
    else:
        xs, ys = xs_all, ys_all
        
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(combined_img.astype(np.uint8))
    for x, y in zip(xs, ys):
        u2, v2 = pixel_match[y, x]  # 右图中的匹配点坐标
        x2, y2 = int(u2), int(v2)
        if 0 <= x2 < W and 0 <= y2 < H:
            ax.plot(x, y, 'ro', markersize=4)
            ax.plot(x2 + W, y2, 'go', markersize=4)
            ax.plot([x, x2 + W], [y, y2], 'y-', linewidth=0.8)


    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved to {save_path}")
    else:
        plt.show()

def matching_project_right2left(extrinsic: torch.Tensor,
                                right_intrinsic: torch.Tensor,
                                right_depth: torch.Tensor, left_intrinsic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = right_depth.device
    dtype = right_depth.dtype
    H, W = right_depth.shape[:2]

    # 1. 右图像素网格
    ys = torch.arange(0, H, device=device, dtype=dtype)
    xs = torch.arange(0, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    ones = torch.ones_like(grid_x)
    pixel_homo = torch.stack([grid_x, grid_y, ones], dim=-1)  # (H, W, 3)

    # 2. 反投影到右相机坐标系
    K_inv = torch.inverse(right_intrinsic)
    camera_dirs = pixel_homo @ K_inv.T
    pts_right = camera_dirs * right_depth # (H, W, 3)

    # 3. 右 → 左 相机坐标变换。
    extrinsic_4x4 = torch.eye(4, device=device, dtype=dtype)
    extrinsic_4x4[:3, :4] = extrinsic
    right_to_left = torch.inverse(extrinsic_4x4)

    ones_col = torch.ones((H, W, 1), device=device, dtype=dtype)
    pts_right_homo = torch.cat([pts_right, ones_col], dim=-1)  # (H, W, 4)
   
    pts_left_homo = pts_right_homo @ right_to_left.T
    pts_left = pts_left_homo[..., :3]

    # 4. 投影到左图像素
    # proj = pts_left @ intrinsic.T
    proj = pts_left @ left_intrinsic.T
    u_proj = proj[..., 0] / (proj[..., 2] + 1e-8)
    v_proj = proj[..., 1] / (proj[..., 2] + 1e-8)

    matching = torch.stack([u_proj, v_proj], dim=-1)  # (H, W, 2)
    pixel_match = torch.round(matching).to(torch.int64)

    # 5. 生成左图视角下的depth map（稀疏）
    depth_map = torch.zeros((H, W), dtype=dtype, device=device)
    z_vals = pts_left[..., 2]  # 投影到左图的深度

    # 只保留投影到图像范围内的像素点
    u_round = pixel_match[..., 0]
    v_round = pixel_match[..., 1]
    valid_mask = (u_round >= 0) & (u_round < W) & (v_round >= 0) & (v_round < H)

    # 展平处理
    u_valid = u_round[valid_mask]
    v_valid = v_round[valid_mask]
    z_valid = z_vals[valid_mask]

    # 生成一个index扁平版：depth_map[v, u] = z
    depth_map[v_valid, u_valid] = z_valid

    return pixel_match, depth_map, valid_mask, z_vals
  
def resize_depth(depth_map, resolution):
    depth = depth_map[0, 0].squeeze(-1)  # (518, 518)

    # 添加 batch 和 channel 维度，变成 (1, 1, 518, 518)
    depth = depth.unsqueeze(0).unsqueeze(0)

    # 使用 bilinear 或 nearest 插值进行 resize
    depth_resized = F.interpolate(depth, size=(resolution, resolution), mode='bilinear', align_corners=False)

    # 去掉 batch 和 channel 维度，变回 (resolution, 512)
    depth_resized = depth_resized.squeeze(0).squeeze(0)
    depth_resized = depth_resized.unsqueeze(-1)
    return depth_resized

def resize_depth_pair(depth_map, resolution):
    """
    Resize [0, 0] 和 [0, 1] 的 depth map 到指定 resolution。

    Args:
        depth_map (torch.Tensor): 输入形状为 [1, 2, H, W, 1] 的 depth tensor
        resolution (int): 输出深度图的高和宽

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: resized left_depth, right_depth，形状为 [resolution, resolution, 1]
    """
    left_depth = depth_map[0, 0].squeeze(-1)   # (H, W)
    right_depth = depth_map[0, 1].squeeze(-1)  # (H, W)

    # 添加 batch 和 channel 维度 -> (1, 1, H, W)
    left_depth = left_depth.unsqueeze(0).unsqueeze(0)
    right_depth = right_depth.unsqueeze(0).unsqueeze(0)

    # 使用 bilinear 插值进行 resize
    left_resized = F.interpolate(left_depth, size=(resolution, resolution), mode='bilinear', align_corners=False)
    right_resized = F.interpolate(right_depth, size=(resolution, resolution), mode='bilinear', align_corners=False)

    # 去掉 batch 和 channel 维度 -> (resolution, resolution)
    left_resized = left_resized.squeeze(0).squeeze(0).unsqueeze(-1)
    right_resized = right_resized.squeeze(0).squeeze(0).unsqueeze(-1)

    return left_resized, right_resized

def resize_intrinsic(intrinsic: torch.Tensor, original_size: tuple, new_size: tuple) -> torch.Tensor:
    """
    Resize camera intrinsic matrix based on new image resolution.

    Args:
        intrinsic (torch.Tensor): shape (3, 3)
        original_size (tuple): (H, W) of original image
        new_size (tuple): (H, W) of resized image

    Returns:
        torch.Tensor: adjusted intrinsic matrix, shape (3, 3)
    """
    orig_h, orig_w = original_size
    new_h, new_w = new_size

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    # Copy to avoid modifying in-place
    new_intrinsic = intrinsic.clone()
    new_intrinsic[0, 0] *= scale_x  # fx
    new_intrinsic[1, 1] *= scale_y  # fy
    new_intrinsic[0, 2] *= scale_x  # cx
    new_intrinsic[1, 2] *= scale_y  # cy

    return new_intrinsic
    
def run_dense_match(image_path_list, model, device, dtype, resolution = 512, light = False,
                    bright_thresh = 25.0, color_thresh = 0.2, transfer = False):
    
    img_load_resolution = 1024
    vggt_fixed_resolution = 518
    
    
    images = load_and_resize_images(image_path_list, img_load_resolution, light=light, brightness_threshold=bright_thresh, 
                                    color_thresh=color_thresh, transfer = transfer)
    
    images = images.to(device)
    # original_coords = original_coords.to(device)
    print(f"Image Loaded for VGGT Finsih!")
    extrinsic, intrinsic, depth_map, conf_map= run_VGGT(model, images, dtype, vggt_fixed_resolution)
    
    
    left_depth, right_depth = resize_depth_pair(depth_map, resolution)        
        
    left_intrinsic = resize_intrinsic(intrinsic[0, 0], (vggt_fixed_resolution, vggt_fixed_resolution), (resolution, resolution))
    right_intrinsic = resize_intrinsic(intrinsic[0, 1], (vggt_fixed_resolution, vggt_fixed_resolution), (resolution, resolution))
    
    #pixel_match = matching_two_imgs(extrinsic[0, 1], left_intrinsic, left_depth, right_intrinsic)
    
    
    # get occupied map ****************** important step *******************
    left2right_match ,left2right_depth, valid_mask, z_vals = matching_and_project(extrinsic[0, 1], left_intrinsic, left_depth, right_intrinsic)
    
    # 代码简化
    #right2left_pixel_match = matching_right_to_left(extrinsic[0, 1], left_intrinsic, left_depth, right_intrinsic)
   # right2left_pixel_match = matching_right_to_left(extrinsic[0, 1], right_intrinsic, right_depth, left_intrinsic)
    
    # 反向投影
    right2left_match ,right2left_depth, right_valid_mask, right2left_z_vals = matching_project_right2left(extrinsic[0, 1], right_intrinsic, right_depth, left_intrinsic)
    
    ignore_left_mask = compute_occlusion_mask(left2right_match, right_depth, z_vals)
    ignore_right_mask = compute_occlusion_mask(right2left_match, left_depth, right2left_z_vals)
    
    left_diated_mask = dilate_occlusion_mask(ignore_left_mask)
    right_diated_mask = dilate_occlusion_mask(ignore_right_mask)    
    
    
    return left2right_match, left_diated_mask, left2right_depth, right2left_match, right_diated_mask, right2left_depth, extrinsic[0, 1]
    
    # return pixel_match, dilated_mask, left2right_depth
def dilate_occlusion_mask(
    ignore_mask: torch.Tensor,  # 输入的遮挡掩码 [H, W] (bool或0/1)
    padding_size: int = 3,      # 膨胀的边界扩展像素数
    kernel_shape: str = 'ellipse'  # 膨胀核形状 ('ellipse', 'rect', 'cross')
) -> torch.Tensor:
   
    # 转换输入为CPU上的uint8格式（OpenCV处理需要）
    mask_np = ignore_mask.cpu().numpy().astype(np.uint8) * 255
    
    # 创建膨胀核
    kernel_size = 2 * padding_size + 1
    if kernel_shape == 'ellipse':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    elif kernel_shape == 'rect':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    elif kernel_shape == 'cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    else:
        raise ValueError(f"Unsupported kernel_shape: {kernel_shape}")
    
    # 执行膨胀操作
    dilated_mask_np = cv2.dilate(mask_np, kernel, iterations=1).astype(bool)
    # dilated_mask_np = mask_np
    return dilated_mask_np