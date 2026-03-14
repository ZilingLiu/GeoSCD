import numpy as np
import os
import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import cv2
import matplotlib.pyplot as plt
from PIL import Image


# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def visualize_flow_matches(image1, image2, flow_final, num_samples=200, seed=42, save_path="flow_matches.png"):
    """
    可视化光流匹配（连线方式），并保存到文件

    Args:
        image1 (torch.Tensor): [1, 3, H, W]
        image2 (torch.Tensor): [1, 3, H, W]
        flow_final (torch.Tensor): [1, 2, H, W]
        num_samples (int): 随机采样的点数
        seed (int): 随机种子，保证可复现
        save_path (str): 保存路径

    Returns:
        str: 保存的文件路径
    """
    # 转 numpy [H,W,3], 保证uint8
    img1 = (image1[0].permute(1,2,0).cpu().numpy()).astype(np.uint8)
    img2 = (image2[0].permute(1,2,0).cpu().numpy()).astype(np.uint8)
    H, W, _ = img1.shape

    # flow: [H,W,2]
    flow = flow_final[0].permute(1,2,0).cpu().numpy()

    # 拼接两张图，并 copy 保证是连续数组
    canvas = np.concatenate([img1, img2], axis=1).copy().astype(np.uint8)

    # 随机采样点
    np.random.seed(seed)
    ys = np.random.randint(0, H, size=num_samples)
    xs = np.random.randint(0, W, size=num_samples)

    for x1, y1 in zip(xs, ys):
        u, v = flow[y1, x1]
        x2 = int(np.clip(x1 + u, 0, W-1))
        y2 = int(np.clip(y1 + v, 0, H-1))

        # 右图的x坐标要加偏移W
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2 + W), int(y2))

        color = tuple(np.random.randint(0, 255, size=3).tolist())
        cv2.circle(canvas, pt1, 2, color, -1)
        cv2.circle(canvas, pt2, 2, color, -1)
        cv2.line(canvas, pt1, pt2, color, 1)

    # 保存结果
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, canvas)

    print(f"Flow matches visualization saved to {save_path}")
    return save_path


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

    ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved to {save_path}")
    else:
        plt.show()
        
def run_flow_match(image1, image2, model, device, dtype, debug=False, save_path="flow_match.png", mask_path = None):
    infer_H = 540
    infer_W = 960
    final_H = 512
    final_W = 512
    img1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
    img2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
    img1 = img1[None].to(device)
    img2 = img2[None].to(device)
    infer_img1 = F.interpolate(img1, size=(infer_H, infer_W), mode='bilinear', align_corners=False)
    infer_img2 = F.interpolate(img2, size=(infer_H, infer_W), mode='bilinear', align_corners=False)
    
    output = model(infer_img1, infer_img2, iters=4, test_mode=True)
    flow_final = output['flow'][-1]
    
    # if debug:
    #     visualize_flow_matches(image1, image2, flow_final)
        
    flow_final = F.interpolate(flow_final, size=(final_H, final_W), mode='bilinear', align_corners=False)
    
    B, C, H, W = flow_final.shape
    # 网格坐标 [H, W]
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=flow_final.device),
                                    torch.arange(W, device=flow_final.device), indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1).float()  # [H, W, 2], (x,y)

    # 光流 [H, W, 2]
    flow = flow_final[0].permute(1, 2, 0)  # [H, W, 2]

    # 右图坐标 = 左图坐标 + flow
    coords_right = grid + flow  # [H, W, 2], 其中[...,0] = x2, [...,1] = y2
    coords_right = coords_right.detach().to(torch.int32)  # 转为整数坐标
    if debug:
        vis_image1 = cv2.resize(image1, (W, H))
        vis_image2 = cv2.resize(image2, (W, H))
        visualize_pixel_matches_beta(vis_image1, vis_image2, coords_right, num_points=150, save_path=save_path, mask_path=mask_path)
        vis_flow = flow_to_image(flow.cpu().detach().numpy())
        cv2.imwrite("raw_flow.png", vis_flow)
    
    return coords_right
    
    




