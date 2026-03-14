# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from torchvision import transforms as TF
import numpy as np
import cv2
import os

def rgb2lab_reinhard(img):
    """Convert RGB to lαβ color space (Reinhard et al. 2001)"""
    img = img.astype(np.float32) / 255.0

    # RGB -> LMS
    M_rgb2lms = np.array([
        [0.3811, 0.5783, 0.0402],
        [0.1967, 0.7244, 0.0782],
        [0.0241, 0.1288, 0.8444]
    ])
    lms = np.dot(img.reshape(-1, 3), M_rgb2lms.T)
    lms[lms <= 1e-6] = 1e-6
    lms = np.log10(lms)

    # LMS -> lαβ
    M_lms2lab = np.array([
        [ 1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)],
        [ 1/np.sqrt(6),  1/np.sqrt(6), -2/np.sqrt(6)],
        [ 1/np.sqrt(2), -1/np.sqrt(2),  0]
    ])
    lab = np.dot(lms, M_lms2lab.T)
    return lab.reshape(img.shape), M_rgb2lms, M_lms2lab

def lab2rgb_reinhard(lab, M_rgb2lms, M_lms2lab):
    """Convert lαβ back to RGB"""
    # lαβ -> log LMS
    M_lab2lms = np.linalg.inv(M_lms2lab)
    lms = np.dot(lab.reshape(-1, 3), M_lab2lms.T)
    lms = 10 ** lms

    # log LMS -> RGB
    M_lms2rgb = np.linalg.inv(M_rgb2lms)
    rgb = np.dot(lms, M_lms2rgb.T)
    rgb = np.clip(rgb, 0, 1)
    return (rgb.reshape(lab.shape) * 255).astype(np.uint8)

def transfer_color(source, target):
    """Transfer color distribution from source to target image"""
    lab_src, M_rgb2lms, M_lms2lab = rgb2lab_reinhard(source)
    lab_tar, _, _ = rgb2lab_reinhard(target)

    # 计算均值和标准差
    mean_src, std_src = np.mean(lab_src.reshape(-1, 3), axis=0), np.std(lab_src.reshape(-1, 3), axis=0)
    mean_tar, std_tar = np.mean(lab_tar.reshape(-1, 3), axis=0), np.std(lab_tar.reshape(-1, 3), axis=0)

    # 匹配分布
    lab_tar_norm = (lab_tar - mean_tar) / std_tar
    lab_transfer = lab_tar_norm * std_src + mean_src

    # 转回 RGB
    result = lab2rgb_reinhard(lab_transfer, M_rgb2lms, M_lms2lab)
    return result

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

def color_and_illum_difference(img1, img2, illum_thresh=10.0, color_thresh=10.0):
    """
    计算两张图片的亮度差和颜色差，并判断光照是否显著变化
    
    参数：
        img1, img2: RGB 图像 (uint8)
        illum_thresh: 光照变化阈值（L通道平均差）
        color_thresh: 颜色变化阈值（ΔE平均差）
    
    返回：
        illum_diff: 光照差值
        color_diff: 颜色差值
        significant_illum_change: bool 是否光照显著变化
    """
    lab1 = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # 亮度差 L 通道
    L1, a1, b1 = cv2.split(lab1)
    L2, a2, b2 = cv2.split(lab2)
    illum_diff = np.abs(np.mean(L1) - np.mean(L2))
    
    # 颜色差 ΔE76 (包含亮度和颜色)
    delta_E = np.linalg.norm(lab1 - lab2, axis=2)
    color_diff = np.mean(delta_E)
    
    significant_illum_change = illum_diff > illum_thresh
    
    return illum_diff, color_diff, significant_illum_change


def compute_brightness(img):
    """计算图像亮度均值（灰度）"""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    return np.mean(gray)

def color_distribution_difference(img1, img2, bins=32, color_space='lab'):
    """
    通过颜色直方图比较两张图片的整体颜色分布差异
    
    参数：
        img1, img2: RGB 图像 (uint8)
        bins: 直方图的 bin 数量
        color_space: 颜色空间 ('lab' 或 'hsv')
    
    返回：
        hist_diff: 直方图差异值（范围 0~1，越小越相似）
    """
    # 转换颜色空间
    if color_space == 'lab':
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)
    elif color_space == 'hsv':
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
    
    # 计算直方图（对每个通道）
    hist1 = [cv2.calcHist([img1], [i], None, [bins], [0, 256]) for i in range(3)]
    hist2 = [cv2.calcHist([img2], [i], None, [bins], [0, 256]) for i in range(3)]
    
    # 归一化直方图
    hist1 = [cv2.normalize(h, None, 0, 1, cv2.NORM_MINMAX) for h in hist1]
    hist2 = [cv2.normalize(h, None, 0, 1, cv2.NORM_MINMAX) for h in hist2]
    
    # 计算直方图相似度（使用相关系数或巴氏距离）
    similarity = [cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) for h1, h2 in zip(hist1, hist2)]
    hist_diff = 1 - np.mean(similarity)  # 转换为差异值（0=完全相同，1=完全不同）
    
    return hist_diff

def load_and_resize_images(image_path_list, target_size=512, brightness_threshold=25, 
                           light = False, transfer = False, color_thresh = 0.20):
    """
    Load images and resize them directly to a fixed square resolution.
    If brightness difference between first two images > threshold, apply Retinex.

    Args:
        image_path_list (list): List of image file paths.
        target_size (int): Target square resolution (e.g. 512).
        brightness_threshold (float): Brightness diff threshold to trigger Retinex.

    Returns:
        torch.Tensor: Batched tensor of shape (N, 3, target_size, target_size)
    """
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images_cv = []
    for image_path in image_path_list:
        img = Image.open(image_path)

        # Remove alpha channel if present
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        img = img.convert("RGB")
        img_cv = np.array(img)  # 转 numpy
        images_cv.append(img_cv)

    # 判断亮度差
    if len(images_cv) >= 2 and light:
        bright_diff = abs(compute_brightness(images_cv[0]) - compute_brightness(images_cv[1]))
        color_diff = color_distribution_difference(images_cv[0], images_cv[1])
        if bright_diff > brightness_threshold or color_diff > color_thresh:
      
            
            if transfer:
                print(f"apply color transfer...")
                bright1 = compute_brightness(images_cv[0])
                bright2 = compute_brightness(images_cv[1])
                
                if bright1 > bright2:
                    images_cv[0] = transfer_color(images_cv[1], images_cv[0])
                else:
                    images_cv[1] = transfer_color(images_cv[0], images_cv[1])
                
            else:
                print(f"[Info] Brightness diff = {bright_diff:.2f}, applying Retinex normalization...")
                images_cv = [simple_retinex(img) for img in images_cv]
                            

    # resize + 转 tensor
    images = []
    to_tensor = TF.ToTensor()
    for img_cv in images_cv:
        img_pil = Image.fromarray(img_cv)
        img_pil = img_pil.resize((target_size, target_size), Image.Resampling.BICUBIC)
        img_tensor = to_tensor(img_pil)
        images.append(img_tensor)

    return torch.stack(images)


def load_and_preprocess_images_square(image_path_list, target_size=1024):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 518.

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 5) containing [x1, y1, x2, y2, width, height] for each image
        )

    Raises:
        ValueError: If the input list is empty
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    original_coords = []  # Renamed from position_info to be more descriptive
    to_tensor = TF.ToTensor()

    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        # Convert to RGB
        img = img.convert("RGB")

        # Get original dimensions
        width, height = img.size

        # Make the image square by padding the shorter dimension
        max_dim = max(width, height)

        # Calculate padding
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        # Calculate scale factor for resizing
        scale = target_size / max_dim

        # Calculate final coordinates of original image in target space
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        # Store original image coordinates and scale
        original_coords.append(np.array([x1, y1, x2, y2, width, height]))

        # Create a new black square image and paste original
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))

        # Resize to target size
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        # Convert to tensor
        img_tensor = to_tensor(square_img)
        images.append(img_tensor)

    # Stack all images
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()

    # Add additional dimension if single image to ensure correct shape
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)

    return images, original_coords


def load_and_preprocess_images(image_path_list, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images
