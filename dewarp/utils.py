import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import glob
import os
from PIL import Image
import argparse
import warnings
import time
warnings.filterwarnings('ignore')

from scipy.interpolate import griddata
import cv2
import numpy as np

# ==============================================================================
# ===== 新增：光流/映射场可视化函数 =====
# ==============================================================================
def visualize_flow(flow_x, flow_y, save_path):
    """
    将光流/映射场的x和y分量可视化为彩色图像。
    
    参数:
    - flow_x (np.array): 包含x方向位移的2D数组。
    - flow_y (np.array): 包含y方向位移的2D数组。
    - save_path (str): 可视化结果的保存路径。
    """
    # 确保输入的维度一致
    h, w = flow_x.shape
    assert flow_y.shape == (h, w), "x和y分量的维度必须相同"

    # 将两个分量堆叠成一个 (h, w, 2) 的数组，这是OpenCV处理光流的标准格式
    flow = np.stack([flow_x, flow_y], axis=-1)

    # 使用cv2.cartToPolar计算每个像素点的光流大小(magnitude)和角度(angle)
    # 角度表示方向，大小表示移动的强度
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 创建一个HSV颜色空间的图像 (h, w, 3)
    # HSV (Hue, Saturation, Value) -> (色调, 饱和度, 亮度)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. 色调 (Hue): 由角度决定，代表光流方向
    # 将弧度制的角度[0, 2*pi]转换为OpenCV的色调范围[0, 179]
    hsv[..., 0] = ang * 180 / np.pi / 2

    # 2. 饱和度 (Saturation): 设为最大值255，使颜色更鲜艳
    hsv[..., 2] = 255

    # 3. 亮度 (Value): 由大小决定，代表光流强度
    # 使用cv2.normalize将大小归一化到[0, 255]范围
    # 这意味着光流最强的地方最亮，光流为0的地方为黑色
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # 将HSV图像转换回BGR图像以便保存
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 保存图像
    cv2.imwrite(save_path, bgr)
    print(f"光流可视化结果已保存至: {save_path}")

def get_inverse_flow_fast(flow, downsample_factor=8):
    """
    通过降采样快速计算逆光流。
    :param flow: [2, H, W] 的反向光流图 (bm)。
    :param downsample_factor: 降采样因子，数值越大，速度越快，但精度可能略有下降。8是一个很好的起始值。
    :return: [H, W] 的 inv_flow_x 和 [H, W] 的 inv_flow_y。
    """
    H, W = flow.shape[1], flow.shape[2]
    H_small, W_small = H // downsample_factor, W // downsample_factor

    # 1. 降采样原始光流图
    # 注意：cv2.resize需要 (W, H) 格式
    # 我们需要分别对 x 和 y 通道进行缩放
    flow_x_small = cv2.resize(flow[0], (W_small, H_small), interpolation=cv2.INTER_LINEAR)
    flow_y_small = cv2.resize(flow[1], (W_small, H_small), interpolation=cv2.INTER_LINEAR)
    
    # 别忘了也要缩放光流值本身！
    flow_small = np.stack([flow_x_small, flow_y_small], axis=0) / downsample_factor

    # 2. 在低分辨率上执行 griddata
    points = flow_small.reshape(2, -1).T
    grid_y, grid_x = np.mgrid[0:H_small, 0:W_small]
    values_x = grid_x.flatten()
    values_y = grid_y.flatten()

    query_points_y, query_points_x = np.mgrid[0:H_small, 0:W_small]
    query_points = np.vstack([query_points_x.ravel(), query_points_y.ravel()]).T

    inv_flow_x_small = griddata(points, values_x, query_points, method='linear', fill_value=0)
    inv_flow_y_small = griddata(points, values_y, query_points, method='linear', fill_value=0)

    # 3. 将低分辨率的逆光流图上采样回原始尺寸
    inv_flow_x_small = inv_flow_x_small.reshape(H_small, W_small)
    inv_flow_y_small = inv_flow_y_small.reshape(H_small, W_small)
    
    # 放大坐标值
    inv_flow_x_large = cv2.resize(inv_flow_x_small, (W, H), interpolation=cv2.INTER_LINEAR) * downsample_factor
    inv_flow_y_large = cv2.resize(inv_flow_y_small, (W, H), interpolation=cv2.INTER_LINEAR) * downsample_factor

    return inv_flow_x_large.astype(np.float32), inv_flow_y_large.astype(np.float32)


def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

def initialize_flow(img):
    N, C, H, W = img.shape
#     print(N, C, H, W)
# def initialize_flow(N, C, H, W):
    coodslar = coords_grid(N, H, W).to(img.device)
    coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
    coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

    return coodslar, coords0, coords1

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    # return coords[None].repeat(batch, 1, 1, 1)
    return coords.unsqueeze(0).expand(batch, -1, -1, -1)

def upsample_flow(flow, mask):
    N, _, H, W = flow.shape
    mask = mask.view(N, 1, 9, 8, 8, H, W)
    mask = torch.softmax(mask, dim=2) 

    up_flow = F.unfold(8 * flow, [3, 3], padding=1) 
    up_flow = up_flow.view(N, 2, 9, 1, 1, H, W) 

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    
    return up_flow.reshape(N, 2, 8 * H, 8 * W)


def find_image_paths_os(root_folder: str) -> list[str]:
    """
    使用 os.walk 递归查找指定文件夹及其子文件夹中所有图片的路径。
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    image_paths = []
    if not os.path.isdir(root_folder):
        print(f"错误: 路径 '{root_folder}' 不是一个有效的文件夹。")
        return []

    print(f"正在扫描图片文件 {root_folder}")
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(image_extensions):
                full_path = os.path.join(dirpath, filename)
                image_paths.append(full_path)
    print(f"扫描完成，共找到 {len(image_paths)} 张图片。")
    return image_paths

def flow_to_image(flow, max_flow=256):
    """
    将光流场转换为可视化的彩色图像 (HSV 颜色空间)。
    Args:
        flow (np.array): 光流场, shape [2, H, W] or [H, W, 2].
        max_flow (int): 用于归一化的最大光流值。
    Returns:
        (np.array): RGB 图像, shape [H, W, 3].
    """
    if flow.ndim == 3 and flow.shape[0] == 2:
        flow = flow.transpose(1, 2, 0)
        
    u = flow[..., 0]
    v = flow[..., 1]

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 2] = 255  # Value

    mag, ang = cv2.cartToPolar(u, v)
    
    # Hue: angle
    hsv[..., 0] = ang * 180 / np.pi / 2
    
    # Saturation: magnitude
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    rgb_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb_img