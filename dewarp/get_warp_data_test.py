from geotr import GeoTr

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
    print(N, C, H, W)
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


class DocDewarp:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                                                                                                                                                                                                                                                                                                                                                  
        self._GeoTr = GeoTr()
        reload_model(self._GeoTr, "/home/lixumin/project/docDewarp/my_model/model.pt")
        self._GeoTr = self._GeoTr.eval().to(self.device)

    def __call__(self, image:Image.Image):
        st = time.time()
        im_ori = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) / 255.

        # im_ori = cv2.imread(img_path) 
        h_,w_,c_ = im_ori.shape
        im_ori = cv2.resize(im_ori, (1024, 1024))

        h, w, _ = im_ori.shape
        im = cv2.resize(im_ori, (288, 288))
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # geometric unwarping
            coodslar, coords0, coords1 = initialize_flow(im)
            coords1 = coords1.detach()
            mask, coords1, feature  = self._GeoTr(im, coords1, return_features=True)
            coords = coords1 - coords0

            flow_up = upsample_flow(coords, mask)
            bm = coodslar + flow_up

            # 将像素坐标归一化到 [-1, 1] 的范围 这是288
            bm = 2 * (bm / 288) - 1
            # 将归一化的坐标 [-1, 1] 转换回像素坐标，但这次是映射到新的、更大的目标尺寸 2560x2560,值为[0， 2560]
            bm = (bm + 1) / 2 * 1024
            bm = F.interpolate(bm, size=(1024, 1024), mode='bilinear', align_corners=True)
            bm = bm.cpu().numpy()[0]
            bm0 = bm[0, :, :]
            bm1 = bm[1, :, :]
            bm0 = cv2.blur(bm0, (3, 3))
            bm1 = cv2.blur(bm1, (3, 3))

            img_geo = cv2.remap(im_ori, bm0, bm1, cv2.INTER_LINEAR)*255
            img_geo = cv2.resize(img_geo, (w_, h_))

            # 2. (核心部分) 计算并应用逆光流
            flat_image_path = "/home/lixumin/project/doclayout-pipeline/dewarp/images/2.png"
            warped_flat_image = None
            if flat_image_path and os.path.exists(flat_image_path):
                print("\n--- Performing Inverse Warping ---")
                # 计算逆光流
                # print(bm0.shape)
                inv_bm0, inv_bm1 = get_inverse_flow_fast(np.stack([bm0, bm1], axis=0))

                # ===== 调用可视化函数 =====
                # 首先，计算位移场 (flow) = 目标坐标 (inv_bm) - 原始坐标 (grid)
                inv_h, inv_w = inv_bm0.shape
                print(inv_h, inv_w)
                grid_y, grid_x = np.mgrid[0:inv_h, 0:inv_w]
                print(inv_bm0)
                # flow_x 和 flow_y 代表了每个像素点需要移动的x,y距离
                flow_x = inv_bm0 - grid_x
                flow_y = inv_bm1 - grid_y
                
                # 可视化这个位移场
                visualize_flow(flow_x, flow_y, "./images/inverse_flow_visualization.png")

                # 应用逆映射场来扭曲一张平坦图片
                flat_image = cv2.imread(flat_image_path)
                # flat_image = flat_image[:639, :853]
                cv2.imwrite("./images/warped_origin_image.jpg", flat_image)
                flat_image_h, flat_image_w = flat_image.shape[:2]
                
                flat_image_resized = cv2.resize(flat_image, (1024, 1024))

                warped_flat_image = cv2.remap(flat_image_resized, inv_bm0, inv_bm1, cv2.INTER_LINEAR)
                warped_flat_image_final = cv2.resize(warped_flat_image, (flat_image_w, flat_image_h))
                
                cv2.imwrite("./images/warped_flat_image.jpg", warped_flat_image_final)

                warped_flat_image = cv2.remap(warped_flat_image, bm0, bm1, cv2.INTER_LINEAR)
                warped_flat_image_final = cv2.resize(warped_flat_image, (flat_image_w, flat_image_h))
                cv2.imwrite("./images/warped_flat_image——1.jpg", warped_flat_image_final)

                print("使用逆映射扭曲的平坦图像已保存为 warped_flat_image.jpg")
                print("--- 逆映射场处理完成 ---")

            success, encoded_image = cv2.imencode('.jpg', img_geo)
            image_bytes = encoded_image.tobytes()
        return image_bytes


if __name__ == '__main__':
    dewarp = DocDewarp()

    # --- 演示1：矫正一张图片 (原始功能) ---
    # print("--- Running standard dewarping demo ---")
    image = Image.open("/home/lixumin/project/doclayout-pipeline/dewarp/images/1.png").convert("RGB")
    image_bytes = dewarp(image)
    with open("./images/1-dewarp.jpg", "wb") as f:
        f.write(image_bytes)
        # /home/lixumin/project/doclayout-pipeline/data.jpg