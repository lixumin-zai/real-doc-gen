from model import GeoTr

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
        self._GeoTr = GeoTr()
        self._GeoTr = self._GeoTr
        reload_model(self._GeoTr, "/home/lixumin/project/docDewarp/my_model/model.pt")

    def __call__(self, image:Image.Image):
        st = time.time()
        im_ori = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) / 255.

        # im_ori = cv2.imread(img_path) 
        h_,w_,c_ = im_ori.shape
        im_ori = cv2.resize(im_ori, (2560, 2560))

        h, w, _ = im_ori.shape
        im = cv2.resize(im_ori, (288, 288))
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float().unsqueeze(0)
        
        with torch.no_grad():
            # geometric unwarping
            coodslar, coords0, coords1 = initialize_flow(im)
            coords1 = coords1.detach()
            mask, coords1, feature  = self._GeoTr(im, coords1, return_features=True)
            print(feature)
            coords = coords1 - coords0

            flow_up = upsample_flow(coords, mask)
            bm = coodslar + flow_up

            bm = 2 * (bm / 288) - 1
            bm = (bm + 1) / 2 * 2560
            bm = F.interpolate(bm, size=(2560, 2560), mode='bilinear', align_corners=True)
            bm = bm.cpu().numpy()[0]
            bm0 = bm[0, :, :]
            bm1 = bm[1, :, :]
            bm0 = cv2.blur(bm0, (3, 3))
            bm1 = cv2.blur(bm1, (3, 3))

            img_geo = cv2.remap(im_ori, bm0, bm1, cv2.INTER_LINEAR)*255
            img_geo = cv2.resize(img_geo, (w_, h_))


            # 2. (核心部分) 计算并应用逆光流
            flat_image_path = "/home/lixumin/project/docDewarp/my_model/images/1.png"
            warped_flat_image = None
            if flat_image_path and os.path.exists(flat_image_path):
                print("\n--- Performing Inverse Warping ---")
                # 计算逆光流
                # print(bm0.shape)
                inv_bm0, inv_bm1 = get_inverse_flow_fast(np.stack([bm0, bm1], axis=0))
                print(inv_bm0.shape, inv_bm1.shape)
                # 读取一张平坦的图片
                flat_image = cv2.imread(flat_image_path)
                flat_image_resized = cv2.resize(flat_image, (2560, 2560))

                # 应用逆光流来扭曲这张平坦图片
                warped_flat_image = cv2.remap(flat_image_resized, inv_bm0, inv_bm1, cv2.INTER_LINEAR)
                warped_flat_image = cv2.resize(warped_flat_image, (w_, h_))
                print("--- Inverse Warping Done ---")

            success, encoded_image = cv2.imencode('.jpg', warped_flat_image)
            image_bytes = encoded_image.tobytes()
        return image_bytes

    def save_fp16(self):
        self._GeoTr = self._GeoTr.cuda().half()
        model_dict = self._GeoTr.state_dict()
        torch.save(model_dict, "model_fp16.pt")

    def save_jit(self):
        self._GeoTr.eval()
        # 3. 创建样本输入
        dummy_input = torch.randn(1, 3, 288, 288).cuda()
        # 4. 导出为 TorchScript
        traced_model = torch.jit.trace(self._GeoTr, dummy_input)
        traced_model.save("model_fp16_jit.pt")

from scipy.interpolate import griddata

def get_inverse_flow(flow):
    """
    计算反向光流图的逆，得到正向光流图。
    :param flow: [2, H, W] 的反向光流图 (bm)，flow[0]是x坐标, flow[1]是y坐标。
                 它表示 output[y, x] = input[flow[1, y, x], flow[0, y, x]]
    :return: [2, H, W] 的正向光流图 (inv_flow)。
             它表示 output[inv_flow[1, y, x], inv_flow[0, y, x]] = input[y, x]
             可以用于 cv2.remap(src, inv_flow[0], inv_flow[1], ...)
    """
    H, W = flow.shape[1], flow.shape[2]

    # 1. 创建源数据
    # flow[0] 和 flow[1] 是原始扭曲图像中的坐标点 (散乱的)
    points = flow.reshape(2, -1).T  # shape: (H*W, 2),  (x, y) 坐标对

    # 这些散乱点对应的值，是矫正后图像的规则网格坐标
    grid_y, grid_x = np.mgrid[0:H, 0:W]
    values_x = grid_x.flatten()
    values_y = grid_y.flatten()

    # 2. 创建插值目标网格 (我们想要求解的逆光流图的坐标)
    # 这也是一个规则的网格
    query_points_y, query_points_x = np.mgrid[0:H, 0:W]
    query_points = np.vstack([query_points_x.ravel(), query_points_y.ravel()]).T

    # 3. 进行插值
    # 对于规则网格上的每个点，插值出它在矫正后图像中对应的坐标
    print("Interpolating inverse flow for X coordinates...")
    inv_flow_x = griddata(points, values_x, query_points, method='linear', fill_value=0)
    print("Interpolating inverse flow for Y coordinates...")
    inv_flow_y = griddata(points, values_y, query_points, method='linear', fill_value=0)

    # 4. 整理成光流图格式
    inv_flow_x = inv_flow_x.reshape(H, W).astype(np.float32)
    inv_flow_y = inv_flow_y.reshape(H, W).astype(np.float32)

    return inv_flow_x, inv_flow_y

from scipy.interpolate import griddata
import cv2
import numpy as np

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

if __name__ == '__main__':
    dewarp = DocDewarp()

    # --- 演示1：矫正一张图片 (原始功能) ---
    # print("--- Running standard dewarping demo ---")
    image = Image.open("/home/lixumin/project/docDewarp/my_model/images/test.jpg").convert("RGB")
    # rectified_bytes = dewarp(distorted_pil)
    # with open("show_rectified.jpg", "wb") as f:
    #     f.write(rectified_bytes)
    # print("Rectified image saved to show_rectified.jpg\n")
        # dewarp = DocDewarp()
    # dewarp.save_jit()
    # image = Image.open("show.jpg").convert("RGB")
    image_bytes = dewarp(image)
    with open("./show1.jpg", "wb") as f:
        f.write(image_bytes)