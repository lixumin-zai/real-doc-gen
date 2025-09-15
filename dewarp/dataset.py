import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import (visualize_flow, get_inverse_flow_fast, reload_model, 
    initialize_flow, coords_grid, upsample_flow, find_image_paths_os)
import torch.multiprocessing as mp
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
from transforms import image_transforms
# --- Define constants for visualization ---

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class WarpDataset(Dataset):
    """
    """
    def __init__(self, warp_image_paths, dewarps_image_paths, bg_image_paths, transform=None, sample=100000):
        super().__init__()
        self.transform = transform
        self.sample = sample

        self.warp_images = []
        for warp_image_path in warp_image_paths:
            self.warp_images.extend(find_image_paths_os(warp_image_path))
        random.shuffle(self.warp_images)

        self.dewarp_images = []
        for dewarp_image_path in dewarps_image_paths:
            self.dewarp_images.extend(find_image_paths_os(dewarp_image_path))
        random.shuffle(self.dewarp_images)

        self.bg_images = []
        for bg_image_path in bg_image_paths:
            self.bg_images.extend(find_image_paths_os(bg_image_path))
        random.shuffle(self.bg_images)

        self.dewarp = DocDewarp()

    def __len__(self):
        return self.sample

    def __getitem__(self, idx):
        warp_image_path = self.warp_images[idx]
        dewarp_image_path = self.dewarp_images[idx]
        bg_image_path = random.choice(self.bg_images)

        new_im, flow_up, coords0 = self.dewarp(dewarp_image_path, warp_image_path, bg_image_path)

        return new_im, flow_up, flow_up

def add_bg(doc_img, bg_img):
    # --- 2. 计算新背景尺寸 ---
    # 获取文档图片的原始尺寸
    doc_h, doc_w, _ = doc_img.shape
    
    # 计算新的背景尺寸，使其比文档图片大 `padding_percent`
    # (1 + padding_percent / 100) 相当于 1.10 (如果 padding_percent 是 10)
    scale_factor_w = random.uniform(1.0, 1.2)
    scale_factor_h = random.uniform(1.0, 1.4)
    new_bg_w = int(doc_w * scale_factor_w)
    new_bg_h = int(doc_h * scale_factor_h)

    # --- 3. 调整背景图片 ---
    # 将背景图片缩放到计算出的新尺寸
    # cv2.INTER_CUBIC 是一种高质量的插值方法，适合放大
    bg_resized = cv2.resize(bg_img, (new_bg_w, new_bg_h), interpolation=cv2.INTER_CUBIC)

    # --- 4. 计算居中粘贴位置 ---
    # (背景宽度 - 文档宽度) / 2
    x_pos = (new_bg_w - doc_w) // 2  # 使用整数除法
    y_pos = (new_bg_h - doc_h) // 2
    paste_position = (x_pos, y_pos)

    # --- 5. 创建文档蒙版 (与之前方法相同) ---
    doc_gray = cv2.cvtColor(doc_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(doc_gray, 127, 255, cv2.THRESH_BINARY_INV)

    # --- 6. 创建带 Alpha 通道的文档图片 ---
    b, g, r = cv2.split(doc_img)
    doc_rgba = cv2.merge([b, g, r, mask])

    # --- 7. 使用 Pillow 进行融合 ---
    # 注意：这里使用的是调整后的背景 bg_resized
    doc_pil = Image.fromarray(cv2.cvtColor(doc_img, cv2.COLOR_RGB2RGBA))
    bg_pil = Image.fromarray(cv2.cvtColor(bg_resized, cv2.COLOR_BGR2RGB))
    
    # 将文档内容粘贴到计算好的居中位置
    bg_pil.paste(doc_pil, paste_position, doc_pil)

    return np.array(bg_pil)


def random_crop(image):
    h, w, _ = image.shape
    # 随机决定裁剪尺寸，例如原图的 70% 到 95%
    crop_scale_w = random.uniform(0.5, 0.9)
    crop_scale_h = random.uniform(0.2, 0.9)
    crop_w, crop_h = int(w * crop_scale_w), int(h * crop_scale_h)
    
    # 随机决定裁剪的起始位置
    if w - crop_w > 0:
        crop_x = random.randint(0, w - crop_w)
    else:
        crop_x = 0
        
    if h - crop_h > 0:
        crop_y = random.randint(0, h - crop_h)
    else:
        crop_y = 0
        
    cropped_image = image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    
    return cropped_image

class DocDewarp:
    def __init__(self):
        self.device = torch.device("cuda")                                                                                                                                                                                                                                                                                                                                                                  
        self._GeoTr = GeoTr()
        reload_model(self._GeoTr, "/home/lixumin/project/docDewarp/my_model/model.pt")
        self._GeoTr = self._GeoTr.eval().to(self.device)

        
    def __call__(self, dewarp_image_path, warp_image_path, bg_image_path):

        im_ori = cv2.imread(dewarp_image_path) / 255.

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
            bm0, bm1 = bm[0, :, :], bm[1, :, :]
            bm0, bm1 = cv2.blur(bm0, (3, 3)), cv2.blur(bm1, (3, 3))

            inv_bm0, inv_bm1 = get_inverse_flow_fast(np.stack([bm0, bm1], axis=0))

            # 1024 1024
            grid_y, grid_x = np.mgrid[0:1024, 0:1024]
            
            # flow_x 和 flow_y 代表了每个像素点需要移动的x,y距离
            flow_x = inv_bm0 - grid_x
            flow_y = inv_bm1 - grid_y

            # visualize_flow(flow_x, flow_y, "./images/inverse_flow_visualization.png")

            warp_image = cv2.imread(warp_image_path)
            warp_image = random_crop(warp_image)

            bg_image = cv2.imread(bg_image_path)
            if bg_image is not None:
                if random.random() < 0.5:
                    warp_image = add_bg(warp_image, bg_image)

            if random.random() < 0.5:
                warp_image = image_transforms(warp_image)["image"]

            # 做一些操作， 裁剪之类
            warp_image_h, warp_image_w = warp_image.shape[:2]

            # 与映射场对齐
            warp_image_resized = cv2.resize(warp_image, (1024, 1024))

            warped_warp_image = cv2.remap(warp_image_resized, inv_bm0, inv_bm1, cv2.INTER_LINEAR)

            # 还原
            # new_im = cv2.remap(warped_warp_image, bm0, bm1, cv2.INTER_LINEAR)
            # warped_flat_image_final = cv2.resize(new_im, (warp_image_w, warp_image_h))
            # cv2.imwrite("./images/show1.jpg", warped_flat_image_final)

            # 背景填充

            new_im = cv2.resize(warped_warp_image, (warp_image_w, warp_image_h))
            # 查看
            # cv2.imwrite("./images/show.jpg", new_im)

            warped_warp_image = warped_warp_image / 255.

            new_im = cv2.resize(warped_warp_image, (288, 288))
            new_im = new_im.transpose(2, 0, 1)
            new_im = torch.from_numpy(new_im).float()
        flow_up = flow_up.squeeze(0).cpu().detach()
        coords0 = coords0.squeeze(0).cpu().detach()
        return new_im, flow_up, coords0


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)

# def collate_fn(batch):
#     images, flow_up, coords0 = list(zip(*batch))
#     images = torch.stack(images, 0)
#     return images, flow_up, coords0


# transform = A.Compose([
#     # 1. 随机裁剪 (Random Cropping)
#     # RandomSizedCrop: 随机裁剪图片的一部分，并将其缩放到指定的大小。
#     # min_max_height: 定义裁剪区域的高度范围，相对于原图的比例。
#     # height, width: 裁剪后缩放到的目标尺寸。
#     # p=1.0: 表示这个操作总是被执行。
#     A.RandomSizedCrop(min_max_height=(256, 480), height=300, width=300, p=1.0),
    
#     # 2. RGB 等颜色变化
#     # 使用 A.OneOf 可以从一组变换中随机选择一个来执行
#     # 这样每次增强时，只会应用其中一种颜色扰动
#     A.OneOf([
#         # RGBShift: 随机改变 R, G, B 通道的值
#         # r_shift_limit, g_shift_limit, b_shift_limit: 定义每个通道变化的范围
#         A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0),
        
#         # HueSaturationValue: 在 HSV 颜色空间中随机改变色调、饱和度、明度
#         A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),

#         # RandomBrightnessContrast: 随机改变图片的亮度和对比度
#         A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
#     ], p=1.0), # p=1.0 保证 A.OneOf 内部的某个变换一定会被执行

#     # 3. 其他常用变换 (可选，用于演示)
#     A.HorizontalFlip(p=0.5), # 50% 的概率水平翻转
#     A.Rotate(limit=30, p=0.5),   # 50% 的概率在 -30 到 30 度之间随机旋转
# ])


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    data = WarpDataset(
        ["/home/lixumin/project/data/Vary-600k/data/pdf_data/pdf_cn_30w/114126"],
        ["/home/lixumin/project/data/question-data/3L2H6Q"],
        ["/home/lixumin/project/nanoseg/data/bg_images"],
        sample=3
    )

    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=2, # Increase batch size to see multiple examples
        num_workers=1,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
    )

    # Fetch one batch from the dataloader
    new_im, flow_up, coords0 = next(iter(data_loader))

    print(f"Batch of images shape: {new_im.shape}")
    print(f"Number of targets in batch: {flow_up.shape}")

    # Visualize each image in the batch
    for i in range(new_im.shape[0]):
        image_tensor = new_im[i]
        target = flow_up[i]
        break