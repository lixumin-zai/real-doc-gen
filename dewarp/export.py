from geocnn import GeoCNN

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

# /home/lixumin/project/docDewarp/My-DocTr-Plus/rectified/screenshot-20250214-151800_geo.png
# /home/lixumin/project/docDewarp/My-DocTr-Plus/rectified/1740144085514-6855_geo.png

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
        self._GeoTr = GeoCNN()
        self._GeoTr = self._GeoTr
        # checkpoint = torch.load("/home/lixumin/project/docDewarp/my_model/ckpt/best_light_model-loss0.0037.pth", map_location='cpu')
        # self._GeoTr.load_state_dict(checkpoint["model_state_dict"])
        checkpoint = torch.load("/home/lixumin/project/doclayout-pipeline/dewarp/dewarping_model_weights.pt", map_location='cpu')
        self._GeoTr.load_state_dict(checkpoint)
        # reload_model(self._GeoTr, "./model_fp16.pt")

    def __call__(self, image):
        st = time.time()
        im_ori = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) / 255.

        # im_ori = cv2.imread(img_path) 
        h_,w_,c_ = im_ori.shape
        im_ori = cv2.resize(im_ori, (1024, 1024))

        h, w, _ = im_ori.shape
        im = cv2.resize(im_ori, (288, 288))
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float().unsqueeze(0)
        
        with torch.no_grad():
            # geometric unwarping
            coodslar, coords0, coords1 = initialize_flow(im)
            coords1 = coords1.detach()
            mask, coords1, feature = self._GeoTr(im, coords1, return_features=True)
            print(feature)
            coords = coords1 - coords0

            flow_up = upsample_flow(coords, mask)
            print("flow", flow_up.shape)
            bm = coodslar + flow_up

            bm = 2 * (bm / 288) - 1
            bm = (bm + 1) / 2 * 1024
            bm = F.interpolate(bm, size=(1024, 1024), mode='bilinear', align_corners=True)
            bm = bm.cpu().numpy()[0]
            bm0 = bm[0, :, :]
            bm1 = bm[1, :, :]
            bm0 = cv2.blur(bm0, (3, 3))
            bm1 = cv2.blur(bm1, (3, 3))

            img_geo = cv2.remap(im_ori, bm0, bm1, cv2.INTER_LINEAR)*255
            img_geo = cv2.resize(img_geo, (w_, h_))
            success, encoded_image = cv2.imencode('.jpg', img_geo)
            image_bytes = encoded_image.tobytes()
        return image_bytes

    def save_fp16(self):
        self._GeoTr = self._GeoTr.cuda().half()
        model_dict = self._GeoTr.state_dict()
        torch.save(model_dict, "model_fp16.pt")

    def save_jit(self):
        self._GeoTr.eval()
        # 3. 创建样本输入
        dummy_input1 = torch.randn(1, 3, 288, 288)
        dummy_input2 = torch.randn(1, 2, 36, 36)
        # 4. 导出为 TorchScript
        # pnnx multi_input_model.pt inputshape=[1,3,32,32],[1,3,32,32]
        traced_model = torch.jit.trace(self._GeoTr,(dummy_input1, dummy_input2))
        traced_model.save("output/model_jit.pt")


if __name__ == '__main__':
    dewarp = DocDewarp()
    dewarp.save_jit()
    image = Image.open("/home/lixumin/project/doclayout-pipeline/dewarp/images/output - 2025-09-10T144252.028.png").convert("RGB")
    image_bytes = dewarp(image)
    with open("./show1.jpg", "wb") as f:
        f.write(image_bytes)

    # image_bytes = dewarp(image)
    # input()
    # pnnx model_jit.pt device=cpu inputshape=[1,3,288,288]f32,[1,2,36,36]f32 fp16=1
