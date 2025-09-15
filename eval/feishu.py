import feishu_sdk
from feishu_sdk.sheet import FeishuSheet, FeishuImage
import os
app_id, app_key = "cli_a12ac5906d7c900e", "jSFDBwE3MbtXA6o9fSsOHdGbd3pVj1WY"
feishu_sdk.login(app_id, app_key)
import numpy as np
import cv2
from PIL import Image, ImageDraw
from io import BytesIO

import os
from paddleocr import DocImgOrientationClassification
from doclayout_yolo import YOLOv10
import json
import uuid
import tqdm
import io
from dewarp import DocDewarp

class Process:
    def __init__(self):
        self.doc_dewarp = DocDewarp()
        self.ori_model = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")
        self.question_cut_model=YOLOv10("/home/lixumin/project/DocLayout-YOLO/test/yolov10./yolov8n.pt_/home/lixumin/project/pre-ocr/paper-cut/data_epoch100_imgsz1024_bs16_pretrain_unknown/weights/best.pt")

    def cut_question(self, image):
        
        det_res = self.question_cut_model.predict(
            image,   # Image to predict
            imgsz=1024,        # Prediction image size
            conf=0.1,          # Confidence threshold
            device="cuda:1"    # Device to use (e.g., 'cuda:0' or 'cpu')
        )

        result_boxes = []
        result_scores = []
        for result in det_res:
            image = result.orig_img
            for i in result.boxes:
                label = int(i.cls.tolist()[0])  # Convert to int if needed
                box = i.xyxy.tolist()[0]
                conf = i.conf.tolist()[0]
                result_boxes.append([int(box[0]), int(box[1]),int(box[2]),int(box[3])])
                result_scores.append(conf)
        iou_threshold = 0.3  # You can adjust this value

        keep_indices = nms_single_class_partial_containment(result_boxes, result_scores, iou_threshold=iou_threshold)
        filtered_boxes = [result_boxes[i] for i in keep_indices]

        # filtered_boxes = result_boxes

        # for index, box in enumerate(filtered_boxes):
        #     image =cv2.rectangle(image, (int(box[0]), int(box[1])),(int(box[2]),int(box[3])), (0,0,255), 3)
        #     cv2.putText(image, str(index+1), (int(box[0]), int(box[3])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0), thickness=2)
        # cv2.imwrite("result.jpg", image)

        return filtered_boxes

    def ori(self, image):
        data = self.ori_model.predict(np.array(image),  batch_size=1)
        angle = data[0].json["res"]["label_names"][0]
        return angle
        # rotated_img = image.rotate(int(angle), expand=True) # Pillow中，正数是逆时针，负数是顺时针
        # rotated_img.save("./1.jpg")

    def dewarp(self, image):
        dewarped_images = self.doc_dewarp(image)
        return dewarped_images

def feishu_test():
    # https://isw1t6yp68.feishu.cn/sheets/Bb3kslUvehQ4bLtoTwZcZMBLnPg?sheet=pKed2B
    sheet_token, sheet_id = "Bb3kslUvehQ4bLtoTwZcZMBLnPg", "pKed2B"
    sheet = FeishuSheet(sheet_token, sheet_id)
    image_col = "B"
    result_col = "C"

    process = Process()

    for i in range(min(sheet.rows+1, 10000)):
        if i < 3:
            continue
        try:
            if sheet[f"{result_col}{i}"]:
                continue

            print(1)
            image_bytes = sheet[f"{image_col}{i}"].image_bytes
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            angle = process.ori(image)
            rotated_img = image.rotate(int(angle), expand=True) # Pillow中，正数是逆时针，负数是顺时针

            with BytesIO() as new_image:
                rotated_img.save(new_image, format='JPEG')
                new_image_bytes = new_image.getvalue()
                sheet[f"{result_col}{i}"] = FeishuImage(img_bytes=new_image_bytes)
        except:
            import traceback
            traceback.print_exc()
            break


def feishu_test_1():
    # https://isw1t6yp68.feishu.cn/sheets/Bb3kslUvehQ4bLtoTwZcZMBLnPg?sheet=pKed2B
    sheet_token, sheet_id = "Bb3kslUvehQ4bLtoTwZcZMBLnPg", "pKed2B"
    sheet = FeishuSheet(sheet_token, sheet_id)
    image_col = "C"
    result_col = "D"

    process = Process()

    for i in range(min(sheet.rows+1, 10000)):
        if i < 3:
            continue
        try:
            if sheet[f"{result_col}{i}"]:
                continue
            print(i)
            image_bytes = sheet[f"{image_col}{i}"].image_bytes
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            new_image_bytes = process.dewarp(image)
            sheet[f"{result_col}{i}"] = FeishuImage(img_bytes=new_image_bytes)
            with open(f"eval/images/{i}.jpg", "wb") as f:
                f.write(new_image_bytes)

        except:
            import traceback
            traceback.print_exc()
            break

if __name__ == "__main__":
    feishu_test_1()