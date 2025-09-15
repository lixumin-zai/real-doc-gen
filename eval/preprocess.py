import os
from paddleocr import DocImgOrientationClassification
from doclayout_yolo import YOLOv10
from PIL import Image, ImageDraw
import cv2
import numpy as np
import json
import uuid
import tqdm
from io import BytesIO

import feishu_sdk
from feishu_sdk.sheet import FeishuSheet, FeishuImage
import os
app_id, app_key = "cli_a12ac5906d7c900e", "jSFDBwE3MbtXA6o9fSsOHdGbd3pVj1WY"
feishu_sdk.login(app_id, app_key)

class Process:
    def __init__(self):
        self.ori_model = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")
        self.question_cut_model=YOLOv10("/home/lixumin/project/DocLayout-YOLO/0903/yolov10./yolov8n.pt_./data_epoch100_imgsz640_bs128_pretrain_unknown/weights/epoch90.pt")

    def cut_question(self, image):
        
        det_res = self.question_cut_model.predict(
            image,   # Image to predict
            imgsz=640,        # Prediction image size
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


def get_predict_box(w, h, box):
    data = {
        "id": f"{uuid.uuid4()}",
        "type": "rectanglelabels",        
        "from_name": "label", "to_name": "image",
        "original_width": w, "original_height": h,
        "image_rotation": 0,
        "value": {
            "rotation": 0,          
            "x": box[0]/w*100, "y": box[1]/h*100,
            "width": (box[2]-box[0])/w*100, "height": (box[3]-box[1])/h*100,
            "rectanglelabels": ["题目"]
        }
    }
    return data

def find_image_paths_os(root_folder: str) -> list[str]:
    """
    使用 os.walk 递归查找指定文件夹及其子文件夹中所有图片的路径。
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    image_paths = []
    if not os.path.isdir(root_folder):
        print(f"错误: 路径 '{root_folder}' 不是一个有效的文件夹。")
        return []
    for dirpath, _, filenames in os.walk(root_folder):
        # if "/images" in dirpath:
        #     continue
        for filename in filenames:
            if filename.lower().endswith(image_extensions):
                full_path = os.path.join(dirpath, filename)
                image_paths.append(full_path)
    return image_paths


def main():
    # https://isw1t6yp68.feishu.cn/sheets/Bb3kslUvehQ4bLtoTwZcZMBLnPg?sheet=pKed2B
    sheet_token, sheet_id = "Bb3kslUvehQ4bLtoTwZcZMBLnPg", "pKed2B"
    sheet = FeishuSheet(sheet_token, sheet_id)
    box_col = "E"
    box_len_col = "F"
    result_col = "G"

    root_path = "/home/lixumin/project/doclayout-pipeline/eval/images/"
    images_path = find_image_paths_os(root_path)

    infos = []
    process = Process()
    print(images_path)
    for image_path in tqdm.tqdm(images_path):
        image = Image.open(image_path).convert("RGB")
        # angle = process.ori(image)
        # rotated_img = image.rotate(int(angle), expand=True) # Pillow中，正数是逆时针，负数是顺时针
        
        idx = int(image_path.split("/")[-1].split(".")[0])

        w, h = image.size
        boxes = process.cut_question(image)

        sheet[f"{box_col}{idx}"] = str(boxes)
        sheet[f"{box_len_col}{idx}"] = len(boxes)

        draw = ImageDraw.Draw(image)

        result = []
        for box in boxes:
            # label_studio json 格式
            x1, y1, x2, y2 = box
            predict_data = get_predict_box(w, h, box)
            result.append(predict_data)
            draw.rectangle((x1, y1, x2, y2), outline="red", width=5)
        
        with BytesIO() as new_image:
            image.save(new_image, format='JPEG')
            new_image_bytes = new_image.getvalue()
            sheet[f"{result_col}{idx}"] = FeishuImage(img_bytes=new_image_bytes)
        
        image_url = f"http://172.20.253.222:65432/{idx}.jpg"
        infos.append({
            "data": {
                "image": image_url 
            },
            "predictions": [{
                "model_version": "one",
                "score": 1,
                "result": result
            }]
        })
        # break
    
    with open("1.json", "w") as f:
        json.dump(infos, f)
        

def nms_single_class_partial_containment(boxes, scores, containment_threshold=0.8, iou_threshold=None):
    """
    Perform non-maximum suppression (NMS) on bounding boxes for a single class,
    removing boxes that are largely contained within another box.

    Parameters:
    - boxes: List of bounding boxes [x1, y1, x2, y2].
    - scores: Confidence scores for each bounding box.
    - containment_threshold: The minimum ratio of the inner box's area that must
                            be within the outer box to consider it largely contained.
    - iou_threshold: Optional IoU threshold for cases that are overlapping but not
                    largely contained. Set to None to only handle containment.

    Returns:
    - A list of indices representing the bounding boxes kept after NMS.
    """

    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    suppressed = np.zeros(len(boxes), dtype=bool)

    for i in range(len(order)):
        if suppressed[:].any() and suppressed[:][order][i]:
            continue

        current_index = order[:][i]
        keep.append(current_index)

        current_box = boxes[:][current_index]
        current_area = areas[:][current_index]
        cx1, cy1, cx2, cy2 = current_box

        for j in range(i + 1, len(order)):
            other_index = order[:][j]
            if suppressed[:][other_index]:
                continue

            other_box = boxes[:][other_index]
            ox1, oy1, ox2, oy2 = other_box
            other_area = areas[:][other_index]

            # Calculate intersection area
            ix1 = np.maximum(cx1, ox1)
            iy1 = np.maximum(cy1, oy1)
            ix2 = np.minimum(cx2, ox2)
            iy2 = np.minimum(cy2, oy2)

            intersection_w = np.maximum(0, ix2 - ix1 + 1)
            intersection_h = np.maximum(0, iy2 - iy1 + 1)
            intersection_area = intersection_w * intersection_h

            # Check if the other box is largely contained within the current box
            if intersection_area > 0 and (intersection_area / other_area) >= containment_threshold:
                suppressed[:][other_index] = True
            elif intersection_area > 0 and (intersection_area / current_area) >= containment_threshold:
                suppressed[:][current_index] = True # Suppress current if other largely contains it
                if current_index in keep:
                    keep.remove(current_index)
                break # Move to the next highest score box

            # Optionally apply IoU threshold for non-containment overlaps
            elif iou_threshold is not None:
                iou = intersection_area / (current_area + other_area - intersection_area)
                if iou > iou_threshold:
                    suppressed[:][other_index] = True

    return keep



def get_yolo_data():
    with open("./1.json", "r") as f:
        data = json.load(f)

    for info in tqdm.tqdm(data):
        image_name = info["data"]["image"].split("/")[-1][:-4]

        with open(f"/home/lixumin/project/data/question-data/train/labels/{image_name}.txt", "w") as f:
            for box in info["predictions"][0]["result"]:
                original_width, original_height = box["original_width"], box["original_height"]
                x, y, w, h = box["value"]["x"]/100+box["value"]["width"]/100/2, box["value"]["y"]/100+box["value"]["height"]/100/2, \
                    box["value"]["width"]/100, box["value"]["height"]/100

                label = f"0 {x} {y} {w} {h}\n"
                f.write(label)

        # break




if __name__ == "__main__":
    main()
    # get_yolo_data()
    # import random
    # with open("./1.json", "r") as f:
    #     data = json.load(f)

    # random.shuffle(data)

    # with open("./2.json", "w") as f:
    #     json.dump(data[:500], f)
