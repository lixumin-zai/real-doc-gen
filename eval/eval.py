import os
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


class Eval:
    def __init__(self):
        self.question_cut_model=YOLOv10("/home/lixumin/project/DocLayout-YOLO/0908/yolov10./yolov8n.pt_./data_epoch10_imgsz640_bs128_pretrain_unknown/weights/best1.pt")

    def __call__(self, image):
        det_res = self.question_cut_model.predict(
            image,   # Image to predict
            imgsz=640,        # Prediction image size
            conf=0.2,          # Confidence threshold
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


        return filtered_boxes


# def evalution():
#     evalor = Eval()
#     data_path = "/home/lixumin/project/doclayout-pipeline/eval/eval_set"
#     for data_name in os.listdir(os.path.join(data_path, "images")):
#         image_path = os.path.join(data_path, "images", data_name)
#         label_path = os.path.join(data_path, "labels", data_name.replace('.jpg', ".txt"))

#         image = Image.open(image_path).convert("RGB")
#         with open(label_path, "r") as f:
#             f.read()
#     pass


def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (list or np.array): [x1, y1, x2, y2]
        box2 (list or np.array): [x1, y1, x2, y2]

    Returns:
        float: The IoU value.
    """
    # Determine the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute the area of intersection
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def parse_yolo_labels(label_path, img_width, img_height):
    """
    Reads a YOLO format label file and converts normalized coordinates to absolute pixel coordinates.

    Args:
        label_path (str): Path to the .txt label file.
        img_width (int): Width of the corresponding image.
        img_height (int): Height of the corresponding image.

    Returns:
        list: A list of ground truth boxes in [x1, y1, x2, y2] format.
    """
    gt_boxes = []
    if not os.path.exists(label_path):
        return gt_boxes
        
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            # YOLO format: class_id, x_center, y_center, width, height (normalized)
            _, x_center, y_center, width, height = map(float, parts)

            # Convert to absolute pixel values
            abs_w = width * img_width
            abs_h = height * img_height
            abs_x_center = x_center * img_width
            abs_y_center = y_center * img_height

            # Convert from center-width-height to x1-y1-x2-y2
            x1 = int(abs_x_center - (abs_w / 2))
            y1 = int(abs_y_center - (abs_h / 2))
            x2 = int(abs_x_center + (abs_w / 2))
            y2 = int(abs_y_center + (abs_h / 2))
            
            gt_boxes.append([x1, y1, x2, y2])
    return gt_boxes


def evalution():
    """
    Main evaluation function. Iterates through the evaluation set,
    compares model predictions with ground truth labels, and calculates metrics.
    """
    evalor = Eval()
    data_path = "/home/lixumin/project/doclayout-pipeline/eval/eval_set"
    image_dir = os.path.join(data_path, "images")
    label_dir = os.path.join(data_path, "labels")
    
    # Evaluation parameters
    # iou_threshold = 0.7
    iou_threshold = 0.5
    
    # Metric counters
    total_tp = 0  # True Positives
    total_fp = 0  # False Positives
    total_fn = 0  # False Negatives

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    print(f"Starting evaluation on {len(image_files)} images with IoU threshold = {iou_threshold}...")

    for data_name in tqdm.tqdm(image_files, desc="Evaluating"):
        # if int(os.path.splitext(data_name)[0]) > 480:
        #     continue
        image_path = os.path.join(image_dir, data_name)
        label_path = os.path.join(label_dir, os.path.splitext(data_name)[0] + ".txt")

        # --- 1. Get Predictions ---
        image = Image.open(image_path).convert("RGB")
        pred_boxes = evalor(image)

        # --- 2. Get Ground Truth ---
        img_width, img_height = image.size
        gt_boxes = parse_yolo_labels(label_path, img_width, img_height)

        # --- 3. Match Predictions to Ground Truth ---
        tp = 0
        matched_gt = [False] * len(gt_boxes)

        # Iterate through each predicted box to find a match
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            # Find the best matching ground truth box for the current prediction
            for i, gt_box in enumerate(gt_boxes):
                if not matched_gt[i]: # Only match with unmatched ground truth boxes
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

            # If a match is found above the threshold, count it as a True Positive
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt[best_gt_idx] = True # Mark this ground truth box as matched

        # --- 4. Calculate TP, FP, FN for the current image ---
        fp = len(pred_boxes) - tp  # Predictions that didn't match any GT
        fn = len(gt_boxes) - tp   # GT boxes that were not matched by any prediction

        # --- 5. Accumulate totals ---
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # --- 6. Calculate Final Metrics ---
    epsilon = 1e-7 # To avoid division by zero
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    print("\n--- Evaluation Results ---")
    print(f"Total True Positives (TP): {total_tp}")
    print(f"Total False Positives (FP): {total_fp}")
    print(f"Total False Negatives (FN): {total_fn}")
    print("--------------------------")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")
    print("--------------------------")


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


if __name__ == "__main__":
    evalution()