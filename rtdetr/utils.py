import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import box_convert, generalized_box_iou
from typing import List, Dict
import math
from scipy.optimize import linear_sum_assignment
import cv2
# ====================================================================================
# --- 6. 损失函数和匹配器 (DETR-style) ---
# ====================================================================================

class HungarianMatcher(nn.Module):
    """
    匈牙利匹配器，用于在模型的300个预测和N个真实对象之间找到最佳匹配。
    """
    def __init__(self, cost_class: float = 2.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 我们将所有batch的预测拼接在一起，以便进行高效计算
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [bs * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [bs * num_queries, 4]

        # 同样拼接所有batch的GT
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # 计算分类代价。代价是 (1 - 预测为GT类别的概率)
        cost_class = -out_prob[:, tgt_ids]

        # 计算L1 bbox代价
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # 计算GIoU代价
        cost_giou = -generalized_box_iou(box_convert(out_bbox, in_fmt='cxcywh', out_fmt='xyxy'),
                                           box_convert(tgt_bbox, in_fmt='cxcywh', out_fmt='xyxy'))

        # 最终代价矩阵
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class SetCriterion(nn.Module):
    """
    此模块计算DETR的损失。
    它首先使用匈牙利算法匹配预测和GT，然后计算相应损失。
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef # "end of sentence" / "no object" class coefficient
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """分类损失 (交叉熵)"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_ce': loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """BBox L1和GIoU损失"""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_convert(src_boxes, in_fmt='cxcywh', out_fmt='xyxy'),
            box_convert(target_boxes, in_fmt='cxcywh', out_fmt='xyxy')))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        # 1. 使用匈牙利匹配器进行匹配
        indices = self.matcher(outputs, targets)

        # 2. 计算batch中所有GT box的总数，用于损失归一化
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        # 3. 计算所有损失
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        
        return losses


def collate_fn(batch):
    """Custom collate_fn for object detection targets."""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

def rescale_boxes_after_padding(boxes, original_size, target_size):
    """
    将经过 LongestMaxSize 和 PadIfNeeded 变换后的边界框坐标映射回原始图像。

    参数:
    - boxes (torch.Tensor): 形状为 [N, 4] 的边界框，格式为 (xmin, ymin, xmax, ymax)，
                            其坐标是相对于 target_size 的。
    - original_size (tuple): 原始图像的 (height, width)。
    - target_size (tuple): 模型输入图像的目标 (height, width)，例如 (640, 640)。

    返回:
    - torch.Tensor: 映射回原始图像坐标的边界框。
    """
    orig_h, orig_w = original_size
    target_h, target_w = target_size

    # 1. 计算在 LongestMaxSize 中使用的缩放比例
    #    这个比例是 target_size / max(original_size)
    scale = min(target_h / orig_h, target_w / orig_w)

    # 2. 计算缩放后、填充前的图像尺寸
    resized_h = int(orig_h * scale)
    resized_w = int(orig_w * scale)

    # 3. 计算添加的 padding 值 (假设是中心填充)
    pad_y = (target_h - resized_h) / 2  # 上方和下方的总 padding 的一半
    pad_x = (target_w - resized_w) / 2  # 左侧和右侧的总 padding 的一半

    # 4. 逆向操作：从坐标中减去 padding
    #    boxes 的坐标是相对于 640x640 画布的
    boxes_no_pad = boxes.clone()
    boxes_no_pad[:, [0, 2]] = boxes_no_pad[:, [0, 2]] - pad_x
    boxes_no_pad[:, [1, 3]] = boxes_no_pad[:, [1, 3]] - pad_y
    
    # 5. 逆向操作：将坐标除以缩放比例，恢复到原始尺寸
    #    注意：在除法之前，要确保坐标不会因为减去 padding 而变成负数，
    #    并限制在缩放后的图像区域内。
    boxes_no_pad[:, [0, 2]] = boxes_no_pad[:, [0, 2]].clamp(min=0, max=resized_w)
    boxes_no_pad[:, [1, 3]] = boxes_no_pad[:, [1, 3]].clamp(min=0, max=resized_h)
    
    # 缩放回原始图像尺寸
    original_boxes = boxes_no_pad / scale
    
    # 再次裁剪，确保坐标在原始图像范围内
    original_boxes[:, [0, 2]] = original_boxes[:, [0, 2]].clamp(min=0, max=orig_w)
    original_boxes[:, [1, 3]] = original_boxes[:, [1, 3]].clamp(min=0, max=orig_h)

    return original_boxes

# 假设 box_cxcywh_to_xyxy 函数已定义
def box_cxcywh_to_xyxy(x):
    """
    将 (center_x, center_y, width, height) 格式的边界框转换为 (xmin, ymin, xmax, ymax) 格式。
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

# --- [修改后的 postprocess 函数] ---
def postprocess(predictions, original_size, target_size=(640, 640), threshold=0.7):
    """
    后处理模型输出，进行过滤、转换和缩放。

    参数:
    - predictions (dict): 模型的原始输出。
    - original_size (tuple): 原始图像的 (height, width)。
    - target_size (tuple): 模型输入的图像尺寸 (height, width)，对应于 PadIfNeeded 的尺寸。
    - threshold (float): 置信度阈值。
    """
    logits = predictions['pred_logits'][0]
    boxes = predictions['pred_boxes'][0]
    
    # 过滤和转换，这部分逻辑不变
    probs = logits.softmax(-1)
    scores, labels = probs[:, :-1].max(-1)
    keep = scores > threshold
    
    final_boxes_cxcywh = boxes[keep]  # 这是 cxcywh 格式，且坐标是相对于 target_size 归一化的
    final_scores = scores[keep]
    final_labels = labels[keep]
    
    # 将归一化的 cxcywh 转换为 xyxy
    final_boxes_xyxy_normalized = box_cxcywh_to_xyxy(final_boxes_cxcywh)
    
    # 将归一化的 xyxy 坐标乘以目标尺寸，得到在 640x640 画布上的绝对坐标
    target_h, target_w = target_size
    scale_fct = torch.tensor([target_w, target_h, target_w, target_h], device=final_boxes_xyxy_normalized.device)
    scaled_boxes_on_target = final_boxes_xyxy_normalized * scale_fct
    
    # --- [核心修改] ---
    # 使用新的函数将 640x640 画布上的坐标映射回原始图像
    final_scaled_boxes = rescale_boxes_after_padding(scaled_boxes_on_target, original_size, target_size)
    
    return final_scores, final_labels, final_scaled_boxes

# visualize_predictions_cv2 函数无需修改，因为它接收的已经是正确的坐标了。
def visualize_predictions_cv2(image_path, scores, labels, boxes, class_names, output_path="show.jpg"):
    """
    在原始图像上使用 OpenCV 绘制预测结果并保存到本地。
    """
    # 读取原始图像 (OpenCV 默认使用 BGR 格式)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not read image at {image_path}")
        return

    print(f"Drawing {len(boxes)} detections on the image using OpenCV...")
    
    # 为每个检测框进行绘制
    for score, label, box in zip(scores, labels, boxes):
        # 将 PyTorch 张量转换为整数坐标的 NumPy 数组
        box_int = box.cpu().numpy().astype(int)
        xmin, ymin, xmax, ymax = box_int
        
        # 定义颜色和字体
        box_color = (0, 0, 255)  # BGR 格式的红色
        text_color = (255, 255, 255) # BGR 格式的白色
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        # 1. 绘制边界框
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), box_color, thickness)
        
        # 2. 准备标签文本
        class_name = class_names[label.item()]
        label_text = f'{class_name}: {score:.2f}'
        
        # 3. 为文本创建一个背景框，以提高可读性
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness - 1)
        bg_rect_start = (xmin, ymin - text_height - baseline)
        bg_rect_end = (xmin + text_width, ymin)
        cv2.rectangle(image, bg_rect_start, bg_rect_end, box_color, -1)
        
        # 4. 在背景框上绘制文本
        text_origin = (xmin, ymin - baseline)
        cv2.putText(image, label_text, text_origin, font, font_scale, text_color, thickness - 1)

    # 保存最终的图像
    cv2.imwrite(output_path, image)
    print(f"Visualization saved successfully to: {output_path}")