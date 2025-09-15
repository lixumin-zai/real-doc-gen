
from lightning_model import LitRTDETR
from dataset import YoloDetectionDataset
from utils import collate_fn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
import torch
from utils import box_cxcywh_to_xyxy, postprocess, visualize_predictions_cv2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def main():
    # --- B. Setup Training ---
    NUM_CLASSES = 23
    BATCH_SIZE = 32
    IMG_SIZE = 640 # Use a smaller size for faster example training

    # --- C. Define Augmentations using Albumentations ---
    # Note: BboxParams format must match your label format ('yolo')
    # and you must specify which field in the transform's output corresponds to labels.
    # train_transform = A.Compose([
    #     A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    #     A.HorizontalFlip(p=0.5),
    #     A.ColorJitter(p=0.3),
    #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ToTensorV2(),
    # ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

    # val_transform = A.Compose([
    #     A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ToTensorV2(),
    # ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

    train_transform = A.Compose([
        A.LongestMaxSize(max_size=640, p=1.0),
        A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=1.0),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1, rotate_limit=0,
            p=0.7, border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114)
        ),
        # A.HorizontalFlip(p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.7),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8, fill_value=(114,114,114), p=0.5),
        # A.Blur(blur_limit=(3, 7), p=0.1),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'], min_visibility=0.1))

    val_transform = A.Compose([
        A.LongestMaxSize(max_size=640, p=1.0),
        A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=1.0),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'], min_visibility=0.1))


    # --- D. Create Datasets and DataLoaders ---
    img_dir_train = "/store/lixumin/layout-data/0828/train/images"
    label_dir_train = "/store/lixumin/layout-data/0828/train/labels"

    train_dataset = YoloDetectionDataset(
        img_dir=img_dir_train,
        label_dir=label_dir_train,
        transform=train_transform
    )

    # For validation, we'll just reuse the same data with different transforms
    img_dir_train = "/home/lixumin/project/pre-ocr/paper-cut/datas/test/images"
    label_dir_train = "/home/lixumin/project/pre-ocr/paper-cut/datas/test/labels"

    val_dataset = YoloDetectionDataset(
        img_dir=img_dir_train,
        label_dir=label_dir_train,
        transform=val_transform
    )

    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4 # Use multiple workers for faster data loading
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=4
    )

    # --- E. Initialize Model and Trainer ---
    model = LitRTDETR(num_classes=NUM_CLASSES, lr=2e-4)
    progress_bar = RichProgressBar()
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='auto',
        devices=[0, 1],
        callbacks=[progress_bar] # <--- 添加 callback
        # fast_dev_run=True, # Uncomment for a quick test run
    )

    print("Starting training with YOLO dataset...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Training complete!")

       # 进行一次推理
    model.eval()

    
    
import cv2

def test():
    CHECKPOINT_PATH = "/home/lixumin/project/doclayout-pipeline/rtdetr/lightning_logs/version_35/checkpoints/epoch=49-step=82750.ckpt"
    IMG_SIZE = 640
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LitRTDETR.load_from_checkpoint(CHECKPOINT_PATH)

    model.eval()
    model.to(DEVICE)

    IMAGE_PATH = "/home/lixumin/project/doclayout-pipeline/images/61b4b264-2e47-4cbd-8133-175eeceedf4b.jpg"

    image = cv2.imread(IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    test_transform = A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    test_transform = A.Compose([
        A.LongestMaxSize(max_size=640, p=1.0),
        A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=1.0),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

    original_h, original_w, _ = image.shape
    processed = test_transform(image=image)
    input_tensor = processed['image']
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    print(input_tensor.shape)

    with torch.no_grad():
        predictions = model(input_tensor)
    
    print(predictions)
    # # --- 6. 查看输出 ---
    # print("\n--- Inference Output ---")
    # # 输出的 logits 形状: [batch_size, num_queries, num_classes + 1]
    # print("Logits shape:", predictions['pred_logits'].shape) 
    # # 输出的 boxes 形状: [batch_size, num_queries, 4]
    # print("Boxes shape:", predictions['pred_boxes'].shape)

    # # 你可以进一步处理这些输出来获得最终的边界框和类别
    # # 例如，找到置信度最高的预测
    # logits = predictions['pred_logits'].softmax(-1)[0] # [num_queries, num_classes + 1]
    # boxes = predictions['pred_boxes'][0] # [num_queries, 4]

    # # 过滤掉 "no object" 类别 (假设它是最后一个类别)
    # # 假设你的 num_classes=1，那么类别0是你的目标，类别1是背景
    # scores, labels = logits[:, :-1].max(-1)

    # # 设置一个置信度阈值
    # keep = scores > 0.3
    # final_boxes = boxes[keep]
    # final_scores = scores[keep]
    # final_labels = labels[keep]

    # print(f"\nFound {len(final_boxes)} objects with confidence > 0.7")
    # for i in range(len(final_boxes)):
    #     print(f"  - Box: {final_boxes[i].cpu().numpy()}, Score: {final_scores[i]:.4f}, Label: {final_labels[i]}")

    CLASS_NAMES = ["question"]*23
    OUTPUT_IMAGE_PATH = "show.jpg"
    print("Post-processing results...")
    scores, labels, boxes = postprocess(predictions, (original_h, original_w), threshold=0.5)
    
    # print(labels, boxes)
    if len(boxes) == 0:
        print(f"No objects found with confidence > {0.3}")
        # 即使没有检测到对象，也保存原图以确认流程正常
        cv2.imwrite(OUTPUT_IMAGE_PATH, image)
        print(f"No objects detected. Original image saved to {OUTPUT_IMAGE_PATH}")
    else:
        # --- [修改] 调用新的 OpenCV 可视化函数 ---
        visualize_predictions_cv2(IMAGE_PATH, scores, labels, boxes, CLASS_NAMES, output_path=OUTPUT_IMAGE_PATH)

    # if len(boxes) == 0:
    #     print(f"No objects found with confidence > {CONFIDENCE_THRESHOLD}")
    # else:
    #     visualize_predictions(IMAGE_PATH, scores, labels, boxes, CLASS_NAMES)

if __name__ == "__main__":
    # main()

    test()