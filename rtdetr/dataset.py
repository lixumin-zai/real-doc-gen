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

# --- Define constants for visualization ---
# IMPORTANT: Update this list with your actual class names
CLASS_NAMES = ['class_0', 'class_1', 'class_2'] # Example: ['car', 'person', 'traffic_light']
# These should be the same as in your transform pipeline
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def unnormalize_image(tensor_image):
    """Reverses the normalization on a tensor image."""
    image = tensor_image.clone()
    for t, m, s in zip(image, MEAN, STD):
        t.mul_(s).add_(m)  # Reverse normalization: image = image * std + mean
    # Convert from C, H, W to H, W, C for displaying
    image = image.permute(1, 2, 0).numpy()
    # Clip values to be between 0 and 1
    image = np.clip(image, 0, 1)
    return image

def visualize_sample(image_tensor, target, class_names):
    """
    Visualizes a single sample (image and its bounding boxes).

    Args:
        image_tensor (torch.Tensor): The transformed image tensor (C, H, W).
        target (dict): A dictionary containing 'boxes' and 'labels'.
        class_names (list): A list of class names for displaying labels.
    """
    # 1. Un-normalize the image for display
    image_np = unnormalize_image(image_tensor)
    height, width, _ = image_np.shape

    # 2. Create plot
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image_np)
    ax.set_title("Augmented Image with Bounding Boxes")
    ax.axis('off')

    # 3. Draw bounding boxes and labels
    boxes = target['boxes'].cpu().numpy()
    labels = target['labels'].cpu().numpy()

    if len(boxes) > 0:
        for box, label_id in zip(boxes, labels):
            # Convert YOLO format (cx, cy, w, h) to (xmin, ymin) for the patch
            cx, cy, w, h = box
            xmin = (cx - w / 2) * width
            ymin = (cy - h / 2) * height
            box_width = w * width
            box_height = h * height

            # Get class name and a random color
            label_name = class_names[label_id]
            color = np.random.rand(3,) # Random color for each class

            # Create a Rectangle patch
            rect = patches.Rectangle(
                (xmin, ymin), box_width, box_height,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )

            # Add the patch to the Axes
            ax.add_patch(rect)

            # Add label text
            ax.text(
                xmin, ymin - 5,
                f'{label_name}',
                bbox=dict(facecolor=color, alpha=0.5, pad=0),
                color='white',
                fontsize=10
            )

    plt.savefig("./show.jpg")


class YoloDetectionDataset(Dataset):
    """
    A PyTorch Dataset for loading object detection data in YOLO format.
    ... (Your class code remains unchanged) ...
    """
    def __init__(self, img_dir, label_dir, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)

        try:
            image = cv2.imread(img_path)
            if image is None:
                raise IOError(f"Could not read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Skipping to next available image.")
            # Fallback to a known good image if the current one fails
            idx = (idx + 1) % len(self.img_files)
            img_name = self.img_files[idx]
            img_path = os.path.join(self.img_dir, img_name)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(self.label_dir, label_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, cx, cy, w, h = map(float, parts)
                        lower_bound, upper_bound = 1e-6, 1.0
                        cx_clean = min(max(cx, lower_bound), upper_bound)
                        cy_clean = min(max(cy, lower_bound), upper_bound)
                        w_clean  = min(max(w,  lower_bound), upper_bound)
                        h_clean  = min(max(h,  lower_bound), upper_bound)
                        labels.append(int(class_id))
                        boxes.append([cx_clean, cy_clean, w_clean, h_clean])

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        target = {}
        if self.transform:
            try:
                transformed = self.transform(image=image, bboxes=boxes, category_ids=labels)
                image = transformed['image']
                boxes = np.array(transformed['bboxes'], dtype=np.float32)
                labels = np.array(transformed['category_ids'], dtype=np.int64)
            except Exception as e:
                print(f"Could not apply transform on {img_name}: {e}")
                # Fallback to a simple resize and ToTensor if augmentation fails
                h, w, _ = image.shape
                fallback_transform = A.Compose([
                    A.Resize(height=640, width=640), # Use a fixed size for consistency
                    A.Normalize(mean=MEAN, std=STD),
                    ToTensorV2()
                ])
                image = fallback_transform(image=image)['image']
                # Bboxes might be lost, so we create an empty target
                boxes, labels = np.array([]), np.array([])


        if len(boxes) > 0:
            target['boxes'] = torch.from_numpy(boxes)
            target['labels'] = torch.from_numpy(labels)
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)

        return image, target


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)

def collate_fn(batch):
    images, targets = list(zip(*batch))
    images = torch.stack(images, 0)
    return images, targets


yolo_like_train_transform = A.Compose([
    A.LongestMaxSize(max_size=640, p=1.0),
    A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=1.0),
    A.ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.1, rotate_limit=0,
        p=0.7, border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114)
    ),
    # A.HorizontalFlip(p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    # A.Blur(blur_limit=(3, 7), p=0.1),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'], min_visibility=0.1))


if __name__ == "__main__":
    data = YoloDetectionDataset(
        "/store/lixumin/layout-data/0828/train/images",
        "/store/lixumin/layout-data/0828/train/labels",
        yolo_like_train_transform
    )

    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=4, # Increase batch size to see multiple examples
        num_workers=1,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        shuffle=True,
        worker_init_fn=seed_worker,
    )

    # Fetch one batch from the dataloader
    images, targets = next(iter(data_loader))

    print(f"Batch of images shape: {images.shape}")
    print(f"Number of targets in batch: {len(targets)}")

    # Visualize each image in the batch
    for i in range(images.shape[0]):
        image_tensor = images[i]
        target = targets[i]
        visualize_sample(image_tensor, target, CLASS_NAMES*8)
        # break