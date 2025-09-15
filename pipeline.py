import os
import uuid
import cv2
import json
import tqdm
from PIL import Image, ImageDraw
import numpy as np
import multiprocessing as mp
from functools import partial
import random

# 假设 dewarp 和 pp_layout 是你本地的模块
from dewarp import DocDewarp
from pp_layout import Layout

class DataGen:
    def __init__(self, save_root_path):
        """
        初始化处理器，包括文档矫正和布局分析模型。
        """
        self.doc_dewarp = DocDewarp()
        self.layout = Layout()

        self.save_path = save_root_path
        # 确保输出目录存在
        os.makedirs(os.path.join(self.save_path, "train/images/"), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "train/labels/"), exist_ok=True)
        
    def __call__(self, images):
        """
        处理一批图像。
        
        Args:
            images (list[PIL.Image]): 一批待处理的PIL图像对象。
        """
        # 1. 文档图像矫正
        dewarped_images = self.doc_dewarp.batch_predict(images)

        # 2. 布局分析
        infos = self.layout(dewarped_images)

        # 3. 保存结果
        for image_np, info in zip(dewarped_images, infos):
            image_name = str(uuid.uuid4())
            # 将 numpy 数组转回 PIL Image 以便使用 cv2 保存（或直接用 cv2 保存）
            # cv2 需要 BGR 格式，而 PIL 是 RGB
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.save_path, f"train/images/{image_name}.jpg"), image_bgr)
            
            with open(os.path.join(self.save_path, f"train/labels/{image_name}.txt"), "w", encoding='utf-8') as f:
                for label_line in info:
                    f.write(label_line + "\n")

def find_image_paths_os(root_folder: str) -> list[str]:
    """
    使用 os.walk 递归查找指定文件夹及其子文件夹中所有图片的路径。
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    image_paths = []
    if not os.path.isdir(root_folder):
        print(f"错误: 路径 '{root_folder}' 不是一个有效的文件夹。")
        return []

    print("正在扫描图片文件...")
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(image_extensions):
                full_path = os.path.join(dirpath, filename)
                image_paths.append(full_path)
    print(f"扫描完成，共找到 {len(image_paths)} 张图片。")
    return image_paths
    
_data_gen_instance = None

def init_worker(save_root_path: str):
    """
    Initializer function for each worker process in the pool.
    This function is called once when a worker process starts.
    """
    global _data_gen_instance
    process_id = os.getpid()
    print(f"Initializing worker process ID: {process_id}...")
    # The expensive model loading happens here, ONCE per process.
    _data_gen_instance = DataGen(save_root_path)
    print(f"Worker {process_id} initialized successfully.")

def process_batch(image_paths_batch: list[str]) -> list[str]:
    """
    The main worker function. It now uses the pre-initialized global DataGen instance.
    Note that it no longer needs `save_root_path` as an argument.
    """
    # Access the global instance created by init_worker.
    global _data_gen_instance
    if _data_gen_instance is None:
        # This is a safeguard; it should not be triggered if the pool is set up correctly.
        raise RuntimeError("DataGen instance not initialized in worker process.")

    try:
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths_batch]
        _data_gen_instance(images)
        # Return this batch's collected labels for aggregation in the main process.
        return _data_gen_instance.layout.label
    except Exception as e:
        process_id = os.getpid()
        print(f"Process {process_id}: An error occurred while processing a batch: {e}. Skipping.")
        print(f"Problematic image paths in batch: {image_paths_batch}")
        return []

def main():
    save_root_path = "/store/lixumin/layout-data/0828"
    images_root_path = "/home/lixumin/project/data/Vary-600k/data/pdf_data"
    
    images_path = find_image_paths_os(images_root_path)
    random.shuffle(images_path)
    # images_path = images_path[:320]
    # Using a smaller slice for quick testing
    # images_path = images_path[:320] 
    
    if not images_path:
        print("未找到任何图片，程序退出。")
        return
        
    step = 32
    batches = [images_path[i: i+step] for i in range(0, len(images_path), step)]
    
    # Adjust based on your GPU memory and CPU cores.
    # If your models are GPU-heavy, you might be limited by VRAM.
    num_processes = 3
    print(f"启动 {num_processes} 个进程进行处理...")

    all_labels = []
    
    # --- MODIFIED: Pool creation ---
    # Use the `initializer` and `initargs` to set up each worker process.
    with mp.Pool(processes=num_processes,
                 initializer=init_worker,
                 initargs=(save_root_path,)) as pool:
        
        # `pool.imap` will now call `process_batch` for each item in `batches`.
        # `process_batch` already knows how to find its `DataGen` instance.
        for labels_from_process in tqdm.tqdm(pool.imap(process_batch, batches), total=len(batches)):
            all_labels.extend(labels_from_process)

    print("所有批次处理完毕。")

    unique_labels = sorted(list(set(all_labels)))
    with open(os.path.join(save_root_path, "labels.json"), "w", encoding='utf-8') as f:
        json.dump(unique_labels, f, indent=4)
        
    print(f"标签已保存到 {os.path.join(save_root_path, 'labels.json')}")

# `show` function remains the same.
def show():
    # ... (code is unchanged) ...
    path = "/store/lixumin/layout-data/0828/train"
    image_dir = os.path.join(path, "images")
    if not os.path.exists(image_dir) or not os.listdir(image_dir):
        print(f"在 {image_dir} 中未找到任何图片用于展示。")
        return
    for i in os.listdir(image_dir):
        image_path = os.path.join(path, "images", i)
        print(f"显示: {image_path}")
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        label_path = image_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
        if not os.path.exists(label_path):
            print(f"警告: 找不到对应的标签文件 {label_path}")
            continue
        with open(label_path, "r") as f:
            infos = f.readlines()
        origin_w, origin_h = image.size
        for info in infos:
            try:
                label, x_center, y_center, w, h = info.strip().split()
                label, x_center, y_center, w, h = int(label), float(x_center), float(y_center), float(w), float(h)
                x1 = (x_center - w / 2) * origin_w
                y1 = (y_center - h / 2) * origin_h
                x2 = (x_center + w / 2) * origin_w
                y2 = (y_center + h / 2) * origin_h
                draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            except ValueError as e:
                print(f"解析行失败: '{info.strip()}', 错误: {e}")
        save_path = "show.jpg"
        image.save(save_path, format="JPEG")
        print(f"可视化结果已保存到 {save_path}")
        break

if __name__ == "__main__":
    main()
    # show()