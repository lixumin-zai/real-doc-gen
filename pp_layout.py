from paddleocr import LayoutDetection
from PIL import Image
import numpy as np

class Layout:
    def __init__(self):
        self.model = LayoutDetection(model_name="PP-DocLayout_plus-L", threshold=0.5, layout_unclip_ratio=1.0)

        self.label = {}

    def __call__(self, images):
        output = self.model.predict(images, batch_size=8, layout_nms=True)
        infos = []
        for image, res in zip(images, output):
            # res.save_to_img(save_path="./output/")
            origin_h, origin_w = image.shape[:2]
            boxes = res.json["res"]["boxes"]
            info = []
            for box in boxes:
                self.label[box["cls_id"]] = box["label"]
                x1, y1, x2, y2 = box["coordinate"]
                cx, cy, w, h = (x1+x2)/2/origin_w, (y1+y2)/2/origin_h, (x2-x1)/origin_w, (y2-y1)/origin_h 
                info.append(f"{box['cls_id']} {cx} {cy} {w} {h}")
                # print(f"{box['cls_id']} {cx} {cy} {w} {h}")
            infos.append(info)

        return infos

    # def __call__(self, images):
    #     output = self.model.predict(images, batch_size=8, layout_nms=True)
    #     infos = []
    #     for image, res in zip(images, output):
    #         # res.save_to_img(save_path="./output/")
    #         origin_h, origin_w = image.shape[:2]
    #         boxes = res.json["res"]["boxes"]
    #         info = []
    #         for box in boxes:
    #             self.label[box["cls_id"]] = box["label"]
    #             x1, y1, x2, y2 = box["coordinate"]
    #             info.append([x1, y1, x2, y2, box["cls_id"]])
    #             # print(f"{box['cls_id']} {cx} {cy} {w} {h}")
    #         infos.append(info)

    #     return infos

if __name__ == "__main__":
    layout = Layout()

    image = Image.open("/home/lixumin/project/doclayout-pipeline/rtdetr/72da64a6b2f93e2d7070acd9ad7b3ffe.JPG").convert("RGB")
    image2 = Image.open("/store/lixumin/layout-data/0828/train/images/8bc488df-7f7f-42aa-9b4b-98186d28f0f8.jpg").convert("RGB")

    image_bytes = layout([np.array(image), np.array(image2)])   
    print(layout.label)