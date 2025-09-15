from pp_layout import Layout
import feishu_sdk
from feishu_sdk.sheet import FeishuSheet, FeishuImage
import os
app_id, app_key = "cli_a12ac5906d7c900e", "jSFDBwE3MbtXA6o9fSsOHdGbd3pVj1WY"
feishu_sdk.login(app_id, app_key)
import numpy as np
import cv2
from PIL import Image, ImageDraw
from io import BytesIO
def feishu_test():
    sheet_token, sheet_id = "Bb3kslUvehQ4bLtoTwZcZMBLnPg", "MIali7"
    sheet = FeishuSheet(sheet_token, sheet_id)
    image_col = "D"
    result_col = "BJ"
    has_figure_col = "BK"

    layout = Layout()

    for i in range(min(sheet.rows+1, 10000)):
        if i < 3:
            continue
        try:
            # if sheet[f"{result_col}{i}"]:
            #     continue
            if isinstance(sheet[f"{image_col}{i}"], FeishuImage):
                image_bytes = sheet[f"{image_col}{i}"].image_bytes
                nparr = np.frombuffer(image_bytes, np.uint8)
                img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                info = layout([img_cv])[0]
                if info:
                    has_figure = 1
                else:
                    has_figure = 0

            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            draw = ImageDraw.Draw(image)
            if has_figure:
                for box in info:
                    if box[4]!=1:
                        continue
                    x1, y1, x2, y2 = box[:4]
                    draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            temp = BytesIO(image_bytes)
            image.save(temp, format='JPEG')
            new_image_bytes = temp.getvalue()
            sheet[f"{result_col}{i}"] = FeishuImage(img_bytes=new_image_bytes)

            sheet[f"{has_figure_col}{i}"] = "1" if has_figure else "0"
                # sheet[f"{has_figure_col}{i}"] = ",".join(labels)
        except:
            import traceback
            traceback.print_exc()
            break

if __name__ == "__main__":
    feishu_test()