# git clone https://github.com/ultralytics/ultralytics
# cd ultralytics
# pip install -e ultralytics==8.0.20

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

model = YOLO('C:\\Users\\user\\Desktop\\Yolov8 Project\\Yolov8_segmentation\\exp_013\\weights\\best.pt')

results = model.predict(              # Step 2 : 使用 YOLOv8 模型進行物件偵測
    source = 'datasets/robo_yolov8_v2/valid/images',        # - 指定代偵測影像子目錄
    conf = 0.9,             # - 偵測信心水準門限值
    save = True,            # - 儲存偵測結果影像
    save_txt = True,        # - 儲存偵測結果物件位置(YOLO格式)
    save_conf = True,       # - 儲存物件偵測結果信心水準值
    save_crop = True,       # - 儲存擷取物件影像
    visualize = False,       # - 偵測過程特徵圖視覺化
    augment = True
)