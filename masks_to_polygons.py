import os
import cv2


input_dir = 'datasets/SegmentationClass'
output_dir = 'datasets/temp_labels'

for j in os.listdir(input_dir):
    image_path = os.path.join(input_dir, j)
    # load the binary mask and get its contours
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    H, W = mask.shape
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours 是一個包含檢測到的輪廓的列表。
    # hierarchy (next, previous, first child, parent)
    # mask 是輸入的二值化影像。
    # cv2.RETR_EXTERNAL 表示只檢測最外部的輪廓，忽略內部的洞穴。
    # cv2.CHAIN_APPROX_SIMPLE 表示簡單地壓縮水平、垂直和對角線方向上的冗餘點，以節省記憶空間。

    # convert the contours to polygons
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            polygon = []
            for point in cnt:
                x, y = point[0]
                polygon.append(x / W)
                polygon.append(y / H)
            polygons.append(polygon)

    # print the polygons
    with open('{}.txt'.format(os.path.join(output_dir, j)[:-4]), 'w') as f:
        for polygon in polygons:
            for p_, p in enumerate(polygon):
                if p_ == len(polygon) - 1:
                    f.write('{}\n'.format(p))
                elif p_ == 0:
                    f.write('0 {} '.format(p))
                else:
                    f.write('{} '.format(p))

        f.close()
