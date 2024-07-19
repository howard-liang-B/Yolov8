<h2>YOLO History Timeline</h2>

<!-- 使用img標籤插入圖片，並使用width和height屬性調整大小 -->
<img src="https://github.com/howard-liang-B/Yolov8-Project-by-Howard/assets/132919105/3cfc4efa-08b7-4656-9c6d-06cdc4f6bb14" alt="YOLO History Timeline" width="500" height="300">

</body>
</html>

## YOLO 非常之簡單介紹
* YOLO（You Only Look Once）的作法是將輸入的影像切割成一個固定大小的網格（grid），這個網格通常是SxS大小的。每個網格都負責偵測該網格內可能存在的物體。如果一個物體的中心落入某個網格內，那麼這個網格就要負責去偵測該物體。
* bounding box(bbox) 是由，"x, y, w, h, confidence"，五個東西所組成，就是預測後的你在影像看到的框框。當一個網格內存在多個bbox時，每個bbox都會進行類別的機率預測。YOLO將選擇具有最高confidence的那個bbox作為最終的預測。
* 模型評估: Precision、Recall、Accuracy、F1-score ...
1. Precision：預測為目標物且實際上也確實為目標物的比例，可以得知當模型預測為目標物時，這個結果是不是準確的。
2. Recall：實際為目標物也確實被預測為目標物的比例，可以得知模型找出目標物的能力。
3. Accuracy : 準確度是指模型正確預測的樣本數占總樣本數的比例。
4. F1 score : 是 Precision 與 Recall 的調和平均數，調和平均值的計算方式是將數值的倒數取平均後再取倒數。
  <img src="readme images/Confusion_matrix_1.png" alt="混和矩陣" width="80%">  
參考資料: https://medium.com/ching-i/yolo-c49f70241aa7

## Colab
[![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1kTU1MA51Hdwt4xGxyyLr2tcZN10Y3Pgh/view?usp=sharing)

## Step_0 : Get dataset
1. 網路上找自己喜歡的照片
2. roboflow 上面有其他人公開的 dataset，都是可以下載的  
   https://universe.roboflow.com/
3. dataset 資料夾格式如下，從官方抓的  
   <img src="readme images/yolo_dataset_dir.png" alt="yolo_dataset_dir" width="60%">  
```
# 這邊是我放 YOLO dataset 的資料夾架構
dataset/
    train/
      images/
          |-- 0001.jpg
          |-- 0002.jpg
              ...
      labels/
          |-- 0001.txt
          |-- 0002.txt
              ...
    val/
      images/
          |-- 0003.jpg
          |-- 0004.jpg
              ...
      labels/
          |-- 0003.txt
          |-- 0004.txt
              ...
    test/
      images/
          |-- 0005.jpg
          |-- 0006.jpg
              ...
      labels/
          |-- 0005.txt
          |-- 0006.txt
              ...
```

## Step_1 : Label
* **為甚麼要對影像進行標記**  
  * 對影像進行物件標記，可以讓模型知道要偵測的物體"中心"和"長寬"，標記的格式輸出為 ".txt" 檔案，"cls, center_x, center_y, w, h"，x 和 y 是物件中心座標，w 和 h 是物件長寬，這些資訊除了是訓練的必要參數，也是評估模型需要的資訊。
* **標記格式**
  * 進行標記時在匯出檔案的同時，需要選擇YOLO格式，export的時候應該就會看到yolo的格式，選擇他就對了。
  * 下面的圖片可以看到我使用 Mask Sense 標記了一個臉，首先我們看到類別，我建立 class 0 為 girl，class 1 為 boy，因為照片被標記的臉部是女生，所以選擇girl。
  * 接著我們看到匯出的 YOLO 格式的".txt"檔案，[0 0.501778 0.279762 0.511979 0.522109] 第一個 0 代表的是 class 0 也就是 girl
      * [0.501778 0.279762] = [center_x, center_y]
      * [0.511979 0.522109] = [w, h]
  * 那至於這邊為甚麼是小數點呢，因為他這邊做了正規化(Normalization)或是標準化，Normalization是應用在機器學習中，通過讓不同單位的資料變成同單位，也就是讓資料的尺度相同。以這張 "bon1.png" 為例，圖片的 [width, height] = [1391, 1454] (自己電腦查的)，我把滑鼠移動到 bbox 的中心(因為是滑鼠移動，所以等等的計算有誤差)
      * 1391 * 0.501778 = 697.97(接近下圖的x:700)
      * 1454 * 0.279762 = 406.77(接近下圖的y:406)。
  <img src="readme images/label_exp_1.png" alt="標記範例" width="80%">  
* **Min-Max 正規化 (Min-Max Normalization)**: 將特徵縮放到一個指定的最小值和最大值之間，通常是 [0, 1] 或 [-1, 1]。  
  <img src="readme images/Min-Max Normalization.png" alt="Min-Max 正規化公式" width="30%">  
* **以下是進行標記的網站**
1. `labelImg` <br>
https://github.com/HumanSignal/labelImg/releases

2. `Make Sense` <br>
https://www.makesense.ai/

3. `CVAT`  <br> 
https://www.cvat.ai/

4. `roboflow` :thumbsup: <br>
https://roboflow.com/

## Step_2 : Setup GPU to train model

1. `cuda` <br>
* CUDA Toolkit Archive : https://developer.nvidia.com/cuda-toolkit-archive
* Tutorial : https://medium.com/ching-i/win10-%E5%AE%89%E8%A3%9D-cuda-cudnn-%E6%95%99%E5%AD%B8-c617b3b76deb

2. `Pytorch` <br>
* https://pytorch.org/get-started/locally/
* https://pytorch.org/get-started/previous-versions/

```
# NVIDIA CUDA Compiler(nvcc)
# confirm if you install successful, and check version !
!nvcc --version
!nvcc -V
```

```
import torch

# 檢查PyTorch是否安裝
if torch.__version__:
    print("PyTorch 安裝成功，版本:", torch.__version__)
    
    if torch.cuda.is_available():
        print("CUDA 可用")
    else:
        print("CUDA 不可用")
```

## Step_3 : Download model
```
$ pip install ultralytics 
$ yolo predict model = yolov8n.pt 
$ yolo predict model = yolov8s.pt 
```






