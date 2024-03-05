<h2>YOLO History Timeline</h2>

<!-- 使用img標籤插入圖片，並使用width和height屬性調整大小 -->
<img src="https://github.com/howard-liang-B/Yolov8-Project-by-Howard/assets/132919105/3cfc4efa-08b7-4656-9c6d-06cdc4f6bb14" alt="YOLO History Timeline" width="500" height="300">

</body>
</html>

## Colab
[![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1kTU1MA51Hdwt4xGxyyLr2tcZN10Y3Pgh/view?usp=sharing)

## Step_1 : Label
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






