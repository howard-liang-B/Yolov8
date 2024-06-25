import torch

# 檢查PyTorch是否安裝
if torch.__version__:
    print("PyTorch 安裝成功，版本:", torch.__version__)
    
    if torch.cuda.is_available():
        print("CUDA 可用")
    else:
        print("CUDA 不可用")