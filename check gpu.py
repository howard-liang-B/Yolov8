import torch

# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()

print(cuda_available)
cuda_version = torch.version.cuda
print(f"CUDA 版本: {cuda_version}")

# 获取cuDNN版本
cudnn_version = torch.backends.cudnn.version()
print(f"cuDNN 版本: {cudnn_version}")