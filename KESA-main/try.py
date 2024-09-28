import torch
print(torch.cuda.is_available())  # 确认CUDA是否可用
print(torch.cuda.device_count())  # 确认GPU的数量
print(torch.cuda.current_device())  # 当前使用的GPU
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # 当前GPU的名称
