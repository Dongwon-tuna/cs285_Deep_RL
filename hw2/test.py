import torch
print(torch.__version__)             # 예: '2.3.0+cu128'
print(torch.cuda.is_available())     # True
print(torch.cuda.get_device_name(0)) # GPU 이름 출력
