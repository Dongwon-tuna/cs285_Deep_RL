import torch

# CUDA 사용 가능한지 여부
cuda_available = torch.cuda.is_available()
print(f"CUDA 사용 가능 여부: {cuda_available}")

if cuda_available:
    print(f"사용 중인 GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 총 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e6:.2f} MB")

    # 간단한 CUDA 연산 테스트
    a = torch.randn(3, 3).cuda()
    b = torch.randn(3, 3).cuda()
    c = torch.matmul(a, b)
    print("CUDA에서 행렬 곱 결과:")
    print(c)
