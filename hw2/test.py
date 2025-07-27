import pkg_resources
import subprocess
import torch

import numpy as np
print(np.__version__)


# 확인할 패키지와 원하는 버전
required_packages = {
    "swig": "4.0.2",
    "mujoco": "2.2.0",
    "gym": "0.25.2",
    "tensorboard": "2.10.0",
    "tensorboardX": "2.5.1",
    "matplotlib": "3.5.3",
    "ipython": "7.34.0",
    "moviepy": "1.0.3",
    "pyvirtualdisplay": "3.0",
    "torch": "1.13.1",
    "opencv-python": "4.6.0.66",
    "ipdb": "0.13.9"
}

print("="*50)
print("🔍 패키지 설치 및 버전 확인")
print("="*50)

for pkg, expected_version in required_packages.items():
    try:
        installed_version = pkg_resources.get_distribution(pkg).version
        if installed_version == expected_version:
            print(f"[✔] {pkg}=={expected_version} ✅")
        else:
            print(f"[!] {pkg}: 설치된 버전 {installed_version} ≠ 요구 버전 {expected_version}")
    except pkg_resources.DistributionNotFound:
        print(f"[✘] {pkg}: 설치되지 않음 ❌")

print("\n" + "="*50)
print("🖥️ PyTorch CUDA / GPU 상태 확인")
print("="*50)

if torch.cuda.is_available():
    print("✅ GPU 사용 가능 (CUDA available)")
    print("→ PyTorch CUDA 버전:", torch.version.cuda)
    print("→ 그래픽카드 모델:", torch.cuda.get_device_name(0))
else:
    print("❌ GPU 사용 불가 (torch는 CPU만 인식 중)")
    print("→ PyTorch CUDA 버전:", torch.version.cuda if torch.version.cuda else "N/A")

# nvidia-smi 체크 (가능한 경우)
print("\n" + "="*50)
print("📊 시스템 GPU 정보 (nvidia-smi)")
print("="*50)

try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
    print(result.stdout)
except FileNotFoundError:
    print("nvidia-smi 명령어를 찾을 수 없습니다. NVIDIA 드라이버가 없거나 설정되지 않았습니다.")
except subprocess.CalledProcessError:
    print("nvidia-smi 실행 중 오류 발생. 드라이버 미설치 또는 비활성일 수 있습니다.")
