import pkg_resources

# 확인할 패키지 목록
requirements_str = """
mujoco==2.2.0
gym==0.25.2
tensorboard==2.10.0
tensorboardX==2.5.1
matplotlib==3.5.3
ipython==7.34.0
moviepy==1.0.3
pyvirtualdisplay==3.0
torch==1.13.1
opencv-python==4.6.0.66
ipdb==0.13.9
swig==4.0.2
box2d-py==2.3.8
tqdm==4.66.1
pyyaml==6.0.1
"""

# 설치된 패키지들의 정보를 가져옵니다.
installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

print("--- 파이썬 패키지 설치 현황 확인 ---")
all_ok = True

# 목록에 있는 각 패키지를 확인합니다.
for line in requirements_str.strip().split('\n'):
    try:
        package_name, required_version = line.strip().split('==')
        
        # 패키지가 설치되어 있는지 확인
        if package_name.lower() in installed_packages:
            installed_version = installed_packages[package_name.lower()]
            # 버전이 일치하는지 확인
            if installed_version == required_version:
                print(f"✅ {package_name}: 일치 (버전: {installed_version})")
            else:
                print(f"❌ {package_name}: 버전 불일치 (필요: {required_version}, 설치됨: {installed_version})")
                all_ok = False
        else:
            print(f"❌ {package_name}: 설치되지 않음")
            all_ok = False
            
    except ValueError:
        print(f"⚠️ '{line}' 라인 형식이 잘못되었습니다. '패키지명==버전' 형식인지 확인하세요.")
        all_ok = False

print("\n--- 확인 완료 ---")
if all_ok:
    print("🎉 모든 패키지가 필요한 버전으로 올바르게 설치되었습니다!")
else:
    print("❗️ 일부 패키지가 설치되지 않았거나 버전이 다릅니다. 위 로그를 확인하세요.")