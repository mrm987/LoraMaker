#!/usr/bin/env python3
"""
LoraMaker 설치 검증 스크립트
필요한 라이브러리와 GPU가 올바르게 설정되었는지 확인
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_python_version():
    """Python 버전 체크"""
    print("📌 Python 버전 체크...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ❌ Python 3.10 이상 필요 (현재: {version.major}.{version.minor})")
        return False


def check_torch():
    """PyTorch 설치 확인"""
    print("\n📌 PyTorch 체크...")
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
        return True
    except ImportError:
        print("  ❌ PyTorch가 설치되지 않았습니다.")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return False


def check_cuda():
    """CUDA 사용 가능 여부 확인"""
    print("\n📌 CUDA 체크...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA {torch.version.cuda}")
            print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("  ❌ CUDA를 사용할 수 없습니다.")
            print("     NVIDIA GPU 드라이버를 설치해주세요.")
            return False
    except Exception as e:
        print(f"  ❌ CUDA 체크 실패: {e}")
        return False


def check_dependencies():
    """주요 의존성 체크"""
    print("\n📌 의존성 체크...")
    deps = {
        "transformers": "Hugging Face Transformers",
        "diffusers": "Diffusers",
        "accelerate": "Accelerate",
        "safetensors": "SafeTensors",
        "gradio": "Gradio",
        "PIL": "Pillow",
        "yaml": "PyYAML"
    }

    all_ok = True
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} 설치 필요")
            all_ok = False

    return all_ok


def check_xformers():
    """xformers 체크 (선택적)"""
    print("\n📌 xformers 체크 (선택적)...")
    try:
        import xformers
        print(f"  ✅ xformers {xformers.__version__}")
        print("     (메모리 최적화 활성화)")
        return True
    except ImportError:
        print("  ⚠️  xformers 미설치 (선택적, 메모리 최적화에 도움)")
        print("     pip install xformers")
        return None  # 필수 아님


def check_gpu_specs():
    """GPU 사양 체크"""
    print("\n📌 GPU 사양 체크...")
    try:
        from backend.utils.gpu_utils import get_gpu_info, check_vram_requirements

        info = get_gpu_info()
        if info["available"]:
            print(f"  GPU: {info['name']}")
            print(f"  VRAM: {info['total_vram_gb']} GB")

            # VRAM 요구사항 체크
            ok, msg = check_vram_requirements(12.0)
            if ok:
                print(f"  ✅ {msg}")
                return True
            else:
                print(f"  ⚠️  {msg}")
                print("     학습이 느리거나 실패할 수 있습니다.")
                return None
        else:
            print("  ❌ GPU 사용 불가")
            return False
    except Exception as e:
        print(f"  ⚠️  GPU 정보 확인 실패: {e}")
        return None


def check_config_files():
    """설정 파일 존재 확인"""
    print("\n📌 설정 파일 체크...")
    config_file = project_root / "configs" / "training_presets.yaml"

    if config_file.exists():
        print(f"  ✅ {config_file}")
        return True
    else:
        print(f"  ❌ {config_file} 파일이 없습니다.")
        return False


def main():
    """메인 검증 함수"""
    print("=" * 50)
    print("LoraMaker 설치 검증")
    print("=" * 50)

    checks = [
        check_python_version(),
        check_torch(),
        check_cuda(),
        check_dependencies(),
        check_config_files(),
    ]

    # 선택적 체크
    optional_checks = [
        check_xformers(),
        check_gpu_specs(),
    ]

    print("\n" + "=" * 50)
    print("검증 결과")
    print("=" * 50)

    required_pass = all(c for c in checks if c is not None)
    optional_pass = all(c for c in optional_checks if c is not None and c is not False)

    if required_pass:
        print("✅ 필수 요구사항 충족!")
        if optional_pass:
            print("✅ 선택적 최적화 완료!")
            print("\n🎉 LoraMaker를 사용할 준비가 되었습니다!")
            print("\n실행 방법:")
            print("  python ui/gradio_app.py")
        else:
            print("⚠️  일부 최적화 미적용 (학습 가능하지만 느릴 수 있음)")
            print("\n실행 방법:")
            print("  python ui/gradio_app.py")
        return 0
    else:
        print("❌ 일부 요구사항 미충족")
        print("\nrequirements.txt를 설치해주세요:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
