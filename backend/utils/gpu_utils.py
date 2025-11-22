"""
GPU 유틸리티
VRAM 체크, 배치 크기 자동 조정 등
"""

import torch
from typing import Optional


def get_gpu_info() -> dict:
    """
    GPU 정보 가져오기

    Returns:
        dict: GPU 정보 (이름, VRAM 등)
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "message": "CUDA GPU를 사용할 수 없습니다."
        }

    gpu_id = 0  # 첫 번째 GPU 사용
    props = torch.cuda.get_device_properties(gpu_id)

    total_vram_gb = props.total_memory / (1024**3)
    free_vram_gb = (props.total_memory - torch.cuda.memory_allocated(gpu_id)) / (1024**3)

    return {
        "available": True,
        "name": props.name,
        "total_vram_gb": round(total_vram_gb, 2),
        "free_vram_gb": round(free_vram_gb, 2),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__
    }


def check_vram_requirements(required_vram_gb: float = 12.0) -> tuple[bool, str]:
    """
    VRAM 요구사항 체크

    Args:
        required_vram_gb: 필요한 VRAM (GB)

    Returns:
        (통과 여부, 메시지)
    """
    info = get_gpu_info()

    if not info["available"]:
        return False, info["message"]

    if info["total_vram_gb"] < required_vram_gb:
        return False, (
            f"VRAM 부족: {info['total_vram_gb']}GB (최소 {required_vram_gb}GB 필요)\n"
            f"GPU: {info['name']}"
        )

    return True, f"✅ GPU 준비 완료: {info['name']} ({info['total_vram_gb']}GB VRAM)"


def recommend_batch_size(resolution: int = 512) -> int:
    """
    VRAM에 따른 배치 크기 추천

    Args:
        resolution: 학습 해상도

    Returns:
        추천 배치 크기
    """
    info = get_gpu_info()

    if not info["available"]:
        return 1

    vram_gb = info["total_vram_gb"]

    # 경험적 추천값
    if resolution <= 512:
        if vram_gb >= 24:
            return 4
        elif vram_gb >= 16:
            return 2
        else:
            return 1
    elif resolution <= 768:
        if vram_gb >= 24:
            return 2
        else:
            return 1
    else:  # 1024+
        if vram_gb >= 24:
            return 2
        else:
            return 1


def estimate_training_time(
    num_images: int,
    epochs: int,
    batch_size: int,
    steps_per_second: float = 2.0
) -> dict:
    """
    학습 시간 추정

    Args:
        num_images: 이미지 수
        epochs: 에폭 수
        batch_size: 배치 크기
        steps_per_second: 초당 스텝 수 (GPU 성능에 따라)

    Returns:
        추정 시간 정보
    """
    total_steps = (num_images * epochs) // batch_size
    total_seconds = total_steps / steps_per_second

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)

    return {
        "total_steps": total_steps,
        "estimated_seconds": total_seconds,
        "estimated_minutes": round(total_seconds / 60, 1),
        "formatted": f"{hours}시간 {minutes}분" if hours > 0 else f"{minutes}분"
    }


# 테스트
if __name__ == "__main__":
    print("=== GPU 정보 ===")
    info = get_gpu_info()
    for key, value in info.items():
        print(f"{key}: {value}")

    print("\n=== VRAM 체크 ===")
    ok, msg = check_vram_requirements(12.0)
    print(msg)

    print("\n=== 배치 크기 추천 ===")
    for res in [512, 768, 1024]:
        batch = recommend_batch_size(res)
        print(f"{res}px: batch_size={batch}")

    print("\n=== 학습 시간 추정 ===")
    estimate = estimate_training_time(num_images=20, epochs=10, batch_size=2)
    print(f"예상 시간: {estimate['formatted']}")
