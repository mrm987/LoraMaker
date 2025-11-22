"""
데이터셋 빌더
Kohya-ss 형식의 학습 데이터셋 생성
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """데이터셋 설정"""
    name: str
    images: List[str]
    captions: Dict[str, str]  # {image_path: caption}
    repeats: int = 10
    flip_aug: bool = False
    shuffle_caption: bool = True
    keep_tokens: int = 1


class DatasetBuilder:
    """Kohya-ss 형식 데이터셋 빌더"""

    def __init__(self, output_root: str = "training_data"):
        self.output_root = Path(output_root)

    def create_dataset_structure(
        self,
        dataset_name: str,
        repeats: int = 10
    ) -> Path:
        """
        Kohya-ss 데이터셋 폴더 구조 생성

        Structure:
        training_data/
        └── {repeats}_{dataset_name}/
            ├── image1.png
            ├── image1.txt
            ├── image2.png
            └── image2.txt

        Args:
            dataset_name: 데이터셋 이름
            repeats: 반복 횟수 (적은 이미지일수록 높게)

        Returns:
            데이터셋 디렉토리 경로
        """
        dataset_dir = self.output_root / f"{repeats}_{dataset_name}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    def copy_with_captions(
        self,
        image_paths: List[str],
        captions: Dict[str, str],
        dataset_dir: Path,
        trigger_word: Optional[str] = None
    ) -> int:
        """
        이미지와 캡션을 데이터셋 디렉토리에 복사

        Args:
            image_paths: 이미지 경로 리스트
            captions: {image_path: caption} 딕셔너리
            dataset_dir: 출력 디렉토리
            trigger_word: 트리거 워드 (캡션 앞에 추가)

        Returns:
            복사된 이미지 수
        """
        copied = 0

        for img_path in image_paths:
            try:
                img_path = Path(img_path)

                # 이미지 복사
                dest_img = dataset_dir / img_path.name
                shutil.copy2(img_path, dest_img)

                # 캡션 저장
                caption = captions.get(str(img_path), "")

                # 트리거 워드 추가
                if trigger_word and trigger_word.strip():
                    caption = f"{trigger_word.strip()}, {caption}"

                # .txt 파일로 저장
                caption_file = dest_img.with_suffix('.txt')
                with open(caption_file, 'w', encoding='utf-8') as f:
                    f.write(caption)

                copied += 1

            except Exception as e:
                print(f"  ❌ {img_path.name} 복사 실패: {e}")

        return copied

    def build_dataset(
        self,
        config: DatasetConfig,
        trigger_word: Optional[str] = None
    ) -> Dict[str, any]:
        """
        전체 데이터셋 빌드

        Args:
            config: 데이터셋 설정
            trigger_word: 트리거 워드

        Returns:
            데이터셋 정보
        """
        print(f"📦 데이터셋 빌드 중: {config.name}")

        # 1. 디렉토리 생성
        dataset_dir = self.create_dataset_structure(
            config.name,
            config.repeats
        )

        # 2. 이미지 및 캡션 복사
        copied = self.copy_with_captions(
            config.images,
            config.captions,
            dataset_dir,
            trigger_word
        )

        # 3. 메타데이터 저장
        metadata = {
            "dataset_name": config.name,
            "num_images": copied,
            "repeats": config.repeats,
            "total_steps": copied * config.repeats,
            "trigger_word": trigger_word,
            "flip_aug": config.flip_aug,
            "shuffle_caption": config.shuffle_caption,
            "keep_tokens": config.keep_tokens,
            "dataset_dir": str(dataset_dir)
        }

        metadata_file = dataset_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✅ 데이터셋 준비 완료: {dataset_dir}")
        print(f"   이미지: {copied}장")
        print(f"   반복: {config.repeats}회")
        print(f"   총 스텝: {copied * config.repeats}")

        return metadata

    def calculate_repeats(
        self,
        num_images: int,
        target_steps: int = 1000
    ) -> int:
        """
        이미지 수에 따른 적절한 반복 횟수 계산

        Args:
            num_images: 이미지 수
            target_steps: 목표 스텝 수

        Returns:
            반복 횟수
        """
        if num_images == 0:
            return 1

        repeats = max(1, target_steps // num_images)

        # 이미지 수에 따른 권장값
        if num_images < 10:
            return min(repeats, 20)  # 너무 많은 반복 방지
        elif num_images < 30:
            return min(repeats, 15)
        else:
            return min(repeats, 10)

    def validate_dataset(self, dataset_dir: Path) -> Tuple[bool, str]:
        """
        데이터셋 검증

        Returns:
            (유효 여부, 메시지)
        """
        if not dataset_dir.exists():
            return False, f"디렉토리가 존재하지 않습니다: {dataset_dir}"

        # 이미지와 캡션 파일 확인
        image_files = list(dataset_dir.glob("*.png")) + \
                     list(dataset_dir.glob("*.jpg")) + \
                     list(dataset_dir.glob("*.jpeg"))

        if not image_files:
            return False, "이미지 파일이 없습니다"

        # 각 이미지마다 캡션 파일 있는지 확인
        missing_captions = []
        for img_file in image_files:
            caption_file = img_file.with_suffix('.txt')
            if not caption_file.exists():
                missing_captions.append(img_file.name)

        if missing_captions:
            return False, f"캡션 파일 누락: {missing_captions[:5]}"

        return True, f"✅ 유효한 데이터셋 ({len(image_files)}장)"

    def create_config_file(
        self,
        dataset_dir: Path,
        config_params: dict
    ) -> Path:
        """
        Kohya-ss용 설정 파일 생성 (TOML)

        Args:
            dataset_dir: 데이터셋 디렉토리
            config_params: 학습 파라미터

        Returns:
            설정 파일 경로
        """
        import toml

        config = {
            "general": {
                "enable_bucket": config_params.get("enable_bucket", True),
                "bucket_reso_steps": 64,
                "shuffle_caption": config_params.get("shuffle_caption", True),
                "keep_tokens": config_params.get("keep_tokens", 1),
            },
            "datasets": [
                {
                    "resolution": config_params.get("resolution", 512),
                    "subsets": [
                        {
                            "image_dir": str(dataset_dir),
                            "num_repeats": config_params.get("repeats", 10),
                            "flip_aug": config_params.get("flip_aug", False),
                        }
                    ]
                }
            ]
        }

        config_file = dataset_dir / "dataset_config.toml"
        with open(config_file, 'w', encoding='utf-8') as f:
            toml.dump(config, f)

        return config_file


# 테스트용
if __name__ == "__main__":
    from typing import Tuple

    builder = DatasetBuilder()

    # 반복 횟수 계산 테스트
    print("=== 반복 횟수 권장 ===")
    for num in [5, 10, 20, 50, 100]:
        repeats = builder.calculate_repeats(num)
        print(f"{num}장 이미지 → {repeats}회 반복 (총 {num * repeats} 스텝)")
