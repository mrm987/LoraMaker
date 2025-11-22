"""
이미지 전처리
리사이징, 크롭, 품질 필터링 등
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm


class ImageProcessor:
    """이미지 전처리 및 검증"""

    def __init__(
        self,
        target_size: int = 512,
        min_size: int = 256,
        max_aspect_ratio: float = 2.0,
        enable_bucket: bool = True
    ):
        """
        Args:
            target_size: 목표 해상도 (512, 768, 1024 등)
            min_size: 최소 이미지 크기
            max_aspect_ratio: 최대 가로세로 비율
            enable_bucket: 버킷 해상도 사용 여부
        """
        self.target_size = target_size
        self.min_size = min_size
        self.max_aspect_ratio = max_aspect_ratio
        self.enable_bucket = enable_bucket

        # 버킷 해상도 (Kohya-ss 방식)
        if enable_bucket:
            self.bucket_resos = self._generate_bucket_resolutions()

    def _generate_bucket_resolutions(
        self,
        min_reso: int = 320,
        max_reso: int = 960,
        step: int = 64
    ) -> List[Tuple[int, int]]:
        """
        버킷 해상도 생성
        총 픽셀 수가 target_size^2와 비슷하도록
        """
        buckets = []
        target_pixels = self.target_size * self.target_size

        for h in range(min_reso, max_reso + 1, step):
            w = int((target_pixels / h) // step) * step
            if min_reso <= w <= max_reso:
                buckets.append((h, w))
                if h != w:
                    buckets.append((w, h))  # 회전 버전도 추가

        return sorted(set(buckets))

    def find_best_bucket(self, width: int, height: int) -> Tuple[int, int]:
        """
        이미지에 가장 적합한 버킷 해상도 찾기

        Args:
            width, height: 원본 이미지 크기

        Returns:
            (bucket_h, bucket_w)
        """
        if not self.enable_bucket:
            return (self.target_size, self.target_size)

        aspect = width / height
        best_bucket = (self.target_size, self.target_size)
        min_diff = float('inf')

        for bh, bw in self.bucket_resos:
            bucket_aspect = bw / bh
            diff = abs(aspect - bucket_aspect)

            if diff < min_diff:
                min_diff = diff
                best_bucket = (bh, bw)

        return best_bucket

    def resize_and_crop(
        self,
        image: Image.Image,
        target_height: int,
        target_width: int,
        crop_mode: str = "center"
    ) -> Image.Image:
        """
        이미지 리사이징 및 크롭

        Args:
            image: PIL Image
            target_height, target_width: 목표 크기
            crop_mode: center, random, face (향후 얼굴 인식 크롭 지원)

        Returns:
            처리된 PIL Image
        """
        w, h = image.size
        aspect = w / h
        target_aspect = target_width / target_height

        # 비율 맞춰 리사이즈 (크롭 여유 확보)
        if aspect > target_aspect:
            # 가로가 더 넓음 → 세로 기준 리사이징
            new_h = target_height
            new_w = int(target_height * aspect)
        else:
            # 세로가 더 김 → 가로 기준 리사이징
            new_w = target_width
            new_h = int(target_width / aspect)

        # 고품질 리샘플링
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 크롭
        if crop_mode == "center":
            left = (new_w - target_width) // 2
            top = (new_h - target_height) // 2
        elif crop_mode == "random":
            left = np.random.randint(0, max(1, new_w - target_width + 1))
            top = np.random.randint(0, max(1, new_h - target_height + 1))
        else:
            left = (new_w - target_width) // 2
            top = (new_h - target_height) // 2

        right = left + target_width
        bottom = top + target_height

        return image.crop((left, top, right, bottom))

    def check_image_quality(self, image: Image.Image) -> Tuple[bool, str]:
        """
        이미지 품질 검사

        Returns:
            (통과 여부, 메시지)
        """
        w, h = image.size

        # 1. 크기 체크
        if w < self.min_size or h < self.min_size:
            return False, f"이미지가 너무 작습니다 ({w}x{h}). 최소 {self.min_size}px 필요"

        # 2. 가로세로 비율 체크
        aspect = max(w, h) / min(w, h)
        if aspect > self.max_aspect_ratio:
            return False, f"가로세로 비율이 너무 극단적입니다 ({aspect:.1f}:1)"

        # 3. 흐릿함 검사 (Laplacian variance)
        try:
            img_array = np.array(image.convert('L'))  # 그레이스케일
            laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
            variance = laplacian.var()

            # 분산이 낮으면 흐릿한 이미지
            if variance < 100:  # 임계값 (조정 가능)
                return False, f"이미지가 너무 흐릿합니다 (sharpness: {variance:.1f})"
        except Exception as e:
            # OpenCV 에러 시 패스
            pass

        return True, "OK"

    def process_image(
        self,
        image_path: str,
        output_dir: str,
        crop_mode: str = "center"
    ) -> Optional[Tuple[str, Tuple[int, int]]]:
        """
        단일 이미지 처리

        Args:
            image_path: 입력 이미지 경로
            output_dir: 출력 디렉토리
            crop_mode: 크롭 모드

        Returns:
            (출력 경로, (h, w)) or None (실패 시)
        """
        try:
            # 이미지 로드
            image = Image.open(image_path)

            # RGB로 변환 (RGBA, L 등 처리)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # 품질 검사
            ok, msg = self.check_image_quality(image)
            if not ok:
                print(f"  ⚠️ {Path(image_path).name}: {msg}")
                return None

            # 버킷 해상도 찾기
            w, h = image.size
            target_h, target_w = self.find_best_bucket(w, h)

            # 리사이징 및 크롭
            processed = self.resize_and_crop(image, target_h, target_w, crop_mode)

            # 저장
            output_path = Path(output_dir) / Path(image_path).name
            processed.save(output_path, quality=95, optimize=True)

            return str(output_path), (target_h, target_w)

        except Exception as e:
            print(f"  ❌ {Path(image_path).name}: 처리 실패 - {e}")
            return None

    def process_batch(
        self,
        image_paths: List[str],
        output_dir: str,
        crop_mode: str = "center",
        show_progress: bool = True
    ) -> List[dict]:
        """
        여러 이미지 배치 처리

        Args:
            image_paths: 이미지 경로 리스트
            output_dir: 출력 디렉토리
            crop_mode: 크롭 모드
            show_progress: 진행률 표시

        Returns:
            처리된 이미지 정보 리스트
            [{"path": "...", "resolution": (h, w)}, ...]
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        results = []
        iterator = tqdm(image_paths, desc="이미지 처리") if show_progress else image_paths

        for img_path in iterator:
            result = self.process_image(img_path, output_dir, crop_mode)
            if result:
                output_path, resolution = result
                results.append({
                    "original_path": img_path,
                    "processed_path": output_path,
                    "resolution": resolution
                })

        print(f"\n✅ {len(results)}/{len(image_paths)} 이미지 처리 완료")
        return results

    def remove_duplicates(self, image_paths: List[str]) -> List[str]:
        """
        중복 이미지 제거 (해시 기반)

        Args:
            image_paths: 이미지 경로 리스트

        Returns:
            중복 제거된 경로 리스트
        """
        import hashlib

        hashes = {}
        unique_paths = []

        for path in image_paths:
            try:
                # 이미지 해시 계산
                with open(path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash not in hashes:
                    hashes[file_hash] = path
                    unique_paths.append(path)
                else:
                    print(f"  ⚠️ 중복 이미지 제거: {Path(path).name}")
            except Exception as e:
                print(f"  ❌ 해시 계산 실패 ({path}): {e}")

        removed = len(image_paths) - len(unique_paths)
        if removed > 0:
            print(f"✅ {removed}개 중복 이미지 제거됨")

        return unique_paths


# 테스트용
if __name__ == "__main__":
    processor = ImageProcessor(target_size=512, enable_bucket=True)

    print("=== 버킷 해상도 ===")
    for i, (h, w) in enumerate(processor.bucket_resos[:10]):
        print(f"{i+1}. {h}x{w} ({h*w} pixels)")

    print("\n=== 버킷 매칭 테스트 ===")
    test_sizes = [(800, 600), (1920, 1080), (512, 512), (400, 800)]
    for w, h in test_sizes:
        bh, bw = processor.find_best_bucket(w, h)
        print(f"{w}x{h} → {bw}x{bh}")
