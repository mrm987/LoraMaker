"""
자동 캡셔닝
BLIP-2, WD14 Tagger를 사용한 이미지 캡션 생성
"""

import torch
from PIL import Image
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm


class AutoCaptioner:
    """자동 캡션 생성기"""

    def __init__(
        self,
        method: str = "blip",
        device: str = "auto"
    ):
        """
        Args:
            method: 캡셔닝 방법 ("blip", "wd14", "both")
            device: 디바이스 ("cuda", "cpu", "auto")
        """
        self.method = method.lower()

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.blip_model = None
        self.blip_processor = None
        self.wd14_model = None
        self.wd14_tags = None

        print(f"💬 AutoCaptioner 초기화 (method={method}, device={self.device})")

    def load_blip(self):
        """BLIP-2 모델 로드"""
        if self.blip_model is not None:
            return

        print("📥 BLIP-2 모델 로드 중...")
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration

            # BLIP-2 OPT 2.7B 모델 (가벼운 버전)
            model_name = "Salesforce/blip2-opt-2.7b"

            self.blip_processor = Blip2Processor.from_pretrained(model_name)
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.blip_model.to(self.device)
            self.blip_model.eval()

            print("✅ BLIP-2 모델 로드 완료")

        except Exception as e:
            print(f"❌ BLIP-2 로드 실패: {e}")
            print("   대안: transformers 라이브러리를 설치하세요")
            print("   pip install transformers")

    def load_wd14(self):
        """WD14 Tagger 모델 로드"""
        if self.wd14_model is not None:
            return

        print("📥 WD14 Tagger 모델 로드 중...")
        try:
            import onnxruntime as ort
            import numpy as np
            from huggingface_hub import hf_hub_download
            import pandas as pd

            # WD14 모델 다운로드 (ONNX 버전)
            model_repo = "SmilingWolf/wd-v1-4-moat-tagger-v2"

            model_path = hf_hub_download(
                repo_id=model_repo,
                filename="model.onnx"
            )

            tags_path = hf_hub_download(
                repo_id=model_repo,
                filename="selected_tags.csv"
            )

            # ONNX 모델 로드
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if self.device == "cuda" else ['CPUExecutionProvider']

            self.wd14_model = ort.InferenceSession(model_path, providers=providers)

            # 태그 리스트 로드
            tags_df = pd.read_csv(tags_path)
            self.wd14_tags = tags_df['name'].tolist()

            print("✅ WD14 Tagger 로드 완료")

        except Exception as e:
            print(f"❌ WD14 로드 실패: {e}")
            print("   대안: onnxruntime과 huggingface-hub를 설치하세요")
            print("   pip install onnxruntime huggingface-hub pandas")

    def caption_with_blip(
        self,
        image: Image.Image,
        prompt: str = "a photo of"
    ) -> str:
        """
        BLIP-2로 캡션 생성

        Args:
            image: PIL Image
            prompt: 프롬프트 (선택)

        Returns:
            생성된 캡션
        """
        if self.blip_model is None:
            self.load_blip()

        if self.blip_model is None:
            return "a photo"  # 폴백

        try:
            inputs = self.blip_processor(image, text=prompt, return_tensors="pt").to(
                self.device, torch.float16 if self.device == "cuda" else torch.float32
            )

            with torch.no_grad():
                generated_ids = self.blip_model.generate(**inputs, max_length=50)

            caption = self.blip_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            return caption

        except Exception as e:
            print(f"  ⚠️ BLIP 캡셔닝 실패: {e}")
            return "a photo"

    def caption_with_wd14(
        self,
        image: Image.Image,
        threshold: float = 0.35,
        max_tags: int = 20
    ) -> str:
        """
        WD14 Tagger로 태그 생성

        Args:
            image: PIL Image
            threshold: 태그 확신도 임계값
            max_tags: 최대 태그 수

        Returns:
            콤마로 구분된 태그 문자열
        """
        if self.wd14_model is None:
            self.load_wd14()

        if self.wd14_model is None:
            return ""  # 폴백

        try:
            import numpy as np

            # 이미지 전처리 (448x448)
            image_rgb = image.convert('RGB')
            image_resized = image_rgb.resize((448, 448), Image.Resampling.LANCZOS)
            image_array = np.array(image_resized).astype(np.float32) / 255.0
            image_array = np.expand_dims(image_array, 0)  # 배치 차원 추가

            # 추론
            input_name = self.wd14_model.get_inputs()[0].name
            output = self.wd14_model.run(None, {input_name: image_array})[0][0]

            # 임계값 이상 태그 선택
            tag_indices = np.where(output > threshold)[0]
            tag_scores = output[tag_indices]

            # 스코어 순 정렬
            sorted_indices = np.argsort(tag_scores)[::-1][:max_tags]
            selected_tags = [self.wd14_tags[tag_indices[i]] for i in sorted_indices]

            # 태그를 언더스코어에서 스페이스로 변환
            selected_tags = [tag.replace('_', ' ') for tag in selected_tags]

            return ", ".join(selected_tags)

        except Exception as e:
            print(f"  ⚠️ WD14 태깅 실패: {e}")
            return ""

    def generate_caption(
        self,
        image_path: str,
        blip_prompt: str = "a photo of",
        wd14_threshold: float = 0.35
    ) -> str:
        """
        이미지 캡션 생성 (설정된 method에 따라)

        Args:
            image_path: 이미지 경로
            blip_prompt: BLIP 프롬프트
            wd14_threshold: WD14 임계값

        Returns:
            생성된 캡션
        """
        try:
            image = Image.open(image_path).convert('RGB')

            if self.method == "blip":
                return self.caption_with_blip(image, blip_prompt)

            elif self.method == "wd14":
                return self.caption_with_wd14(image, wd14_threshold)

            elif self.method == "both":
                # BLIP + WD14 조합
                blip_caption = self.caption_with_blip(image, blip_prompt)
                wd14_tags = self.caption_with_wd14(image, wd14_threshold)

                if wd14_tags:
                    return f"{blip_caption}, {wd14_tags}"
                else:
                    return blip_caption

            else:
                return "a photo"

        except Exception as e:
            print(f"  ❌ 캡션 생성 실패 ({Path(image_path).name}): {e}")
            return "a photo"

    def generate_captions_batch(
        self,
        image_paths: List[str],
        blip_prompt: str = "a photo of",
        wd14_threshold: float = 0.35,
        show_progress: bool = True
    ) -> Dict[str, str]:
        """
        여러 이미지 캡션 배치 생성

        Args:
            image_paths: 이미지 경로 리스트
            blip_prompt: BLIP 프롬프트
            wd14_threshold: WD14 임계값
            show_progress: 진행률 표시

        Returns:
            {image_path: caption} 딕셔너리
        """
        captions = {}
        iterator = tqdm(image_paths, desc="캡션 생성") if show_progress else image_paths

        for img_path in iterator:
            caption = self.generate_caption(img_path, blip_prompt, wd14_threshold)
            captions[img_path] = caption

        print(f"✅ {len(captions)}개 캡션 생성 완료")
        return captions

    def cleanup(self):
        """메모리 정리"""
        if self.blip_model is not None:
            del self.blip_model
            del self.blip_processor
            self.blip_model = None
            self.blip_processor = None

        if self.wd14_model is not None:
            del self.wd14_model
            del self.wd14_tags
            self.wd14_model = None
            self.wd14_tags = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("🧹 메모리 정리 완료")


# 테스트용
if __name__ == "__main__":
    print("=== AutoCaptioner 테스트 ===")
    print("실제 이미지 경로를 지정하여 테스트하세요")
    print("\n사용 예시:")
    print("""
    captioner = AutoCaptioner(method="blip", device="cuda")
    caption = captioner.generate_caption("test.png")
    print(caption)
    """)
