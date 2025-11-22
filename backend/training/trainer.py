"""
메인 LoRA 트레이너
전체 파이프라인 조율
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable
from datetime import datetime

from backend.preprocessing.image_processor import ImageProcessor
from backend.preprocessing.captioner import AutoCaptioner
from backend.preprocessing.dataset_builder import DatasetBuilder, DatasetConfig
from backend.training.kohya_wrapper import KohyaTrainer, KohyaConfig
from backend.utils.config_loader import ConfigLoader
from backend.utils.gpu_utils import get_gpu_info, check_vram_requirements


class LoRATrainer:
    """LoRA 학습 메인 클래스"""

    def __init__(
        self,
        output_root: str = "outputs",
        cache_root: str = "cache",
        config_path: str = "configs/training_presets.yaml"
    ):
        """
        Args:
            output_root: 생성된 LoRA 저장 디렉토리
            cache_root: 캐시 디렉토리 (전처리된 이미지, 데이터셋 등)
            config_path: 프리셋 설정 파일 경로
        """
        self.output_root = Path(output_root)
        self.cache_root = Path(cache_root)
        self.config_loader = ConfigLoader(config_path)

        self.output_root.mkdir(exist_ok=True)
        self.cache_root.mkdir(exist_ok=True)

    def train_lora(
        self,
        image_paths: List[str],
        lora_name: str,
        base_model_path: str,
        preset: str = "character",
        trigger_word: Optional[str] = None,
        caption_method: str = "blip",
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, any]:
        """
        LoRA 학습 전체 파이프라인

        Args:
            image_paths: 학습 이미지 경로 리스트
            lora_name: 생성할 LoRA 이름
            base_model_path: 베이스 모델 경로
            preset: 학습 프리셋 (test, character, style, concept 등)
            trigger_word: 트리거 워드 (선택)
            caption_method: 캡셔닝 방법 (blip, wd14, both)
            progress_callback: 진행 상황 콜백 (message, progress)

        Returns:
            학습 결과 딕셔너리
        """
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        def report(msg: str, progress: float = 0.0):
            """진행 상황 보고"""
            print(msg)
            if progress_callback:
                progress_callback(msg, progress)

        try:
            # === 0. 환경 체크 ===
            report("🔍 환경 체크 중...", 0.0)

            gpu_info = get_gpu_info()
            if not gpu_info["available"]:
                return {
                    "status": "error",
                    "error": "CUDA GPU를 사용할 수 없습니다. NVIDIA GPU가 필요합니다."
                }

            vram_ok, vram_msg = check_vram_requirements(12.0)
            report(vram_msg, 0.05)

            # === 1. 프리셋 로드 ===
            report(f"⚙️ 프리셋 로드 중: {preset}", 0.1)

            try:
                preset_config = self.config_loader.get_preset(preset)
            except KeyError as e:
                return {
                    "status": "error",
                    "error": str(e)
                }

            # === 2. 이미지 전처리 ===
            report(f"🖼️ 이미지 전처리 중 ({len(image_paths)}장)...", 0.15)

            processor = ImageProcessor(
                target_size=preset_config.get("resolution", 512),
                enable_bucket=preset_config.get("enable_bucket", True)
            )

            # 중복 제거
            image_paths = processor.remove_duplicates(image_paths)

            # 전처리
            processed_dir = self.cache_root / f"{lora_name}_{timestamp}" / "processed"
            processed_results = processor.process_batch(
                image_paths,
                str(processed_dir),
                show_progress=False
            )

            if not processed_results:
                return {
                    "status": "error",
                    "error": "전처리된 이미지가 없습니다. 이미지 품질을 확인해주세요."
                }

            processed_paths = [r["processed_path"] for r in processed_results]
            report(f"✅ {len(processed_paths)}장 전처리 완료", 0.25)

            # === 3. 자동 캡셔닝 ===
            report(f"💬 자동 캡셔닝 중 ({caption_method})...", 0.30)

            captioner = AutoCaptioner(method=caption_method, device="cuda")
            captions = captioner.generate_captions_batch(
                processed_paths,
                show_progress=False
            )
            captioner.cleanup()

            report(f"✅ {len(captions)}개 캡션 생성 완료", 0.45)

            # === 4. 데이터셋 빌드 ===
            report("📦 데이터셋 빌드 중...", 0.50)

            builder = DatasetBuilder(str(self.cache_root / f"{lora_name}_{timestamp}"))

            # 반복 횟수 자동 계산
            repeats = builder.calculate_repeats(
                len(processed_paths),
                target_steps=preset_config.get("max_train_epochs", 10) * 100
            )

            dataset_config = DatasetConfig(
                name=lora_name,
                images=processed_paths,
                captions=captions,
                repeats=repeats,
                shuffle_caption=preset_config.get("shuffle_caption", True),
                keep_tokens=preset_config.get("keep_tokens", 1)
            )

            dataset_metadata = builder.build_dataset(dataset_config, trigger_word)
            dataset_dir = dataset_metadata["dataset_dir"]

            report(f"✅ 데이터셋 준비 완료: {dataset_dir}", 0.60)

            # === 5. Kohya 학습 설정 ===
            report("⚙️ 학습 설정 중...", 0.65)

            output_dir = self.output_root / f"{lora_name}_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)

            logging_dir = output_dir / "logs"

            kohya_config = KohyaConfig(
                # 필수
                pretrained_model_name_or_path=base_model_path,
                train_data_dir=dataset_dir,
                output_dir=str(output_dir),
                output_name=lora_name,

                # 네트워크
                network_dim=preset_config.get("network_dim", 32),
                network_alpha=preset_config.get("network_alpha", 16),

                # 학습률
                learning_rate=preset_config.get("learning_rate", 1e-4),
                unet_lr=preset_config.get("unet_lr"),
                text_encoder_lr=preset_config.get("text_encoder_lr"),
                lr_scheduler=preset_config.get("lr_scheduler", "cosine"),
                lr_warmup_steps=preset_config.get("lr_warmup_steps", 0),
                lr_warmup_ratio=preset_config.get("lr_warmup_ratio", 0.0),

                # 에폭
                max_train_epochs=preset_config.get("max_train_epochs", 10),
                max_train_steps=preset_config.get("max_train_steps"),
                save_every_n_epochs=preset_config.get("save_every_n_epochs", 2),

                # 배치
                train_batch_size=preset_config.get("train_batch_size", 1),
                gradient_accumulation_steps=preset_config.get("gradient_accumulation_steps", 1),
                max_grad_norm=preset_config.get("max_grad_norm", 1.0),

                # 해상도
                resolution=preset_config.get("resolution", 512),
                enable_bucket=preset_config.get("enable_bucket", True),
                min_bucket_reso=preset_config.get("min_bucket_reso", 256),
                max_bucket_reso=preset_config.get("max_bucket_reso", 1024),
                bucket_reso_steps=preset_config.get("bucket_reso_steps", 64),

                # 최적화
                optimizer_type=preset_config.get("optimizer_type", "AdamW8bit"),
                mixed_precision=preset_config.get("mixed_precision", "fp16"),
                save_precision=preset_config.get("save_precision", "fp16"),
                gradient_checkpointing=preset_config.get("gradient_checkpointing", True),
                xformers=preset_config.get("xformers", True),
                cache_latents=preset_config.get("cache_latents", True),
                cache_latents_to_disk=preset_config.get("cache_latents_to_disk", True),

                # Regularization
                min_snr_gamma=preset_config.get("min_snr_gamma"),
                noise_offset=preset_config.get("noise_offset"),

                # 샘플링
                sample_every_n_epochs=preset_config.get("sample_every_n_epochs"),
                sample_sampler=preset_config.get("sample_sampler", "euler_a"),
                sample_prompts=preset_config.get("sample_prompts"),

                # 기타
                seed=preset_config.get("seed", 42),
                clip_skip=preset_config.get("clip_skip", 2),
                logging_dir=str(logging_dir),
                log_with=preset_config.get("log_with", "tensorboard"),
                save_model_as=preset_config.get("save_model_as", "safetensors"),
                shuffle_caption=preset_config.get("shuffle_caption", True),
                keep_tokens=preset_config.get("keep_tokens", 1),
            )

            # === 6. LoRA 학습 실행 ===
            report("🚀 LoRA 학습 시작!", 0.70)

            trainer = KohyaTrainer()

            def training_progress(line: str):
                """학습 진행 상황"""
                # epoch, loss 등 파싱하여 진행률 업데이트 가능
                if progress_callback:
                    # 70% ~ 95% 사이에서 업데이트
                    progress_callback(line, 0.70)

            training_result = trainer.train(
                kohya_config,
                progress_callback=training_progress,
                dry_run=False
            )

            # === 7. 결과 처리 ===
            if training_result["status"] == "success":
                elapsed_time = time.time() - start_time
                report(f"✅ LoRA 학습 완료! ({elapsed_time:.1f}초)", 1.0)

                return {
                    "status": "success",
                    "lora_name": lora_name,
                    "output_files": training_result.get("output_files", []),
                    "output_dir": str(output_dir),
                    "dataset_dir": dataset_dir,
                    "num_images": len(processed_paths),
                    "preset": preset,
                    "trigger_word": trigger_word,
                    "elapsed_time": elapsed_time,
                    "config": kohya_config
                }
            else:
                report(f"❌ 학습 실패: {training_result.get('error')}", 0.95)
                return training_result

        except Exception as e:
            error_msg = f"학습 중 오류 발생: {str(e)}"
            report(f"❌ {error_msg}", 0.0)
            return {
                "status": "error",
                "error": error_msg
            }


# 사용 예시
if __name__ == "__main__":
    trainer = LoRATrainer()

    # 테스트 (실제 경로로 변경 필요)
    result = trainer.train_lora(
        image_paths=["test1.png", "test2.png"],
        lora_name="test_lora",
        base_model_path="models/sd-v1-5.safetensors",
        preset="test",
        trigger_word="mychar"
    )

    print("\n=== 결과 ===")
    for key, value in result.items():
        print(f"{key}: {value}")
