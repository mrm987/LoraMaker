"""
Kohya-ss 학습 Wrapper
검증된 Kohya-ss 스크립트를 Python에서 실행
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Dict, Optional, Callable
from dataclasses import dataclass, asdict


@dataclass
class KohyaConfig:
    """Kohya-ss 학습 설정"""

    # 필수 파라미터
    pretrained_model_name_or_path: str
    train_data_dir: str
    output_dir: str
    output_name: str

    # 네트워크 설정
    network_module: str = "networks.lora"
    network_dim: int = 32
    network_alpha: int = 16

    # 학습 파라미터
    learning_rate: float = 1e-4
    unet_lr: Optional[float] = None
    text_encoder_lr: Optional[float] = None
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 0
    lr_warmup_ratio: float = 0.0

    # 에폭 및 스텝
    max_train_epochs: int = 10
    max_train_steps: Optional[int] = None
    save_every_n_epochs: int = 2
    save_every_n_steps: Optional[int] = None

    # 배치 설정
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # 해상도
    resolution: int = 512
    enable_bucket: bool = True
    min_bucket_reso: int = 256
    max_bucket_reso: int = 1024
    bucket_reso_steps: int = 64

    # 최적화
    optimizer_type: str = "AdamW8bit"
    mixed_precision: str = "fp16"
    save_precision: str = "fp16"
    gradient_checkpointing: bool = True
    xformers: bool = True
    cache_latents: bool = True
    cache_latents_to_disk: bool = True

    # Regularization
    min_snr_gamma: Optional[float] = None
    noise_offset: Optional[float] = None

    # 샘플링
    sample_every_n_epochs: Optional[int] = None
    sample_every_n_steps: Optional[int] = None
    sample_sampler: str = "euler_a"
    sample_prompts: Optional[str] = None

    # 기타
    seed: int = 42
    clip_skip: int = 2
    logging_dir: Optional[str] = None
    log_with: str = "tensorboard"
    save_model_as: str = "safetensors"

    # 데이터셋
    shuffle_caption: bool = True
    keep_tokens: int = 1

    def to_args(self) -> list:
        """딕셔너리를 CLI 인자로 변환"""
        args = []
        config_dict = asdict(self)

        for key, value in config_dict.items():
            if value is None:
                continue

            # 언더스코어를 하이픈으로
            arg_name = f"--{key.replace('_', '-')}"

            if isinstance(value, bool):
                if value:
                    args.append(arg_name)
            elif isinstance(value, (int, float, str)):
                args.extend([arg_name, str(value)])

        return args


class KohyaTrainer:
    """Kohya-ss 학습 실행기"""

    def __init__(
        self,
        sd_scripts_path: Optional[str] = None,
        venv_python: Optional[str] = None
    ):
        """
        Args:
            sd_scripts_path: Kohya-ss sd-scripts 경로
            venv_python: 가상환경 Python 경로 (None이면 현재 환경)
        """
        # sd-scripts 경로 찾기
        if sd_scripts_path is None:
            # 기본 경로들 시도
            candidates = [
                "external/sd-scripts",
                "../sd-scripts",
                "sd-scripts"
            ]
            for path in candidates:
                if Path(path).exists():
                    sd_scripts_path = path
                    break

        self.sd_scripts_path = Path(sd_scripts_path) if sd_scripts_path else None
        self.venv_python = venv_python or "python"

        if self.sd_scripts_path and not self.sd_scripts_path.exists():
            print(f"⚠️ Kohya-ss 경로를 찾을 수 없습니다: {self.sd_scripts_path}")
            print("   Kohya-ss를 설치해주세요:")
            print("   git clone https://github.com/kohya-ss/sd-scripts.git external/sd-scripts")

    def get_train_script(self) -> Path:
        """학습 스크립트 경로"""
        if self.sd_scripts_path is None:
            raise RuntimeError("sd-scripts 경로가 설정되지 않았습니다")

        script = self.sd_scripts_path / "train_network.py"
        if not script.exists():
            raise FileNotFoundError(f"학습 스크립트를 찾을 수 없습니다: {script}")

        return script

    def build_command(self, config: KohyaConfig) -> list:
        """학습 명령어 빌드"""
        train_script = self.get_train_script()

        # Python + 스크립트 + 인자
        cmd = [self.venv_python, str(train_script)] + config.to_args()

        return cmd

    def train(
        self,
        config: KohyaConfig,
        progress_callback: Optional[Callable[[str], None]] = None,
        dry_run: bool = False
    ) -> Dict[str, any]:
        """
        LoRA 학습 실행

        Args:
            config: 학습 설정
            progress_callback: 진행 상황 콜백 함수
            dry_run: True면 명령어만 출력하고 실행 안함

        Returns:
            학습 결과 정보
        """
        # 출력 디렉토리 생성
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        if config.logging_dir:
            Path(config.logging_dir).mkdir(parents=True, exist_ok=True)

        # 명령어 빌드
        cmd = self.build_command(config)

        print("=" * 60)
        print("🚀 LoRA 학습 시작")
        print("=" * 60)
        print(f"모델: {config.pretrained_model_name_or_path}")
        print(f"데이터: {config.train_data_dir}")
        print(f"출력: {config.output_dir}/{config.output_name}")
        print(f"Epochs: {config.max_train_epochs}")
        print(f"Dim: {config.network_dim}, Alpha: {config.network_alpha}")
        print(f"LR: {config.learning_rate}")
        print("=" * 60)

        if dry_run:
            print("\n[DRY RUN] 실행 명령어:")
            print(" ".join(cmd))
            return {"status": "dry_run", "command": cmd}

        # 학습 실행
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # 출력 스트리밍
            for line in process.stdout:
                line = line.strip()
                if line:
                    print(line)
                    if progress_callback:
                        progress_callback(line)

            # 완료 대기
            return_code = process.wait()

            if return_code == 0:
                print("\n✅ 학습 완료!")

                # 생성된 파일 찾기
                output_files = list(Path(config.output_dir).glob(f"{config.output_name}*.safetensors"))

                return {
                    "status": "success",
                    "return_code": return_code,
                    "output_files": [str(f) for f in output_files],
                    "output_dir": config.output_dir
                }
            else:
                print(f"\n❌ 학습 실패 (exit code: {return_code})")
                return {
                    "status": "failed",
                    "return_code": return_code,
                    "error": f"프로세스가 코드 {return_code}로 종료되었습니다"
                }

        except FileNotFoundError:
            error_msg = f"Python 인터프리터를 찾을 수 없습니다: {self.venv_python}"
            print(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg
            }

        except Exception as e:
            error_msg = f"학습 중 오류 발생: {e}"
            print(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg
            }

    def validate_environment(self) -> Dict[str, any]:
        """환경 검증"""
        results = {}

        # 1. sd-scripts 존재 확인
        if self.sd_scripts_path and self.sd_scripts_path.exists():
            results["sd_scripts"] = True
            results["sd_scripts_path"] = str(self.sd_scripts_path)
        else:
            results["sd_scripts"] = False
            results["error"] = "sd-scripts를 찾을 수 없습니다"
            return results

        # 2. 학습 스크립트 존재 확인
        try:
            train_script = self.get_train_script()
            results["train_script"] = True
            results["train_script_path"] = str(train_script)
        except Exception as e:
            results["train_script"] = False
            results["error"] = str(e)
            return results

        # 3. Python 실행 가능 확인
        try:
            result = subprocess.run(
                [self.venv_python, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                results["python"] = True
                results["python_version"] = result.stdout.strip()
            else:
                results["python"] = False
        except Exception as e:
            results["python"] = False
            results["python_error"] = str(e)

        results["status"] = "ok" if all([
            results.get("sd_scripts"),
            results.get("train_script"),
            results.get("python")
        ]) else "error"

        return results


# 테스트용
if __name__ == "__main__":
    trainer = KohyaTrainer()

    print("=== Kohya Trainer 환경 검증 ===")
    validation = trainer.validate_environment()

    for key, value in validation.items():
        print(f"{key}: {value}")

    if validation["status"] == "ok":
        print("\n✅ 환경 준비 완료!")
    else:
        print("\n❌ 환경 설정 필요")
