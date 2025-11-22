"""
LoraMaker - ComfyUI LoRA 자동 생성 도구
간단한 UI로 LoRA 학습 자동화
"""

import gradio as gr
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.training.trainer import LoRATrainer
from backend.utils.gpu_utils import get_gpu_info


class LoraMakerUI:
    def __init__(self):
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        self.trainer = LoRATrainer()

        # 베이스 모델 경로 매핑
        self.model_paths = {
            "SD 1.5 (기본)": "runwayml/stable-diffusion-v1-5",
            "SD 2.1": "stabilityai/stable-diffusion-2-1",
            "SDXL 1.0": "stabilityai/stable-diffusion-xl-base-1.0",
        }

    def train_lora(
        self,
        images,
        base_model,
        lora_name,
        training_mode,
        trigger_word,
        caption_method,
        progress=gr.Progress()
    ):
        """
        LoRA 학습 메인 함수

        Args:
            images: 업로드된 이미지 리스트
            base_model: 베이스 모델 이름
            lora_name: 생성할 LoRA 이름
            training_mode: 학습 모드
            trigger_word: 트리거 워드 (선택)
            caption_method: 캡셔닝 방법
            progress: Gradio Progress tracker
        """
        # 입력 검증
        if not images:
            return "❌ 이미지를 업로드해주세요!", None

        if not lora_name:
            return "❌ LoRA 이름을 입력해주세요!", None

        # 이미지 경로 리스트로 변환
        if isinstance(images, list):
            image_paths = images
        else:
            image_paths = [images]

        # 모드에서 프리셋 이름 추출
        mode_map = {
            "🚀 테스트 모드 (2분 - 개발용)": "test",
            "👤 캐릭터 (20-30분)": "character",
            "🎨 스타일/화풍 (30-60분)": "style",
            "📦 오브젝트/컨셉 (20-40분)": "concept",
        }
        preset = mode_map.get(training_mode, "character")

        # 베이스 모델 경로
        model_path = self.model_paths.get(base_model, self.model_paths["SD 1.5 (기본)"])

        # 캡셔닝 방법 매핑
        caption_map = {
            "BLIP (자연어)": "blip",
            "WD14 (태그)": "wd14",
            "BLIP + WD14 (조합)": "both"
        }
        caption_method_key = caption_map.get(caption_method, "blip")

        # 진행 상황 메시지들
        status_log = []

        def update_progress(msg: str, prog: float):
            """진행 상황 업데이트"""
            status_log.append(msg)
            progress(prog, desc=msg)

        try:
            # GPU 체크
            gpu_info = get_gpu_info()
            if not gpu_info["available"]:
                return "❌ CUDA GPU를 사용할 수 없습니다. NVIDIA GPU가 필요합니다.", None

            status_log.append(f"✅ GPU: {gpu_info['name']} ({gpu_info['total_vram_gb']}GB)")
            status_log.append(f"📊 이미지: {len(image_paths)}장")
            status_log.append(f"⚙️ 모드: {preset}")
            status_log.append(f"🎯 모델: {model_path}")
            status_log.append("=" * 50)

            # 학습 시작
            result = self.trainer.train_lora(
                image_paths=image_paths,
                lora_name=lora_name,
                base_model_path=model_path,
                preset=preset,
                trigger_word=trigger_word if trigger_word.strip() else None,
                caption_method=caption_method_key,
                progress_callback=update_progress
            )

            # 결과 처리
            if result["status"] == "success":
                status_log.append("")
                status_log.append("=" * 50)
                status_log.append("✅ LoRA 생성 완료!")
                status_log.append(f"📁 출력 디렉토리: {result['output_dir']}")
                status_log.append(f"⏱️ 소요 시간: {result['elapsed_time']:.1f}초")
                status_log.append("")
                status_log.append("💾 생성된 파일:")
                for f in result.get("output_files", []):
                    status_log.append(f"  - {Path(f).name}")

                # 최종 LoRA 파일 경로
                output_files = result.get("output_files", [])
                output_file = output_files[0] if output_files else None

                return "\n".join(status_log), output_file

            else:
                # 에러
                status_log.append("")
                status_log.append("❌ 학습 실패!")
                status_log.append(f"오류: {result.get('error', '알 수 없는 오류')}")
                return "\n".join(status_log), None

        except Exception as e:
            status_log.append("")
            status_log.append(f"❌ 예외 발생: {str(e)}")
            import traceback
            status_log.append(traceback.format_exc())
            return "\n".join(status_log), None

    def create_ui(self):
        """Gradio UI 생성"""

        with gr.Blocks(title="LoraMaker - LoRA 자동 생성", theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
                # 🎨 LoraMaker
                ### ComfyUI용 LoRA를 자동으로 생성하는 도구

                이미지만 업로드하면 자동으로 LoRA를 학습합니다!
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📁 1. 이미지 업로드")
                    images = gr.File(
                        label="학습용 이미지 (여러 장)",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath"
                    )
                    gr.Markdown("💡 **팁**: 5-50장 권장, 다양한 각도/포즈가 좋습니다")

                    gr.Markdown("### ⚙️ 2. 기본 설정")

                    lora_name = gr.Textbox(
                        label="LoRA 이름",
                        placeholder="예: my_character_v1",
                        value="my_lora"
                    )

                    training_mode = gr.Radio(
                        choices=[
                            "🚀 테스트 모드 (2분 - 개발용)",
                            "👤 캐릭터 (20-30분)",
                            "🎨 스타일/화풍 (30-60분)",
                            "📦 오브젝트/컨셉 (20-40분)",
                        ],
                        label="학습 모드",
                        value="👤 캐릭터 (20-30분)",
                        info="이미지 유형에 맞는 모드를 선택하세요"
                    )

                    base_model = gr.Dropdown(
                        choices=[
                            "SD 1.5 (기본)",
                            "SD 2.1",
                            "SDXL 1.0",
                            "커스텀 모델..."
                        ],
                        label="베이스 모델",
                        value="SD 1.5 (기본)",
                        info="학습할 베이스 모델"
                    )

                    trigger_word = gr.Textbox(
                        label="트리거 워드 (선택)",
                        placeholder="예: ohwx, sks, mychar",
                        info="LoRA 활성화 키워드 (비워두면 자동)"
                    )

                    caption_method = gr.Radio(
                        choices=[
                            "BLIP (자연어)",
                            "WD14 (태그)",
                            "BLIP + WD14 (조합)"
                        ],
                        label="캡셔닝 방법",
                        value="BLIP (자연어)",
                        info="이미지 설명 생성 방법"
                    )

                    with gr.Accordion("🔧 고급 설정", open=False):
                        gr.Markdown("프리셋 파일(`configs/training_presets.yaml`)에서 수정 가능합니다.")
                        gr.Markdown("현재 UI에서 직접 수정은 지원하지 않습니다.")

                    train_btn = gr.Button(
                        "🚀 LoRA 학습 시작!",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 📊 진행 상황")
                    status_output = gr.Textbox(
                        label="상태",
                        lines=15,
                        max_lines=20,
                        value="이미지를 업로드하고 설정을 완료한 후 '학습 시작'을 눌러주세요."
                    )

                    gr.Markdown("### 💾 결과")
                    output_file = gr.File(
                        label="생성된 LoRA 파일",
                        type="filepath"
                    )

                    gr.Markdown(
                        """
                        ### 📝 사용 방법
                        1. ComfyUI의 `models/loras/` 폴더에 생성된 파일 복사
                        2. 프롬프트에 트리거 워드 추가
                        3. LoRA 강도는 0.7-0.9 추천
                        """
                    )

            # 이벤트 핸들러
            train_btn.click(
                fn=self.train_lora,
                inputs=[
                    images,
                    base_model,
                    lora_name,
                    training_mode,
                    trigger_word,
                    caption_method
                ],
                outputs=[status_output, output_file]
            )

            gr.Markdown(
                """
                ---
                ### ℹ️ 참고사항
                - **GPU 필요**: NVIDIA GPU (VRAM 12GB 이상 권장)
                - **학습 시간**: 모드에 따라 2분~1시간
                - **이미지 품질**: 고품질, 비슷한 해상도 권장
                - **데이터 다양성**: 다양한 각도/포즈가 학습 품질 향상

                **문제가 발생하면**: GitHub Issues에 제보해주세요
                """
            )

        return demo


def main():
    """메인 실행 함수"""
    app = LoraMakerUI()
    demo = app.create_ui()

    # 서버 실행
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
