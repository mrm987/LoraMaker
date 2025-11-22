"""
LoraMaker - ComfyUI LoRA 자동 생성 도구
간단한 UI로 LoRA 학습 자동화
"""

import gradio as gr
import os
from pathlib import Path

# TODO: 실제 구현 시 import
# from backend.training.trainer import LoRATrainer
# from backend.preprocessing.image_processor import ImageProcessor
# from backend.preprocessing.captioner import AutoCaptioner


class LoraMakerUI:
    def __init__(self):
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)

    def train_lora(
        self,
        images,
        base_model,
        lora_name,
        training_mode,
        trigger_word,
        advanced_settings
    ):
        """
        LoRA 학습 메인 함수

        Args:
            images: 업로드된 이미지 리스트
            base_model: 베이스 모델 경로
            lora_name: 생성할 LoRA 이름
            training_mode: 학습 모드 (test/character/style/concept)
            trigger_word: 트리거 워드 (선택)
            advanced_settings: 고급 설정 표시 여부
        """
        if not images:
            return "❌ 이미지를 업로드해주세요!", None

        if not lora_name:
            return "❌ LoRA 이름을 입력해주세요!", None

        # TODO: 실제 구현
        status = f"""
        ✅ 설정 확인 완료!

        📊 학습 정보:
        - 이미지 수: {len(images)}장
        - 모드: {training_mode}
        - LoRA 이름: {lora_name}
        - 트리거 워드: {trigger_word if trigger_word else '(없음)'}

        🚧 [개발 중] 실제 학습 기능은 구현 예정입니다.

        다음 단계:
        1. 이미지 전처리
        2. 자동 캡셔닝
        3. LoRA 학습 시작
        4. safetensors 저장
        """

        return status, None

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

                    with gr.Accordion("🔧 고급 설정", open=False):
                        gr.Markdown("개발 중...")
                        learning_rate = gr.Slider(
                            minimum=1e-5,
                            maximum=1e-3,
                            value=1e-4,
                            step=1e-5,
                            label="Learning Rate",
                            interactive=False
                        )
                        network_dim = gr.Slider(
                            minimum=8,
                            maximum=128,
                            value=32,
                            step=8,
                            label="Network Dimension",
                            interactive=False
                        )

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
                    gr.State(False)  # advanced_settings
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
