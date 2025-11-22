# 🚀 빠른 시작 가이드

## 1. 설치

### 필수 요구사항
- Python 3.10+
- NVIDIA GPU (CUDA 지원)
- Git

### 설치 단계

```bash
# 1. 저장소 클론
git clone https://github.com/your-username/LoraMaker.git
cd LoraMaker

# 2. 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. Kohya-ss 설치 (중요!)
bash scripts/setup_kohya.sh
# 또는 수동:
# git clone https://github.com/kohya-ss/sd-scripts.git external/sd-scripts
# cd external/sd-scripts && pip install -r requirements.txt && cd ../..

# 5. 환경 검증
python scripts/validate_setup.py
```

## 2. 실행

```bash
python ui/gradio_app.py
```

브라우저에서 `http://localhost:7860` 접속

## 3. 사용 방법

### 기본 워크플로우

1. **이미지 업로드**
   - 5-50장의 학습용 이미지 업로드
   - 같은 캐릭터/스타일의 다양한 각도/포즈

2. **설정**
   - LoRA 이름 입력
   - 학습 모드 선택 (캐릭터/스타일/컨셉)
   - 베이스 모델 선택
   - (선택) 트리거 워드 입력

3. **캡셔닝 방법**
   - BLIP: 자연어 설명 (일반적)
   - WD14: 태그 형식 (애니메이션/일러스트)
   - BLIP + WD14: 조합 (최고 품질)

4. **학습 시작**
   - "LoRA 학습 시작!" 버튼 클릭
   - 진행 상황 모니터링
   - 완료 시 .safetensors 파일 다운로드

## 4. 빠른 테스트

처음 사용자는 **테스트 모드**를 추천합니다:

1. 이미지 3-5장 업로드
2. 모드: "🚀 테스트 모드 (2분)"
3. 나머지 기본값 사용
4. 학습 시작

→ 2분 내에 파이프라인이 정상 작동하는지 확인 가능

## 5. ComfyUI에서 사용

```bash
# 1. 생성된 LoRA를 ComfyUI로 복사
cp outputs/my_lora_*/my_lora.safetensors /path/to/ComfyUI/models/loras/

# 2. ComfyUI 워크플로우에서:
# - "Load LoRA" 노드 추가
# - LoRA 파일 선택
# - 강도: 0.7-0.9 추천
# - 프롬프트에 트리거 워드 포함 (설정한 경우)
```

## 6. 문제 해결

### CUDA 오류
```bash
# PyTorch CUDA 재설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Kohya-ss 없음
```bash
bash scripts/setup_kohya.sh
```

### 메모리 부족 (OOM)
- 테스트 모드로 시작
- 이미지 수 줄이기
- 해상도 낮추기 (configs/training_presets.yaml 수정)

### 학습이 느림
- GPU 드라이버 최신 버전 확인
- xformers 설치: `pip install xformers`

## 7. 팁

### 좋은 결과를 위한 팁
- ✅ 고품질 이미지 사용 (선명, 고해상도)
- ✅ 다양한 각도/포즈 (최소 10장)
- ✅ 일관된 스타일 유지
- ✅ 배경이 다양한 이미지 포함
- ❌ 흐릿한 이미지 피하기
- ❌ 워터마크 있는 이미지 피하기
- ❌ 극단적인 가로세로 비율 피하기

### 프리셋 커스터마이징
`configs/training_presets.yaml` 파일 수정:
- Learning rate 조정
- Epochs 수 변경
- Network dimension 변경
- 등등...

### 베이스 모델 다운로드
UI에서 자동으로 Hugging Face에서 다운로드됩니다.
로컬 모델을 사용하려면:
1. `backend/training/trainer.py` 수정
2. 또는 UI 코드에서 model_paths 변경

## 8. 다음 단계

- [전체 문서](README.md) 읽기
- [기술 리뷰](TECHNICAL_REVIEW.md) 확인
- 프리셋 실험하기
- 결과 공유하기

**즐거운 LoRA 학습 되세요!** 🎨
