# 🎨 LoraMaker

**ComfyUI용 LoRA를 자동으로 생성하는 간단한 도구**

복잡한 설정 없이 이미지만 업로드하면 자동으로 LoRA를 학습합니다!

## ✨ 특징

- 🚀 **원클릭 학습**: 이미지 업로드 → 버튼 클릭 → LoRA 완성
- 🤖 **자동 캡셔닝**: BLIP/WD14로 이미지 설명 자동 생성
- ⚙️ **검증된 설정**: Kohya-ss 기반 커뮤니티 베스트 프랙티스 적용
- 📊 **실시간 모니터링**: 학습 진행 상황 및 샘플 이미지 확인
- 🎯 **프리셋 제공**: 캐릭터/스타일/오브젝트 등 용도별 최적화
- 💾 **ComfyUI 호환**: safetensors 형식으로 즉시 사용 가능

## 📋 요구사항

### 하드웨어
- **최소**: NVIDIA GPU (12GB VRAM), RAM 16GB
- **권장**: NVIDIA RTX 4090 (24GB VRAM), RAM 32GB
- **저장공간**: 50GB 이상 (SSD 권장)

### 소프트웨어
- Python 3.10+
- CUDA 11.8+ or 12.1+
- Git

## 🚀 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/your-username/LoraMaker.git
cd LoraMaker
```

### 2. 가상환경 생성 (권장)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 의존성 설치

**빠른 설치** (최소 기능):
```bash
pip install -r requirements-minimal.txt
```

**전체 설치** (모든 기능):
```bash
pip install -r requirements.txt
```

### 4. (중요) Kohya-ss 설치

LoraMaker는 Kohya-ss를 학습 엔진으로 사용합니다:

```bash
# 서브모듈로 추가
git submodule add https://github.com/kohya-ss/sd-scripts.git external/sd-scripts
cd external/sd-scripts
pip install -r requirements.txt
cd ../..
```

또는 별도 설치:
```bash
pip install git+https://github.com/kohya-ss/sd-scripts.git
```

### 5. 실행
```bash
python ui/gradio_app.py
```

브라우저에서 `http://localhost:7860` 접속

## 📖 사용 방법

### 기본 사용
1. **이미지 업로드**: 5-50장 권장 (같은 캐릭터/스타일)
2. **모드 선택**:
   - 👤 캐릭터: 인물, 캐릭터 학습
   - 🎨 스타일: 화풍, 아트스타일
   - 📦 오브젝트: 특정 물체, 개념
3. **학습 시작**: 버튼 클릭!
4. **결과 다운로드**: ComfyUI에 복사

### 빠른 검증 (개발자용)
```bash
# 테스트 모드로 2분 내 파이프라인 검증
python ui/gradio_app.py
# UI에서 "🚀 테스트 모드" 선택
```

## 🧪 품질 검증 방법

### Golden Standard 비교

LoraMaker가 제대로 작동하는지 확인하려면:

1. **레퍼런스 데이터셋 준비**
```bash
# 테스트용 이미지 5-10장 준비
mkdir test_dataset
# 이미지들을 test_dataset/에 복사
```

2. **기준 LoRA 생성** (Kohya-ss GUI 등으로 한 번만)
   - 설정 저장 (learning rate, epochs 등)
   - `reference_lora.safetensors` 생성

3. **LoraMaker로 재현**
   - 같은 이미지, 같은 설정 사용
   - `our_lora.safetensors` 생성

4. **결과 비교**
   - ComfyUI에서 두 LoRA로 같은 프롬프트 테스트
   - 시각적으로 유사하면 ✅ 검증 완료

### 자동 검증 스크립트 (예정)
```bash
python scripts/validate_lora.py \
  --reference reference_lora.safetensors \
  --test our_lora.safetensors
```

## 📁 프로젝트 구조

```
LoraMaker/
├── backend/              # 백엔드 로직
│   ├── training/         # LoRA 학습
│   ├── preprocessing/    # 이미지 전처리, 캡셔닝
│   └── utils/            # 유틸리티
├── ui/                   # Gradio UI
│   └── gradio_app.py
├── configs/              # 학습 프리셋 설정
│   └── training_presets.yaml
├── models/               # 베이스 모델 (자동 다운로드)
├── outputs/              # 생성된 LoRA 파일
├── tests/                # 테스트 코드
├── requirements.txt      # 전체 의존성
├── requirements-minimal.txt  # 최소 의존성
└── README.md
```

## ⚙️ 학습 프리셋

`configs/training_presets.yaml`에 검증된 설정 포함:

- **test**: 개발용 (2분, dim=8)
- **character**: 캐릭터 학습 (20-30분, dim=32)
- **style**: 스타일 학습 (30-60분, dim=64)
- **concept**: 컨셉 학습 (20-40분, dim=32)
- **sdxl_character**: SDXL용 (40-60분, dim=64)

모든 프리셋은 Civitai, Kohya-ss 커뮤니티 베스트 프랙티스 기반입니다.

## 🎯 베스트 프랙티스

### 이미지 준비
- ✅ **수량**: 10-50장 권장
- ✅ **품질**: 고해상도, 선명한 이미지
- ✅ **다양성**: 다양한 각도, 포즈, 표정
- ✅ **일관성**: 같은 캐릭터/스타일 유지
- ❌ 흐릿한 이미지, 저해상도, 워터마크 피하기

### 학습 팁
- 처음엔 **테스트 모드**로 파이프라인 확인
- 이미지 10장 미만: **과적합 주의** (epochs 줄이기)
- 이미지 50장 이상: **학습 시간 증가** (강력한 LoRA)
- 결과가 과적합되면: epochs 감소, learning rate 낮추기

### ComfyUI 사용
```
1. 생성된 .safetensors 파일을 ComfyUI/models/loras/에 복사
2. "Load LoRA" 노드 추가
3. 프롬프트에 트리거 워드 포함 (설정한 경우)
4. LoRA 강도: 0.7-0.9 추천 (1.0은 과할 수 있음)
```

## 🔧 개발 로드맵

### Phase 1: MVP (현재)
- [x] 프로젝트 구조
- [x] 검증된 학습 프리셋
- [x] 기본 Gradio UI
- [ ] 이미지 전처리
- [ ] Kohya-ss wrapper
- [ ] 기본 캡셔닝

### Phase 2: 핵심 기능
- [ ] BLIP-2 자동 캡셔닝
- [ ] WD14 태거
- [ ] 실시간 진행률 표시
- [ ] 샘플 이미지 생성
- [ ] 자동 검증 스크립트

### Phase 3: 고급 기능
- [ ] 하이퍼파라미터 자동 조정
- [ ] Regularization 이미지 지원
- [ ] 배치 학습 큐
- [ ] 학습 이력 관리
- [ ] 클라우드 GPU 통합 (선택)

## 🤝 기여하기

버그 리포트, 기능 제안, PR 환영합니다!

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 라이선스

MIT License

## 🙏 감사

이 프로젝트는 다음 오픈소스에 기반합니다:
- [Kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) - LoRA 학습 엔진
- [Stability AI](https://stability.ai/) - Stable Diffusion
- [Hugging Face](https://huggingface.co/) - Transformers, Diffusers
- [Gradio](https://gradio.app/) - UI 프레임워크

## 📞 문의

- GitHub Issues: 버그 리포트, 기능 제안
- Discussions: 사용법 질문, 아이디어 공유

---

**LoraMaker**로 쉽고 빠르게 LoRA를 만들어보세요! 🚀
