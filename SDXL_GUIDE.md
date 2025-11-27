# SDXL 모델 학습 가이드 (Illustrious, NoobAI 등)

## RTX 4080 Super (16GB) 사용자용 가이드

### ✅ 하드웨어 요구사항
- **GPU**: RTX 4080 Super (16GB VRAM) ✅
- **RAM**: 32GB 권장
- **저장공간**: 100GB+ (모델 + 데이터 + 출력)

### 📊 16GB VRAM으로 SDXL 학습 가능성
**가능합니다!** 다만 다음 설정이 적용됩니다:
- Batch Size: 1
- Gradient Checkpointing: 활성화
- Mixed Precision: bf16 (더 메모리 효율적)
- Gradient Accumulation: 2 (효과적인 배치 크기 증가)

## 🎨 Illustrious, NoobAI 학습하기

### 1. 모델 준비
Illustrious나 NoobAI 체크포인트를 다운로드:
```bash
# 예시 경로 (실제 경로로 변경)
/path/to/models/illustrious-v1.0.safetensors
/path/to/models/noobai-xl-v1.0.safetensors
```

### 2. UI 설정

1. **이미지 업로드**: 10-50장 (SDXL은 데이터가 많을수록 좋음)

2. **학습 모드 선택**:
   - `👤 SDXL 캐릭터 (40-60분)` 선택

3. **베이스 모델**:
   - `커스텀 모델 경로 입력` 선택

4. **커스텀 모델 경로**:
   ```
   /path/to/models/illustrious-v1.0.safetensors
   ```

5. **캡셔닝 방법** (중요!):
   - Illustrious/NoobAI는 **Danbooru 태그 스타일**을 사용
   - 따라서 `WD14 (태그)` 또는 `BLIP + WD14 (조합)` 추천
   - BLIP만 사용하면 자연어 캡션이 생성되어 품질 저하 가능

6. **트리거 워드** (선택):
   - 예: `1girl`, `mychar`, 캐릭터 이름 등

### 3. SDXL 프리셋 설정 (자동 적용)

`configs/training_presets.yaml`의 `sdxl_character` 프리셋:
```yaml
sdxl_character:
  max_train_epochs: 8
  network_dim: 64        # SDXL은 더 큰 rank 필요
  network_alpha: 32
  learning_rate: 5e-5    # SDXL은 낮은 LR
  train_batch_size: 1    # 16GB VRAM
  resolution: 1024       # SDXL 기본 해상도
  mixed_precision: "bf16"  # BF16 권장
  gradient_accumulation_steps: 2
```

### 4. 메모리 최적화 팁

#### 만약 OOM (Out of Memory) 발생 시:
`configs/training_presets.yaml`에서 수정:

```yaml
sdxl_character:
  # 해상도 낮추기
  resolution: 768  # 1024 → 768

  # 또는 bucket 최대 해상도 낮추기
  max_bucket_reso: 1536  # 2048 → 1536

  # Cache latents (속도 향상 + 메모리 절약)
  cache_latents: true
  cache_latents_to_disk: true
```

### 5. 학습 시간 예상
**RTX 4080 Super 기준**:
- 10장 이미지, 8 epochs: ~30-40분
- 30장 이미지, 8 epochs: ~60-90분
- 50장 이미지, 8 epochs: ~2-3시간

### 6. 학습 결과 활용

#### ComfyUI에서 사용:
```
1. 생성된 .safetensors를 ComfyUI/models/loras/에 복사
2. Load LoRA 노드 추가
3. 강도: 0.6-0.8 (SDXL은 SD 1.5보다 강도 낮춤)
4. 프롬프트 예시:
   - "1girl, mychar, portrait, high quality"
   - Illustrious/NoobAI는 quality tags 중요:
     "masterpiece, best quality, ..."
```

## 🔧 Illustrious/NoobAI 특화 팁

### 캡셔닝 전략
1. **WD14 Tagger 사용** (강력 추천)
   - Danbooru 태그 스타일과 완벽 호환
   - `1girl, blonde hair, blue eyes, ...` 형식

2. **수동 캡션 편집** (최고 품질)
   - 생성된 데이터셋의 .txt 파일을 직접 수정
   - 캐릭터 특징 태그 추가
   - 불필요한 태그 제거

### 트리거 워드 전략
- **옵션 1**: 캐릭터 이름 사용
  - 예: `hatsune_miku`

- **옵션 2**: 간단한 고유 토큰
  - 예: `ohwx`, `sks`

- **옵션 3**: 없음
  - LoRA가 자연스럽게 스타일 학습
  - 프롬프트에 특징 명시 필요

### 품질 향상 팁
1. **이미지 해상도**: 최소 1024px (SDXL 네이티브)
2. **이미지 다양성**: 다양한 포즈, 표정, 각도
3. **배경 다양성**: 단색 배경만 피하기
4. **일관성**: 같은 캐릭터/스타일 유지

## ⚠️ 주의사항

### VRAM 관리
- 다른 프로그램 종료 (브라우저, Discord 등)
- 학습 중 모니터링: `nvidia-smi -l 1`
- OOM 발생 시: 해상도 낮추기

### 학습 중단 방지
- 화면 보호기 비활성화
- 전원 관리 "고성능" 모드
- 안정적인 전력 공급

### 체크포인트 활용
- `save_every_n_epochs: 2` (기본값)
- 중간 결과 확인 가능
- 과적합 조기 발견

## 📈 학습 결과 평가

### 좋은 LoRA 판단 기준:
- ✅ 캐릭터 특징 재현 (머리색, 눈색, 복장 등)
- ✅ 다양한 포즈/각도에서 일관성
- ✅ 베이스 모델 스타일 유지
- ✅ 0.6-0.8 강도로 적절한 영향

### 문제 징후:
- ❌ 과적합: 학습 이미지와 똑같이만 생성
- ❌ 과소학습: 캐릭터 특징 재현 안됨
- ❌ 스타일 파괴: 베이스 모델 품질 저하

### 해결 방법:
**과적합**:
- Epochs 줄이기 (8 → 6)
- Learning rate 낮추기
- 데이터 더 다양하게

**과소학습**:
- Epochs 늘리기 (8 → 10-12)
- Network dim 늘리기 (64 → 96-128)
- 데이터 품질 개선

## 🎯 추천 설정 (RTX 4080 Super)

```yaml
# configs/training_presets.yaml 수정
sdxl_character:
  max_train_epochs: 8           # 기본값
  network_dim: 64               # 64-96 추천
  network_alpha: 32
  learning_rate: 5e-5           # 안정적
  train_batch_size: 1           # 16GB 최적
  resolution: 1024              # SDXL 기본
  mixed_precision: "bf16"       # 필수
  gradient_accumulation_steps: 2
  cache_latents_to_disk: true   # 메모리 절약

  # 품질 향상
  min_snr_gamma: 5
  noise_offset: 0.05

  # 샘플링 (진행 확인용)
  sample_every_n_epochs: 2
  sample_prompts: "masterpiece, best quality, 1girl, portrait"
```

**행운을 빕니다!** 🚀 Illustrious/NoobAI로 멋진 LoRA를 만들어보세요!
