# SDXL 스타일 LoRA 학습 가이드 (화풍/아트스타일)

## 작가 화풍 학습하기 (Illustrious, NoobAI 기반)

스타일 LoRA는 **특정 작가의 화풍**이나 **아트스타일**을 학습합니다. 캐릭터가 아닌 **그림체, 색감, 구도, 질감** 등을 재현합니다.

### 🎨 스타일 LoRA vs 캐릭터 LoRA

| 비교 항목 | 스타일 LoRA | 캐릭터 LoRA |
|----------|------------|-------------|
| **목적** | 화풍, 그림체 재현 | 특정 인물 재현 |
| **Network Dim** | 96-128 (큼) | 32-64 (중간) |
| **Learning Rate** | 3e-5 (낮음) | 5e-5 (보통) |
| **Epochs** | 12-15 (많음) | 8-10 (보통) |
| **이미지 수** | 30-100장 | 10-50장 |
| **다양성** | 여러 주제 필요 | 같은 캐릭터 |
| **학습 시간** | 60-120분 | 40-60분 |

### ⚙️ SDXL 스타일 프리셋 설정

이제 `🎨 SDXL 스타일/화풍 (60-90분)` 모드가 추가되었습니다!

**자동 적용 설정**:
```yaml
network_dim: 96          # 더 큰 표현력
network_alpha: 48
learning_rate: 3e-5      # 안정적 학습
epochs: 12               # 충분한 학습
resolution: 1024         # SDXL 기본
noise_offset: 0.1        # 스타일 학습 강화
```

### 📸 이미지 준비 (중요!)

#### ✅ 좋은 스타일 학습 데이터:
1. **다양한 주제**
   - 인물 (남/여, 다양한 포즈)
   - 풍경
   - 정물
   - 다양한 구도

2. **일관된 화풍**
   - 같은 작가의 작품만
   - 비슷한 시기 작품 (화풍 변화 주의)
   - 완성작만 (러프 스케치 제외)

3. **고품질**
   - 1024px 이상
   - 워터마크 없음
   - 압축 적음

#### ❌ 피해야 할 것:
- 다른 작가 작품 섞기
- 너무 비슷한 이미지 (중복 효과)
- 텍스트가 많은 이미지
- 극단적으로 다른 스타일 섞기

### 🎯 권장 이미지 수

| 이미지 수 | 학습 결과 | 추천 |
|----------|-----------|------|
| 10-20장 | 약한 스타일 | ⚠️ 부족 |
| 30-50장 | 적절한 스타일 | ✅ 권장 |
| 50-100장 | 강한 스타일 | ✅ 최적 |
| 100장+ | 매우 강한 스타일 | ⚡ 최고 |

### 💬 캡셔닝 전략 (화풍 학습)

#### WD14 Tagger (강력 추천!)
```
장점:
- Illustrious/NoobAI와 완벽 호환
- 화풍 관련 태그 자동 추출
  (sketch, lineart, watercolor, oil painting 등)
- Danbooru 스타일 태그
```

**예시 캡션**:
```
1girl, solo, long hair, blue eyes, portrait, sketch, monochrome, upper body
landscape, scenery, sky, cloud, detailed background, watercolor (medium)
still life, flower, vase, painting (medium), soft lighting
```

#### 수동 편집 (최고 품질)
생성된 .txt 파일에 **공통 스타일 태그** 추가:
```
# 모든 이미지 캡션에 추가
artist_name_style, painterly, soft shading, warm colors

# 또는 트리거 워드 사용
mystyle, detailed, cinematic lighting
```

### 🚀 UI 사용법 (화풍 학습)

1. **이미지 업로드**: 30-100장 (다양한 주제)

2. **학습 모드**: `🎨 SDXL 스타일/화풍 (60-90분)` ← 새로 추가됨!

3. **베이스 모델**: `커스텀 모델 경로 입력`

4. **커스텀 모델 경로**:
   ```
   /path/to/illustrious-v1.0.safetensors
   또는
   /path/to/noobai-xl-v1.0.safetensors
   ```

5. **캡셔닝 방법**: `WD14 (태그)` ← 기본값으로 설정됨

6. **트리거 워드** (선택):
   - 작가 이름: `artist_name_style`
   - 간단한 태그: `mystyle`, `artstyle`
   - 비워두기: 자연스러운 화풍 학습

7. **LoRA 이름**: `artist_name_style_v1`

### ⏱️ 학습 시간 (RTX 4080 Super)

| 이미지 수 | Epochs | 예상 시간 |
|----------|--------|----------|
| 30장 | 12 | 60-75분 |
| 50장 | 12 | 90-120분 |
| 100장 | 12 | 150-180분 |

### 🔧 고급 설정 (프리셋 커스터마이징)

`configs/training_presets.yaml`의 `sdxl_style` 수정:

#### 더 강한 스타일 학습:
```yaml
sdxl_style:
  network_dim: 128      # 96 → 128 (더 큰 표현력)
  max_train_epochs: 15  # 12 → 15 (더 많은 학습)
  noise_offset: 0.15    # 0.1 → 0.15 (더 강한 효과)
```

#### 메모리 절약 (OOM 시):
```yaml
sdxl_style:
  resolution: 768       # 1024 → 768
  max_bucket_reso: 1536 # 2048 → 1536
  network_dim: 64       # 96 → 64
```

#### 더 안정적 학습:
```yaml
sdxl_style:
  learning_rate: 2e-5   # 3e-5 → 2e-5 (더 낮게)
  lr_warmup_ratio: 0.2  # 0.15 → 0.2 (긴 warmup)
```

### 💾 결과 활용 (ComfyUI)

#### 프롬프트 사용법:
```
# 트리거 워드 사용한 경우
mystyle, 1girl, portrait, masterpiece, best quality

# 트리거 워드 없는 경우
1girl, portrait, masterpiece, best quality
(LoRA 강도로 스타일 조절)
```

#### LoRA 강도 권장:
```
0.4-0.6: 은은한 화풍 (베이스 모델 스타일 유지)
0.6-0.8: 적절한 화풍 (추천)
0.8-1.0: 강한 화풍 (작가 스타일 강조)
1.0+: 매우 강한 화풍 (과할 수 있음)
```

### 📊 품질 평가

#### ✅ 좋은 스타일 LoRA:
- 다양한 주제에서 일관된 화풍
- 베이스 모델 품질 유지
- 색감, 선, 질감 등 재현
- 0.6-0.8 강도로 적절한 영향

#### ❌ 문제 징후:
- **과적합**: 학습 이미지와 똑같이만 생성
  - 해결: Epochs 줄이기, LR 낮추기

- **과소학습**: 화풍이 약함
  - 해결: Epochs 늘리기, dim 늘리기

- **스타일 파괴**: 이상한 결과물
  - 해결: 데이터 정리, LR 낮추기

### 🎨 화풍별 팁

#### 수채화 스타일:
```yaml
noise_offset: 0.15  # 부드러운 효과
sample_prompts: "watercolor (medium), soft colors, painting"
```
- 이미지: 수채화 작품 30-50장
- 태그: `watercolor, soft shading, light colors`

#### 유화 스타일:
```yaml
network_dim: 128    # 복잡한 질감
sample_prompts: "oil painting, thick paint, textured"
```
- 이미지: 유화 작품 40-60장
- 태그: `oil painting, impasto, thick paint`

#### 애니메이션 스타일:
```yaml
network_dim: 96
sample_prompts: "anime style, cel shading, clean lines"
```
- 이미지: 애니메이션 스틸컷 30-50장
- 태그: `anime screencap, official art, clean lineart`

#### 만화/일러스트 스타일:
```yaml
learning_rate: 3e-5
sample_prompts: "manga style, screentones, detailed lineart"
```
- 이미지: 만화/일러스트 40-80장
- 태그: `manga, lineart, screentones`

### 💡 프로 팁

1. **이미지 큐레이션이 핵심**
   - 품질 > 수량
   - 일관성 체크
   - 대표작 위주

2. **중간 체크포인트 활용**
   - 2 epoch마다 저장됨
   - ComfyUI에서 테스트
   - 과적합 조기 발견

3. **여러 버전 학습**
   - dim 64, 96, 128 각각 학습
   - epochs 변경해서 테스트
   - 가장 좋은 결과 선택

4. **A/B 테스트**
   - 같은 프롬프트로 여러 LoRA 비교
   - 베이스 모델과 비교
   - 강도별 비교

5. **데이터셋 개선**
   - 결과 나쁘면 데이터 재점검
   - 일관성 없는 이미지 제거
   - 대표적 작품 추가

### 🎯 권장 워크플로우

1. **데이터 수집** (가장 중요!)
   - 30-50장 고품질 이미지
   - 다양한 주제 (인물, 풍경, 정물 등)
   - 일관된 화풍

2. **첫 학습** (기본 설정)
   - SDXL 스타일 모드
   - WD14 캡셔닝
   - 기본 프리셋 (dim=96, epochs=12)

3. **테스트**
   - ComfyUI에서 여러 프롬프트 테스트
   - 강도 0.4-1.0 실험

4. **평가 및 조정**
   - 과적합: epochs 줄이기
   - 과소학습: epochs 늘리기, dim 늘리기
   - 만족: 완료!

5. **파인튜닝** (선택)
   - 더 많은 이미지 추가
   - 프리셋 미세 조정
   - 재학습

**화풍 LoRA 학습 성공을 기원합니다!** 🎨✨
