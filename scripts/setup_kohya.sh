#!/bin/bash
# Kohya-ss 설치 스크립트

echo "=========================================="
echo "Kohya-ss sd-scripts 설치"
echo "=========================================="

# 1. external 디렉토리 생성
mkdir -p external

# 2. Kohya-ss 클론
if [ -d "external/sd-scripts" ]; then
    echo "✅ sd-scripts가 이미 존재합니다. 업데이트 중..."
    cd external/sd-scripts
    git pull
    cd ../..
else
    echo "📥 sd-scripts 클론 중..."
    git clone https://github.com/kohya-ss/sd-scripts.git external/sd-scripts
fi

# 3. 의존성 설치
echo "📦 sd-scripts 의존성 설치 중..."
cd external/sd-scripts
pip install -r requirements.txt
cd ../..

echo ""
echo "=========================================="
echo "✅ Kohya-ss 설치 완료!"
echo "=========================================="
echo ""
echo "다음 단계:"
echo "  python ui/gradio_app.py"
