# 🎨 Flux LoRA Training Pipeline

FLUX.1 모델을 위한 LoRA(Low-Rank Adaptation) 훈련 파이프라인입니다.

## ⚠️ 시스템 요구사항

### 최소 요구사양:
- **GPU**: 16GB VRAM 이상 (RTX 4080, A100 등)
- **RAM**: 32GB 이상
- **저장공간**: 50GB 여유공간

### 권장 사양:
- **GPU**: 24GB+ VRAM (RTX 4090, A100 40GB 등)
- **RAM**: 64GB 이상

### 🔥 Google Colab 사용 권장!
- **Colab Pro**: A100 40GB GPU + 51GB RAM
- **Colab Free**: T4 16GB GPU + 12GB RAM (최적화 필요)

## 구조

```
flux_custom/
├── dataset_processor.py    # 데이터셋 처리 및 로딩
├── train_lora.py          # LoRA 훈련 스크립트
├── inference.py           # 추론 및 이미지 생성
├── config.json           # 훈련 설정 파일
├── requirements.txt      # 필요한 패키지 목록
├── train.sh             # 훈련 실행 스크립트
├── generate.sh          # 생성 실행 스크립트
└── README.md            # 이 파일
```

## 설치

1. 가상환경 활성화:
```bash
source env/bin/activate  # Linux/Mac
# 또는
env\Scripts\activate     # Windows
```

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 데이터셋 준비

### 방법 1: 이미지-텍스트 쌍 파일
각 이미지에 대해 동일한 이름의 .txt 파일을 생성:
```
training_data/
├── image1.jpg
├── image1.txt
├── image2.png
├── image2.txt
└── ...
```

### 방법 2: JSON 형식
```json
[
  {
    "image": "image1.jpg",
    "caption": "이미지 설명"
  },
  {
    "image": "image2.png",
    "caption": "다른 이미지 설명"
  }
]
```

### 방법 3: CSV 형식
```csv
image,caption
image1.jpg,"이미지 설명"
image2.png,"다른 이미지 설명"
```

## 훈련

### 기본 훈련:
```bash
./train.sh --data_dir ./training_data
```

### 고급 옵션:
```bash
./train.sh --data_dir ./training_data \
          --epochs 20 \
          --batch_size 2 \
          --learning_rate 5e-5 \
          --lora_rank 32 \
          --use_wandb
```

### Python 직접 실행:
```bash
python train_lora.py --data_dir ./training_data \
                     --output_dir ./my_lora_model \
                     --epochs 15 \
                     --batch_size 1 \
                     --learning_rate 1e-4
```

## 추론

### 단일 프롬프트 생성:
```bash
./generate.sh --lora_path ./flux_lora_output/flux-lora-final \
              --prompt "a beautiful landscape with mountains"
```

### 여러 프롬프트 배치 생성:
```bash
# prompts.txt 파일에 프롬프트들을 한 줄씩 작성
./generate.sh --lora_path ./flux_lora_output/flux-lora-final \
              --prompts_file prompts.txt
```

### 테스트 실행:
```bash
./generate.sh --lora_path ./flux_lora_output/flux-lora-final --test
```

### 모델 비교:
```bash
./generate.sh --lora_path ./flux_lora_output/flux-lora-final --compare
```

### Python 직접 실행:
```bash
python inference.py --lora_path ./flux_lora_output/flux-lora-final \
                    --prompt "your prompt here" \
                    --height 1024 \
                    --width 1024 \
                    --steps 50
```

## 설정

`config.json` 파일을 수정하여 기본 설정을 변경할 수 있습니다:

```json
{
  "model_name": "black-forest-labs/FLUX.1-dev",
  "batch_size": 1,
  "epochs": 10,
  "learning_rate": 1e-4,
  "lora_rank": 16,
  "lora_alpha": 32,
  "image_size": 512
}
```

## 주요 매개변수

### 훈련 매개변수:
- `--epochs`: 훈련 에포크 수
- `--batch_size`: 배치 크기 (GPU 메모리에 따라 조정)
- `--learning_rate`: 학습률
- `--lora_rank`: LoRA 랭크 (작을수록 파라미터 적음)
- `--lora_alpha`: LoRA 알파 (일반적으로 rank의 2배)

### 추론 매개변수:
- `--height`, `--width`: 생성 이미지 크기
- `--steps`: 추론 스텝 수 (많을수록 품질 좋지만 느림)
- `--guidance_scale`: 가이던스 스케일 (7.5가 일반적)
- `--seed`: 재현 가능한 결과를 위한 시드

## 메모리 요구사항

- **최소**: 12GB VRAM (batch_size=1, fp16)
- **권장**: 24GB VRAM (batch_size=2-4)
- **CPU**: 16GB RAM 이상

## 문제해결

1. **CUDA 메모리 부족**: batch_size를 줄이거나 image_size를 512로 설정
2. **훈련 속도 느림**: gradient_accumulation_steps를 사용하여 effective batch size 증가
3. **품질 문제**: epochs 수 증가, learning_rate 조정, 더 많은 훈련 데이터 사용

## 출력 구조

훈련 후 다음과 같은 구조로 출력됩니다:

```
flux_lora_output/
├── flux-lora-final/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── training_config.json
├── checkpoint-500/
├── checkpoint-1000/
└── ...
```