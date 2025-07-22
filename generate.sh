#!/bin/bash

# Flux LoRA Inference Script

set -e

# Default values
LORA_PATH=""
PROMPT=""
OUTPUT_DIR="./generated_images"
HEIGHT=1024
WIDTH=1024
STEPS=50
GUIDANCE_SCALE=7.5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lora_path)
            LORA_PATH="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --prompts_file)
            PROMPTS_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --guidance_scale)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --seed)
            SEED="--seed $2"
            shift 2
            ;;
        --test)
            TEST_FLAG="--test"
            shift
            ;;
        --compare)
            COMPARE_FLAG="--compare"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --lora_path         Path to LoRA model"
            echo "  --prompt            Text prompt for generation"
            echo "  --prompts_file      File containing multiple prompts"
            echo "  --output_dir        Output directory (default: ./generated_images)"
            echo "  --height            Image height (default: 1024)"
            echo "  --width             Image width (default: 1024)"
            echo "  --steps             Inference steps (default: 50)"
            echo "  --guidance_scale    Guidance scale (default: 7.5)"
            echo "  --seed              Random seed"
            echo "  --test              Run test with default prompts"
            echo "  --compare           Compare base model vs LoRA"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Flux LoRA Inference ==="
echo "LoRA path: $LORA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Image size: ${WIDTH}x${HEIGHT}"
echo "Inference steps: $STEPS"
echo "Guidance scale: $GUIDANCE_SCALE"
echo "=========================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if required packages are installed
python -c "import torch, diffusers, peft" 2>/dev/null || {
    echo "Error: Required packages not found"
    echo "Please install requirements: pip install -r requirements.txt"
    exit 1
}

# Build command
CMD="python inference.py --output_dir \"$OUTPUT_DIR\" --height $HEIGHT --width $WIDTH --steps $STEPS --guidance_scale $GUIDANCE_SCALE"

if [ -n "$LORA_PATH" ]; then
    CMD="$CMD --lora_path \"$LORA_PATH\""
fi

if [ -n "$PROMPT" ]; then
    CMD="$CMD --prompt \"$PROMPT\""
fi

if [ -n "$PROMPTS_FILE" ]; then
    CMD="$CMD --prompts_file \"$PROMPTS_FILE\""
fi

if [ -n "$SEED" ]; then
    CMD="$CMD $SEED"
fi

if [ -n "$TEST_FLAG" ]; then
    CMD="$CMD $TEST_FLAG"
fi

if [ -n "$COMPARE_FLAG" ]; then
    CMD="$CMD $COMPARE_FLAG"
fi

# Run inference
echo "Starting inference..."
eval $CMD

echo "Generation completed!"
echo "Images saved in: $OUTPUT_DIR"