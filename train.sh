#!/bin/bash

# Flux LoRA Training Script

set -e

# Default values
DATA_DIR=""
OUTPUT_DIR="./flux_lora_output"
EPOCHS=10
BATCH_SIZE=1
LEARNING_RATE=1e-4
LORA_RANK=16

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lora_rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --use_wandb)
            USE_WANDB="--use_wandb"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --data_dir <path> [options]"
            echo "Options:"
            echo "  --data_dir          Training data directory (required)"
            echo "  --output_dir        Output directory (default: ./flux_lora_output)"
            echo "  --epochs            Number of epochs (default: 10)"
            echo "  --batch_size        Batch size (default: 1)"
            echo "  --learning_rate     Learning rate (default: 1e-4)"
            echo "  --lora_rank         LoRA rank (default: 16)"
            echo "  --use_wandb         Enable Weights & Biases logging"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if data directory is provided
if [ -z "$DATA_DIR" ]; then
    echo "Error: --data_dir is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist"
    exit 1
fi

echo "=== Flux LoRA Training ==="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "LoRA rank: $LORA_RANK"
echo "=========================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected"
    echo "It's recommended to use a virtual environment"
fi

# Check if required packages are installed
python -c "import torch, diffusers, peft, transformers" 2>/dev/null || {
    echo "Error: Required packages not found"
    echo "Please install requirements: pip install -r requirements.txt"
    exit 1
}

# Start training
echo "Starting training..."
python train_lora.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --lora_rank "$LORA_RANK" \
    $USE_WANDB

echo "Training completed!"
echo "Model saved in: $OUTPUT_DIR"