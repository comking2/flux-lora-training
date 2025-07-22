#!/bin/bash

# Environment setup script for Flux LoRA

# Load .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(cat .env | grep -v '#' | xargs)
else
    echo "No .env file found, using defaults"
fi

# Display current model setting
echo "Current FLUX model: ${FLUX_MODEL_NAME:-black-forest-labs/FLUX.1-schnell}"

# Examples of usage:
echo ""
echo "Usage examples:"
echo "  # Use FLUX.1-dev model:"
echo "  export FLUX_MODEL_NAME=black-forest-labs/FLUX.1-dev"
echo ""
echo "  # Use FLUX.1-schnell model (default):"
echo "  export FLUX_MODEL_NAME=black-forest-labs/FLUX.1-schnell"
echo ""
echo "  # Then run training:"
echo "  python train_lora.py --data_dir /path/to/data"