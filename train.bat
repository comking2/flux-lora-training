@echo off
echo === Flux LoRA Training (Windows) ===

REM Check if data directory is provided
if "%1"=="" (
    echo Error: Please provide data directory
    echo Usage: train.bat "F:\path\to\data" [options]
    exit /b 1
)

set DATA_DIR=%1
set OUTPUT_DIR=%2
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=.\flux_lora_output
set EPOCHS=%3
if "%EPOCHS%"=="" set EPOCHS=10
set BATCH_SIZE=%4
if "%BATCH_SIZE%"=="" set BATCH_SIZE=1
set LORA_RANK=%5
if "%LORA_RANK%"=="" set LORA_RANK=8

echo Data directory: %DATA_DIR%
echo Output directory: %OUTPUT_DIR%
echo Epochs: %EPOCHS%
echo Batch size: %BATCH_SIZE%
echo LoRA rank: %LORA_RANK%
echo.

REM Activate virtual environment
if exist "env\Scripts\activate.bat" (
    echo Activating virtual environment...
    call env\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found
)

REM Run training
echo Starting training...
python train_lora.py --data_dir "%DATA_DIR%" --output_dir "%OUTPUT_DIR%" --epochs %EPOCHS% --batch_size %BATCH_SIZE% --lora_rank %LORA_RANK%

echo Training completed!
pause