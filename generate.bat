@echo off
echo === Flux LoRA Inference (Windows) ===

REM Default values
set LORA_PATH=%1
set PROMPT=%2
set OUTPUT_DIR=%3
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=.\generated_images

if "%LORA_PATH%"=="" (
    echo Error: Please provide LoRA model path
    echo Usage: generate.bat "path\to\lora" "your prompt" [output_dir]
    exit /b 1
)

if "%PROMPT%"=="" (
    echo Error: Please provide prompt
    echo Usage: generate.bat "path\to\lora" "your prompt" [output_dir]
    exit /b 1
)

echo LoRA path: %LORA_PATH%
echo Prompt: %PROMPT%
echo Output directory: %OUTPUT_DIR%
echo.

REM Activate virtual environment
if exist "env\Scripts\activate.bat" (
    echo Activating virtual environment...
    call env\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found
)

REM Run inference
echo Starting generation...
python inference.py --lora_path "%LORA_PATH%" --prompt "%PROMPT%" --output_dir "%OUTPUT_DIR%"

echo Generation completed!
pause