@echo off
title Visdeurbel - Night Training Run
echo ============================================
echo   Visdeurbel Night Training Run
echo   Started: %date% %time%
echo ============================================
echo.
echo Step 1/2: Training (no eval pauses, ~40 min)
echo Close this window ONLY if you want to cancel.
echo.

cd /d "%~dp0"
python train_fast.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Training failed - check output above
    pause
    exit /b 1
)

echo.
echo ============================================
echo Step 2/2: Evaluating fine-tuned adapter...
echo ============================================
echo.

python inference.py --eval
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Evaluation failed - check output above
    pause
    exit /b 1
)

echo.
echo ============================================
echo   ALL DONE - %date% %time%
echo   Adapter saved to: output/granite-fish-lora-fast/
echo ============================================
pause
