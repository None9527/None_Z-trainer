@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ============================================================================
:: None Trainer - Windows Setup Script
:: ============================================================================

echo ================================================
echo    None Trainer - Setup
echo ================================================
echo.

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Embedded or system Python
set "PYTHON_EXE=%SCRIPT_DIR%python\python.exe"
set "PIP_EXE=%SCRIPT_DIR%python\Scripts\pip.exe"
set "NODE_EXE=%SCRIPT_DIR%nodejs\node.exe"
set "NPM_CMD=%SCRIPT_DIR%nodejs\npm.cmd"

:: Fallback to system Python
if not exist "%PYTHON_EXE%" (
    where python >nul 2>&1
    if %errorlevel%==0 (
        for /f "tokens=*" %%i in ('where python') do set "PYTHON_EXE=%%i"
        set "PIP_EXE=pip"
    ) else (
        echo [ERROR] Python not found!
        pause
        exit /b 1
    )
)

:: [1/6] Check Python
echo [1/6] Check Python...
for /f "tokens=2" %%i in ('"%PYTHON_EXE%" --version 2^>^&1') do set PYTHON_VERSION=%%i
echo   Python: %PYTHON_VERSION%

:: [2/6] Check CUDA
echo.
echo [2/6] Check CUDA...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo   [WARNING] No NVIDIA GPU detected
) else (
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do echo   GPU: %%i
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu^=memory.total --format^=csv^,noheader 2^>nul') do echo   VRAM: %%i
)

:: [3/6] Check PyTorch
echo.
echo [3/6] Check PyTorch...
"%PYTHON_EXE%" -c "import torch; print('PyTorch:', torch.__version__)" 2>nul
if errorlevel 1 (
    echo   [ERROR] PyTorch not installed!
    echo   Install manually:
    echo     CUDA 12.8: "%PIP_EXE%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    echo     CUDA 12.1: "%PIP_EXE%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo.
    pause
    exit /b 1
)

:: [4/6] Install Python dependencies
echo.
echo [4/6] Install Python dependencies...
"%PIP_EXE%" install -r requirements.txt -q

:: Install diffusers latest
echo   Installing diffusers (git latest)...
"%PIP_EXE%" install git+https://github.com/huggingface/diffusers.git -q

:: [5/6] Create .env
if not exist ".env" (
    copy env.example .env >nul
    echo   Created .env config file
)

:: [6/6] Check Node.js and build frontend
echo.
echo [6/6] Build frontend...
if not exist "%NODE_EXE%" (
    where npm >nul 2>&1
    if %errorlevel%==0 (
        set "NPM_CMD=npm"
    ) else (
        echo   [WARNING] Node.js not found, skip frontend build
        echo   Backend will still work.
        goto :done
    )
)

cd /d "%SCRIPT_DIR%frontend"
if not exist "node_modules" (
    echo   Installing frontend dependencies...
    call "%NPM_CMD%" install --silent
)
echo   Building frontend...
call "%NPM_CMD%" run build --silent
cd /d "%SCRIPT_DIR%"
echo   Frontend build done

:done
echo.
echo ================================================
echo   Setup complete!
echo ================================================
echo.
echo Next steps:
echo   1. Edit .env to set model path
echo   2. Run start.bat to launch
echo.
pause
