@echo off
cd /d "%~dp0"
chcp 65001 >nul

echo ============================================================
echo [Dart Scorer] Windows Installation
echo ============================================================
echo Installiere notwendige Pakete in Anaconda...

REM Python suchen: eigenes Anaconda, sonst Python aus dem PATH
set PYTHON_EXE=%USERPROFILE%\anaconda3\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=%LOCALAPPDATA%\anaconda3\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python

"%PYTHON_EXE%" --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [FEHLER] Python nicht gefunden. Bitte Anaconda oder Python installieren.
    pause
    exit /b 1
)
set PIP_EXE="%PYTHON_EXE%" -m pip

echo.
echo [1/2] Installiere PyTorch (CPU-Version fuer Laptop)...
%PIP_EXE% install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo [2/2] Installiere Ultralytics (YOLO) und OpenCV...
%PIP_EXE% install ultralytics opencv-python

echo.
echo ============================================================
echo [OK] Installation abgeschlossen!
echo Starte nun das Programm mit run_dart_scorer.bat.
echo ============================================================
pause
