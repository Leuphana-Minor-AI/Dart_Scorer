@echo off
cd /d "%~dp0"
chcp 65001 >nul

echo ============================================================
echo [Dart Scorer] Starte Live Dart Scorer AI...
echo ============================================================

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

"%PYTHON_EXE%" live_scorer.py --model models/dart_sense_yolov8n.pt %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [HINWEIS] Falls Pakete fehlen, fuehre zuerst setup_windows.bat aus!
    pause
)
