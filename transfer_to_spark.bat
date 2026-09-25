@echo off
REM ==============================================================================
REM Helper script to transfer Dart Training dataset and scripts to Nvidia Spark
REM ==============================================================================

echo ============================================================
echo   Transfer Dart Training to Nvidia Spark (Linux)
echo ============================================================
echo.

set /p SPARK_USER="Enter Spark username (e.g. nvidia): "
set /p SPARK_IP="Enter Spark IP address or hostname (e.g. 192.168.1.50): "
set /p SPARK_DEST="Enter destination path on Spark [default: ~/dart-training]: "

if "%SPARK_DEST%"=="" set SPARK_DEST=~/dart-training

echo.
echo Target: %SPARK_USER%@%SPARK_IP%:%SPARK_DEST%
echo Starting transfer via SCP (approx. 3 GB, this may take a few minutes)...
echo.

REM Create destination directory on Spark first
ssh %SPARK_USER%@%SPARK_IP% "mkdir -p %SPARK_DEST%"

REM Copy files and dataset
scp -r * %SPARK_USER%@%SPARK_IP%:%SPARK_DEST%/

echo.
echo ============================================================
echo Transfer complete!
echo Next steps:
echo   1. Connect via SSH:  ssh %SPARK_USER%@%SPARK_IP%
echo   2. Go to directory: cd %SPARK_DEST%
echo   3. Make runnable:   chmod +x train_spark.sh
echo   4. Start training:  ./train_spark.sh
echo ============================================================
pause
