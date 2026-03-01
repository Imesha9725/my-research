@echo off
REM Edit the path below to YOUR IEMOCAP folder, then run this script.
REM Example: set IEMOCAP_PATH=D:\Datasets\iemocap-emotion-speech-database

set IEMOCAP_PATH=D:\My Research Project\IEMOCAP_full_release

cd /d "%~dp0.."
echo Installing Python dependencies...
cd ml
pip install -r requirements.txt
cd ..
echo.
echo Training model with IEMOCAP at: %IEMOCAP_PATH%
python ml/train_ser.py
echo.
if exist ml\models\ser_model.joblib (
  echo SUCCESS. Model saved in ml\models\
  echo Next: start Emotion API and Node backend - see HOW-TO-RUN.md
) else (
  echo Training may have failed. Check the path and that Session1, Session2 exist.
)
pause
