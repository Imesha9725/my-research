@echo off
REM Starts Emotion API (port 5002) and Node backend (port 5001) in two new windows.
REM Run this from project root, or double-click. Then run "npm start" for the React app.

cd /d "%~dp0.."

echo Starting Emotion API in new window (port 5002)...
start "Emotion API" cmd /k "python -m uvicorn ml.app:app --reload --port 5002"

timeout /t 3 /nobreak >nul

echo Starting Node backend in new window (port 5001)...
start "Node Backend" cmd /k "cd /d %cd%\server && npm start"

echo.
echo Two windows opened: Emotion API and Node backend.
echo Now run in THIS window:  npm start   (to open the React app at http://localhost:3000)
echo.
pause
