# Run both backends: Emotion API (port 5002) + Node chat server (port 5001)
# Usage: .\scripts\run-backend.ps1
# Or: powershell -File scripts/run-backend.ps1

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $projectRoot

Write-Host "Starting Emotion API (IEMOCAP) on port 5002..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectRoot'; python -m uvicorn ml.app:app --reload --port 5002"

Start-Sleep -Seconds 3
Write-Host "Starting Node backend on port 5001..." -ForegroundColor Cyan
Set-Location server
node index.js
