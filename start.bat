@echo off
setlocal EnableExtensions
title Visual Defect Detection - launcher
cd /d "%~dp0"

set "API_PORT=8000"
set "UI_PORT=5173"
set "API_URL=http://localhost:%API_PORT%"
set "UI_URL=http://localhost:%UI_PORT%"

echo.
echo  ==========================================
echo    Visual Defect Detection - starting up
echo  ==========================================
echo.

REM ---- 1. Find a Python environment that has the project installed -------------
set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "%USERPROFILE%\.venvs\vdd\Scripts\python.exe" set "PY=%USERPROFILE%\.venvs\vdd\Scripts\python.exe"
if not defined PY set "PY=python"

"%PY%" -c "import defect_detection, onnxruntime, fastapi" >nul 2>&1
if errorlevel 1 (
    echo  [x] No Python environment with the project installed was found.
    echo      Tried: .venv\  then  %%USERPROFILE%%\.venvs\vdd\  then  python on PATH
    echo.
    echo      Set one up once with:
    echo        python -m venv .venv
    echo        .venv\Scripts\pip install -r requirements\serve.txt
    echo        .venv\Scripts\pip install --no-deps -e .
    goto :fail
)
echo  [ok] Python: %PY%

REM ---- 2. Check the trained model bundle exists ------------------------------------
if not exist "artifacts\model\model.onnx" (
    echo  [x] artifacts\model\model.onnx not found. Train and promote a model first:
    echo        vdd pipeline ^&^& vdd promote
    goto :fail
)
echo  [ok] Model bundle: artifacts\model

REM ---- 3. Backend (reuse a healthy API already on the port, e.g. the Docker one) ---
call :is_ready
if not errorlevel 1 (
    echo  [ok] API already running at %API_URL% - reusing it
) else (
    netstat -ano | findstr /R /C:":%API_PORT% .*LISTENING" >nul
    if not errorlevel 1 (
        echo  [x] Port %API_PORT% is used by another program that is not this API.
        echo      Free the port, e.g.  docker rm -f vdd-api
        goto :fail
    )
    echo  [..] Starting API on %API_URL%
    start "VDD API (close to stop)" cmd /k "set VDD_JSON_LOGS=false&& "%PY%" -m defect_detection serve --host 127.0.0.1 --port %API_PORT%"
)

REM ---- 4. Frontend ---------------------------------------------------------------------
netstat -ano | findstr /R /C:":%UI_PORT% .*LISTENING" >nul
if not errorlevel 1 (
    echo  [ok] Something is already serving on port %UI_PORT% - assuming it is the UI
) else (
    echo  [..] Starting UI on %UI_URL%
    start "VDD UI (close to stop)" cmd /k ""%PY%" scripts\serve_frontend.py --port %UI_PORT%"
)

REM ---- 5. Wait for the API, then open the browser ------------------------------------
echo  [..] Waiting for the API to load the model
for /L %%i in (1,1,60) do (
    call :is_ready
    if not errorlevel 1 goto :ready
    timeout /t 1 /nobreak >nul
)
echo  [x] The API did not become ready within 60 s. Check the "VDD API" window for errors.
goto :fail

:ready
echo  [ok] API ready
start "" "%UI_URL%"
echo.
echo  UI:        %UI_URL%
echo  API docs:  %API_URL%/docs
echo.
echo  Close the "VDD API" and "VDD UI" windows (or run stop.bat) to shut down.
echo.
timeout /t 8 >nul
exit /b 0

:is_ready
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 '%API_URL%/health/ready'; exit [int]($r.StatusCode -ne 200) } catch { exit 1 }" >nul 2>&1
exit /b %errorlevel%

:fail
echo.
pause
exit /b 1
