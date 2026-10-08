@echo off
REM Stops the API and UI windows opened by start.bat (does not touch Docker containers).
taskkill /FI "WINDOWTITLE eq VDD API*" /T /F >nul 2>&1 && echo Stopped API. || echo API was not running.
taskkill /FI "WINDOWTITLE eq VDD UI*" /T /F >nul 2>&1 && echo Stopped UI. || echo UI was not running.
timeout /t 3 >nul
