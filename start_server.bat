@echo off
echo ========================================
echo   Housing Buddy Server Starting...
echo ========================================
echo.
echo Your app will be available at:
echo   http://localhost:9000/
echo.
echo Press Ctrl+C to stop the server.
echo ========================================

cd /d "%~dp0"
python api/main.py

pause
