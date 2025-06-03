@echo off
title Ligoo Signal Bot
color 0A

echo ============================================
echo   🚀 Starting Ligoo Signals Trading Bot
echo ============================================
echo.

REM Optional: activate virtualenv if you use one
REM call venv\Scripts\activate

REM Run the bot
python main.py

echo.
echo ✅ Bot stopped. Press any key to exit.
pause >nul
