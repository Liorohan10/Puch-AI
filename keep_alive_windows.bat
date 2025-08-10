@echo off
REM Keep-Alive Script for Render Service
REM Run this script on any Windows machine to keep your service active

setlocal EnableDelayedExpansion

set "TARGET_URL=https://puch-ai-ssnl.onrender.com"
set "PING_INTERVAL=600"
set "LOG_FILE=keep_alive_log.txt"

echo Starting Keep-Alive for %TARGET_URL%
echo Ping interval: %PING_INTERVAL% seconds (10 minutes)
echo Log file: %LOG_FILE%
echo.

:LOOP
    set "TIMESTAMP=%DATE% %TIME%"
    echo [!TIMESTAMP!] Pinging server... | tee -a %LOG_FILE%
    
    REM Try health check endpoint first
    curl -s -o nul -w "Health check status: %%{http_code}" --max-time 30 --retry 2 "%TARGET_URL%/health"
    
    if !ERRORLEVEL! == 0 (
        echo [!TIMESTAMP!] Ping successful | tee -a %LOG_FILE%
    ) else (
        echo [!TIMESTAMP!] Health check failed, trying root endpoint... | tee -a %LOG_FILE%
        curl -s -o nul -w "Root status: %%{http_code}" --max-time 30 --retry 2 "%TARGET_URL%"
        
        if !ERRORLEVEL! == 0 (
            echo [!TIMESTAMP!] Fallback ping successful | tee -a %LOG_FILE%
        ) else (
            echo [!TIMESTAMP!] All pings failed | tee -a %LOG_FILE%
        )
    )
    
    echo [!TIMESTAMP!] Waiting %PING_INTERVAL% seconds until next ping... | tee -a %LOG_FILE%
    timeout /t %PING_INTERVAL% /nobreak >nul
    
goto LOOP
