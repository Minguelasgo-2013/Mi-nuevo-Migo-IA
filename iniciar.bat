@echo off
title Mi Gemini - Interfaz de Chat
cls
echo ==================================================
echo 🤖 INICIANDO LA INTERFAZ WEB DE TU IA
echo ==================================================
echo.
streamlit run app.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ Ocurrio un error al intentar iniciar la interfaz.
    pause
    exit
)
pause