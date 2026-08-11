@echo off
title Entrenador de IA - Gemini Desde Cero
cls
echo ==================================================
echo 🧠 INICIANDO EL COLEGIO DE TU IA (MODO ENTRENAMIENTO)
echo ==================================================
echo.
python main.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ Ocurrio un error al intentar entrenar.
    pause
    exit
)
pause
