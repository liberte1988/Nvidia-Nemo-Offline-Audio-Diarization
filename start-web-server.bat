@echo off
chcp 65001 > nul
SETLOCAL ENABLEDELAYEDEXPANSION
echo 🛠️ Установка и запуск веб-сервера
echo ===================================

REM Активируем виртуальное окружение
call .venv\Scripts\activate

REM Запускаем скрипт установки
python server.py

pause
