# setup.ps1 — ПОЛНОСТЬЮ ИСПРАВЛЕННЫЙ, С ЛОГАМИ И БЕЗ ОШИБОК
Write-Host "🚀 Установка окружения для Speech Recognition с Diarization (NVIDIA NeMo) в venv..." -ForegroundColor Cyan
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Проверка Python
Write-Host "🔍 Поиск Python..." -ForegroundColor Yellow
$python = Get-Command python -ErrorAction SilentlyContinue
if (!$python) {
    Write-Error "❌ Python не найден! Установите Python 3.10–3.12 с https://python.org и отметьте 'Add to PATH'"
    exit 1
}
Write-Host "✅ Найден Python: $($python.Source)" -ForegroundColor Green

# Удаление старого venv
if (Test-Path "venv") {
    Write-Host "🗑️ Удаление старого venv..." -ForegroundColor Gray
    Remove-Item -Recurse -Force venv
}

# Создание venv
Write-Host "📦 Создание venv..." -ForegroundColor Yellow
python -m venv venv
Write-Host "✅ venv создан" -ForegroundColor Green

# Активация
$venvActivate = "venv\Scripts\Activate.ps1"
& $venvActivate
Write-Host "⚡ venv активирован" -ForegroundColor Green

# Обновление pip
Write-Host "🔄 Обновление pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Установка cuda-python
Write-Host "🔧 Установка cuda-python..." -ForegroundColor Yellow
pip install cuda-python>=12.3
Write-Host "✅ cuda-python установлен" -ForegroundColor Green

# Установка PyTorch
Write-Host "🔧 Установка PyTorch с CUDA 12.1..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Установка nemo-toolkit
Write-Host "🔧 Установка nemo-toolkit..." -ForegroundColor Yellow
pip install nemo-toolkit>=2.4.0 --upgrade

# Установка зависимостей
Write-Host "📥 Установка зависимостей из requirements.txt..." -ForegroundColor Yellow
Write-Host "   (Лог установки будет отображаться ниже)" -ForegroundColor DarkGray
pip install -r requirements.txt

# Проверка установки ключевых пакетов
Write-Host "`n🔍 Проверка установки nemo-toolkit, soundfile, hydra-core и cuda-python..." -ForegroundColor Yellow
$has_nemo = pip list | Select-String "nemo-toolkit"
$has_soundfile = pip list | Select-String "soundfile"
$has_hydra = pip list | Select-String "hydra-core"
$has_cuda_python = pip list | Select-String "cuda-python"

if ($has_nemo) {
    Write-Host "✅ nemo-toolkit установлен" -ForegroundColor Green
    $nemo_version = pip show nemo-toolkit | Select-String "Version"
    Write-Host "📌 Версия nemo-toolkit: $nemo_version" -ForegroundColor Green
} else {
    Write-Error "❌ nemo-toolkit НЕ установлен! Проверьте requirements.txt"
}

if ($has_soundfile) {
    Write-Host "✅ soundfile установлен" -ForegroundColor Green
} else {
    Write-Error "❌ soundfile НЕ установлен! Проверьте requirements.txt"
}

if ($has_hydra) {
    Write-Host "✅ hydra-core установлен" -ForegroundColor Green
} else {
    Write-Error "❌ hydra-core НЕ установлен! Убедитесь, что в requirements.txt указано 'hydra-core'"
}

if ($has_cuda_python) {
    Write-Host "✅ cuda-python установлен" -ForegroundColor Green
} else {
    Write-Error "❌ cuda-python НЕ установлен! Убедитесь, что установка прошла успешно"
}

# Папки
New-Item -ItemType Directory -Path "audio", "results" -Force | Out-Null

# GPU
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    Write-Host "✅ NVIDIA GPU обнаружена:" -ForegroundColor Green
    & nvidia-smi
} else {
    Write-Warning "⚠️ NVIDIA GPU не найдена"
}

# Проверка CUDA в PyTorch
Write-Host "`n🧪 Проверка CUDA в PyTorch..." -ForegroundColor Yellow
python -c "
import torch
print('✅ PyTorch версия:', torch.__version__)
print('✅ CUDA доступна:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU:', torch.cuda.get_device_name(0))
"

Write-Host "Устанавливаем значение PYTORCH_CUDA_ALLOC_CONF = expandable_segments:True" -ForegroundColor Yellow
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

Write-Host "`n🎉 Установка завершена!" -ForegroundColor Green
Write-Host "👉 Убедитесь, что файл diarizer_config.yaml существует в корне проекта"
Write-Host "👉 Запустите веб сервер: python main.py" -ForegroundColor Cyan
Write-Host "👉 1: python main.py" -ForegroundColor Cyan
Write-Host "👉 2: через start-web-server.bat" -ForegroundColor Cyan