#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
from pathlib import Path
from flask import Flask, request, render_template_string, send_from_directory, flash, redirect, url_for, jsonify
import logging

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Для флеш-сообщений

# Отключение логов Werkzeug уровня INFO
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

AUDIO_DIR = Path("audio")
RESULTS_DIR = Path("results")
TEMPLATES_DIR = Path("templates")
AUDIO_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# HTML шаблоны с Tailwind CSS (без изменений)
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Transcription</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        function updateModels() {
            var language = document.getElementById("language").value;
            var modelSelect = document.getElementById("model_name");
            modelSelect.innerHTML = "";
            var models = [];
            if (language === "ru") {
                models = [
                    {value: "stt_ru_conformer_transducer_large", text: "Transducer: stt_ru_conformer_transducer_large"},
                    {value: "stt_ru_conformer_ctc_large", text: "CTC: stt_ru_conformer_ctc_large"}
                ];
            } else {
                models = [
                    {% for model in transducer_models_en %}
                        {value: "{{ model }}", text: "Transducer: {{ model }}"},
                    {% endfor %}
                    {% for model in ctc_models_en %}
                        {value: "{{ model }}", text: "CTC: {{ model }}"},
                    {% endfor %}
                ];
            }
            models.forEach(function(model) {
                var option = document.createElement("option");
                option.value = model.value;
                option.text = model.text;
                modelSelect.appendChild(option);
            });
            updateModelType();
        }
        function updateModelType() {
            var modelName = document.getElementById("model_name").value;
            var modelType = modelName.includes("conformer_transducer") || 
                            modelName.includes("contextnet") || 
                            modelName.includes("fastconformer_transducer") ? "transducer" : "ctc";
            document.getElementById("model_type").value = modelType;
        }
        async function uploadFiles() {
            var files = document.getElementById("audio_files").files;
            if (files.length === 0) {
                showToast("Выберите хотя бы один аудиофайл!", "error");
                return;
            }
            var formData = new FormData();
            for (var i = 0; i < files.length; i++) {
                formData.append("audio_files", files[i]);
            }
            try {
                let response = await fetch("/upload", {
                    method: "POST",
                    body: formData
                });
                let result = await response.json();
                if (result.success) {
                    let fileList = document.getElementById("file_list");
                    fileList.innerHTML = "";
                    result.files.forEach(file => {
                        let li = document.createElement("li");
                        li.className = "flex items-center text-gray-700";
                        li.innerHTML = `<svg class="w-5 h-5 mr-2 text-blue-500" fill="currentColor" viewBox="0 0 20 20"><path d="M10 2a6 6 0 00-6 6v3.586l-.707.707A1 1 0 004 14h12a1 1 0 00.707-1.707L16 11.586V8a6 6 0 00-6-6z"></path></svg>${file}`;
                        fileList.appendChild(li);
                    });
                    document.getElementById("process_button").disabled = false;
                    showToast("Файлы успешно загружены!", "success");
                } else {
                    showToast("Ошибка: " + result.message, "error");
                }
            } catch (error) {
                showToast("Ошибка при загрузке: " + error.message, "error");
            }
        }
        async function processFiles() {
            var button = document.getElementById("process_button");
            button.disabled = true;
            button.textContent = "Обработка...";
            var loader = document.getElementById("loader");
            loader.classList.remove("hidden");
            var formData = new FormData(document.getElementById("process_form"));
            try {
                let response = await fetch("/process", {
                    method: "POST",
                    body: formData
                });
                let result = await response.json();
                if (result.success) {
                    let resultList = document.getElementById("result_list");
                    resultList.innerHTML = "";
                    result.files.forEach(file => {
                        let li = document.createElement("li");
                        li.className = "flex items-center";
                        li.innerHTML = `<svg class="w-5 h-5 mr-2 text-green-500" fill="currentColor" viewBox="0 0 20 20"><path d="M6 2a2 2 0 00-2 2v12a2 2 0 002 2h8a2 2 0 002-2V7.414A2 2 0 0015.414 6L12 2.586A2 2 0 0010.586 2H6z"></path></svg><a href="/download/${file}" class="text-blue-600 hover:underline">${file}</a>`;
                        resultList.appendChild(li);
                    });
                    document.getElementById("results_section").classList.remove("hidden");
                    showToast("Обработка завершена!", "success");
                } else {
                    showToast("Ошибка: " + result.message, "error");
                }
            } catch (error) {
                showToast("Ошибка при обработке: " + error.message, "error");
            } finally {
                button.disabled = false;
                button.textContent = "Обработать";
                loader.classList.add("hidden");
            }
        }
        function showToast(message, type) {
            var toast = document.getElementById("toast");
            toast.textContent = message;
            toast.className = `fixed top-4 right-4 p-4 rounded-md shadow-lg transition-opacity duration-300 ${type === "success" ? "bg-green-500" : "bg-red-500"} text-white`;
            toast.style.opacity = "1";
            setTimeout(() => {
                toast.style.opacity = "0";
                setTimeout(() => toast.className = "hidden", 300);
            }, 3000);
        }
        window.onload = function() {
            updateModels();
        };
    </script>
</head>
<body class="bg-gradient-to-br from-blue-100 to-gray-100 min-h-screen flex items-center justify-center">
    <div class="container max-w-3xl mx-auto p-8 bg-white rounded-2xl shadow-xl">
        <h1 class="text-4xl font-bold text-center text-gray-800 mb-8">Audio Transcription</h1>
        <div id="toast" class="hidden"></div>
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                    <p class="text-red-500 text-center font-medium mb-4">{{ message }}</p>
                {% endfor %}
            {% endif %}
        {% endwith %}
        <form id="upload_form" enctype="multipart/form-data">
            <div class="mb-6">
                <label for="audio_files" class="block text-gray-700 font-semibold mb-2">Выберите аудиофайлы:</label>
                <input type="file" id="audio_files" name="audio_files" multiple accept=".wav,.mp3,.flac,.ogg" class="mt-1 block w-full border-gray-300 rounded-lg p-2 focus:ring-2 focus:ring-blue-500 transition">
            </div>
            <button type="button" onclick="uploadFiles()" class="w-full bg-blue-600 text-white font-semibold py-3 px-4 rounded-lg hover:bg-blue-700 transition flex items-center justify-center">
                <svg class="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20"><path d="M10 2.5v11m0 0l-4-4m4 4l4-4m-8 4h8"></path></svg>
                Загрузить аудиофайлы
            </button>
        </form>
        <ul id="file_list" class="mt-6 space-y-2"></ul>
        <form id="process_form" class="mt-6">
            <div class="mb-6">
                <label for="language" class="block text-gray-700 font-semibold mb-2">Язык:</label>
                <select id="language" name="language" onchange="updateModels()" class="mt-1 block w-full border-gray-300 rounded-lg p-2 focus:ring-2 focus:ring-blue-500 transition">
                    <option value="ru" selected>Русский</option>
                    <option value="en">Английский</option>
                </select>
            </div>
            <div class="mb-6">
                <label for="model_name" class="block text-gray-700 font-semibold mb-2">Модель:</label>
                <select id="model_name" name="model_name" onchange="updateModelType()" class="mt-1 block w-full border-gray-300 rounded-lg p-2 focus:ring-2 focus:ring-blue-500 transition">
                    <option value="stt_ru_conformer_transducer_large" selected>Transducer: stt_ru_conformer_transducer_large</option>
                    <option value="stt_ru_conformer_ctc_large">CTC: stt_ru_conformer_ctc_large</option>
                </select>
            </div>
            <input type="hidden" id="model_type" name="model_type" value="transducer">
            <div class="mb-6">
                <label class="inline-flex items-center">
                    <input type="checkbox" name="diarization" class="form-checkbox h-5 w-5 text-blue-600 rounded">
                    <span class="ml-2 text-gray-700 font-medium">Включить диаризацию</span>
                </label>
            </div>
            <button id="process_button" type="button" onclick="processFiles()" disabled class="w-full bg-green-600 text-white font-semibold py-3 px-4 rounded-lg hover:bg-green-700 transition flex items-center justify-center">
                <svg class="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20"><path d="M5 4v12l11-6-11-6z"></path></svg>
                Обработать
            </button>
        </form>
        <div id="loader" class="hidden text-center mt-6">
            <div class="animate-spin rounded-full h-10 w-10 border-t-4 border-blue-600 mx-auto"></div>
            <p class="text-gray-700 mt-3 font-medium">Идёт обработка, ожидайте...</p>
        </div>
        <div id="results_section" class="hidden mt-8">
            <h3 class="text-2xl font-semibold text-gray-700 mb-4">Скачать результаты:</h3>
            <ul id="result_list" class="space-y-3"></ul>
            <a href="{{ url_for('index') }}" class="mt-6 inline-block w-full bg-blue-600 text-white font-semibold py-3 px-4 rounded-lg hover:bg-blue-700 transition text-center">Вернуться</a>
        </div>
    </div>
</body>
</html>
"""

RESULTS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Results</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-blue-100 to-gray-100 min-h-screen flex items-center justify-center">
    <div class="container max-w-3xl mx-auto p-8 bg-white rounded-2xl shadow-xl">
        <h1 class="text-4xl font-bold text-center text-gray-800 mb-8">Results</h1>
        <p class="text-green-600 text-center font-medium mb-6">Файлы успешно обработаны!</p>
        <h3 class="text-2xl font-semibold text-gray-700 mb-4">Скачать результаты:</h3>
        <ul class="space-y-3">
            {% for file in result_files %}
                <li class="flex items-center">
                    <svg class="w-5 h-5 mr-2 text-green-500" fill="currentColor" viewBox="0 0 20 20"><path d="M6 2a2 2 0 00-2 2v12a2 2 0 002 2h8a2 2 0 002-2V7.414A2 2 0 0015.414 6L12 2.586A2 2 0 0010.586 2H6z"></path></svg>
                    <a href="{{ url_for('download_file', filename=file) }}" class="text-blue-600 hover:underline">{{ file }}</a>
                </li>
            {% endfor %}
        </ul>
        <a href="{{ url_for('index') }}" class="mt-6 inline-block w-full bg-blue-600 text-white font-semibold py-3 px-4 rounded-lg hover:bg-blue-700 transition text-center">Вернуться</a>
    </div>
</body>
</html>
"""

# Списки моделей для английского языка
transducer_models_en = [
    "stt_en_conformer_transducer_large", "stt_en_conformer_transducer_large_ls",
    "stt_en_conformer_transducer_medium", "stt_en_conformer_transducer_small",
    "stt_en_conformer_transducer_xlarge", "stt_en_conformer_transducer_xxlarge",
    "stt_en_contextnet_1024", "stt_en_contextnet_1024_mls",
    "stt_en_contextnet_256", "stt_en_contextnet_256_mls",
    "stt_en_contextnet_512", "stt_en_contextnet_512_mls",
    "stt_en_fastconformer_transducer_large", "stt_en_fastconformer_transducer_large_ls",
    "stt_en_fastconformer_transducer_xlarge", "stt_en_fastconformer_transducer_xxlarge",
    "stt_enes_conformer_transducer_large", "stt_enes_conformer_transducer_large_codesw",
    "stt_enes_contextnet_large"
]
ctc_models_en = [
    "stt_en_citrinet_1024", "stt_en_citrinet_1024_gamma_0_25",
    "stt_en_citrinet_256", "stt_en_citrinet_256_gamma_0_25",
    "stt_en_citrinet_512", "stt_en_citrinet_512_gamma_0_25",
    "stt_en_conformer_ctc_large", "stt_en_conformer_ctc_large_ls",
    "stt_en_conformer_ctc_medium", "stt_en_conformer_ctc_medium_ls",
    "stt_en_conformer_ctc_small", "stt_en_conformer_ctc_small_ls",
    "stt_en_conformer_ctc_xlarge", "stt_en_fastconformer_ctc_large",
    "stt_en_fastconformer_ctc_large_ls", "stt_en_fastconformer_ctc_xlarge",
    "stt_en_fastconformer_ctc_xxlarge", "stt_en_squeezeformer_ctc_large_ls",
    "stt_en_squeezeformer_ctc_medium_large_ls", "stt_en_squeezeformer_ctc_medium_ls",
    "stt_en_squeezeformer_ctc_small_ls", "stt_en_squeezeformer_ctc_small_medium_ls",
    "stt_en_squeezeformer_ctc_xsmall_ls", "stt_enes_conformer_ctc_large",
    "stt_enes_conformer_ctc_large_codesw", "stt_fr_no_hyphen_citrinet_1024_gamma_0_25",
    "stt_fr_no_hyphen_conformer_ctc_large"
]

def clear_directory(directory: Path):
    """Очищает указанную директорию."""
    for item in directory.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
    logger.info(f"🧹 Очищена директория: {directory}")

def generate_templates():
    """Генерирует HTML шаблоны в папке templates."""
    index_path = TEMPLATES_DIR / "index.html"
    results_path = TEMPLATES_DIR / "results.html"
    
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(INDEX_HTML)
    logger.info(f"📝 Сгенерирован шаблон: {index_path}")
    
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(RESULTS_HTML)
    logger.info(f"📝 Сгенерирован шаблон: {results_path}")

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, transducer_models_en=transducer_models_en, ctc_models_en=ctc_models_en)

@app.route("/upload", methods=["POST"])
def upload():
    clear_directory(AUDIO_DIR)
    audio_files = request.files.getlist("audio_files")
    if not audio_files or all(f.filename == '' for f in audio_files):
        return jsonify({"success": False, "message": "Не выбраны аудиофайлы"})
    
    uploaded_files = []
    for audio_file in audio_files:
        if audio_file and audio_file.filename:
            file_path = AUDIO_DIR / audio_file.filename
            audio_file.save(file_path)
            logger.info(f"📥 Загружен файл: {file_path}")
            uploaded_files.append(audio_file.filename)
    
    return jsonify({"success": True, "files": uploaded_files})

@app.route("/process", methods=["POST"])
def process():
    clear_directory(RESULTS_DIR)
    
    language = request.form.get("language", "ru")
    model_name = request.form.get("model_name")
    model_type = request.form.get("model_type")
    diarization = "diarization" in request.form

    if not model_name or not model_type:
        return jsonify({"success": False, "message": "Модель или тип модели не выбраны"})

    cmd = [sys.executable, "main.py", "--language", language, "--model_name", model_name]
    if model_type == "transducer":
        cmd.append("--transducer")
    else:
        cmd.append("--ctc")
    if diarization:
        cmd.append("--diarization")

    logger.info(f"🚀 Запуск команды: {' '.join(cmd)}")
    try:
        # Явно указываем кодировку utf-8 для subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'  # Заменяем неподдерживаемые символы
        )
        logger.info(result.stdout)
        # Игнорируем предупреждения NeMo в stderr
        if result.stderr and "[NeMo W" in result.stderr:
            logger.info("Игнорируем предупреждения NeMo в stderr")
        elif result.stderr:
            logger.error(result.stderr)
            return jsonify({"success": False, "message": result.stderr})
    except subprocess.CalledProcessError as e:
        logger.error(e.stderr)
        return jsonify({"success": False, "message": e.stderr})
    except UnicodeDecodeError as e:
        logger.error(f"Ошибка декодирования вывода: {e}")
        return jsonify({"success": False, "message": f"Ошибка декодирования: {e}"})

    result_files = [f.name for f in RESULTS_DIR.iterdir() if f.is_file()]
    if not result_files:
        return jsonify({"success": False, "message": "Результаты не созданы"})

    return jsonify({"success": True, "files": result_files})

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(RESULTS_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    generate_templates()
    logger.info("🌐 Запуск веб-сервера на http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000)