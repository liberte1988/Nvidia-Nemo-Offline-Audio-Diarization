#!/usr/bin/env python3
import os
import sys
import torch
import soundfile as sf
from pydub import AudioSegment
from pathlib import Path
from nemo.collections.asr.models import EncDecRNNTBPEModel, EncDecCTCModelBPE
from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf
import logging
import argparse

# Настройка кодировки консоли для Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

AUDIO_DIR = Path("audio")
RESULTS_DIR = Path("results")
AUDIO_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    logger.warning("⚠️ GPU не обнаружен! Работа будет происходить на CPU — это очень медленно!")
else:
    logger.info(f"✅ Используется GPU: {torch.cuda.get_device_name(0)}")

def load_asr_model(language: str, model_type: str, model_name: str) -> object:
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
    transducer_models_ru = ["stt_ru_conformer_transducer_large"]
    ctc_models_ru = ["stt_ru_conformer_ctc_large"]

    if language == "ru":
        if model_type == "transducer":
            if model_name not in transducer_models_ru:
                raise ValueError(f"Модель '{model_name}' не поддерживается для ru (transducer). Доступно: {transducer_models_ru}")
            model_class = EncDecRNNTBPEModel
        elif model_type == "ctc":
            if model_name not in ctc_models_ru:
                raise ValueError(f"Модель '{model_name}' не поддерживается для ru (ctc). Доступно: {ctc_models_ru}")
            model_class = EncDecCTCModelBPE
        else:
            raise ValueError(f"Тип модели '{model_type}' не поддерживается. Используйте 'transducer' или 'ctc'.")
    elif language == "en":
        if model_type == "transducer":
            if model_name not in transducer_models_en:
                raise ValueError(f"Модель '{model_name}' не поддерживается для en (transducer). Доступно: {transducer_models_en}")
            model_class = EncDecRNNTBPEModel
        elif model_type == "ctc":
            if model_name not in ctc_models_en:
                raise ValueError(f"Модель '{model_name}' не поддерживается для en (ctc). Доступно: {ctc_models_en}")
            model_class = EncDecCTCModelBPE
        else:
            raise ValueError(f"Тип модели '{model_type}' не поддерживается. Используйте 'transducer' или 'ctc'.")
    else:
        raise ValueError(f"Язык '{language}' не поддерживается. Используйте 'ru' или 'en'.")

    logger.info(f"📥 Загрузка модели ASR ({language}, {model_type}, {model_name})...")
    model = model_class.from_pretrained(model_name=model_name)
    model = model.to(device)
    model.eval()
    return model

def convert_to_wav(audio_path: Path) -> Path:
    if audio_path.suffix.lower() != ".wav":
        logger.info(f">> Конвертация {audio_path.name} в WAV...")
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        wav_path = audio_path.with_suffix(".wav")
        audio.export(wav_path, format="wav")
        logger.info(f"✅ Сохранено: {wav_path.name}")
        audio_path.unlink()
        return wav_path
    return audio_path

def transcribe_audio(audio_path: Path, asr_model) -> str:
    logger.info(f"🎙️ Распознавание речи для {audio_path.name}...")
    audio_data, sample_rate = sf.read(audio_path)
    if sample_rate != 16000:
        raise ValueError(f"Аудио должно быть 16kHz, а не {sample_rate}Hz")

    audio_tensor = torch.tensor(audio_data, dtype=torch.float32).to(device)
    with torch.no_grad():
        hypotheses = asr_model.transcribe([audio_tensor], batch_size=1)
    text = hypotheses[0].text if hasattr(hypotheses[0], 'text') else hypotheses[0]
    logger.info(f"📝 Распознано: {text[:100]}...")
    torch.cuda.empty_cache()
    return text

def diarize_and_transcribe(audio_path: Path, asr_model) -> str:
    logger.info(f"🗣️ Диаризация и распознавание для {audio_path.name}...")

    temp_dir = RESULTS_DIR / "temp_diarization"
    temp_dir.mkdir(exist_ok=True)

    manifest_path = temp_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        import json
        entry = {
            "audio_filepath": str(audio_path.resolve()),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "rttm_filepath": "",
            "uem_filepath": ""
        }
        f.write(json.dumps(entry) + "\n")

    config_path = Path("diarizer_config.yaml")
    if not config_path.exists():
        logger.error(f"❌ Файл конфигурации {config_path} не найден!")
        raise FileNotFoundError(f"Файл {config_path} необходим для диаризации")
    
    logger.info(f"📄 Загрузка конфигурации из {config_path}...")
    config = OmegaConf.load(config_path)
    logger.info(f"🔍 Конфигурация VAD: {config.diarizer.vad}")

    config.diarizer.manifest_filepath = str(manifest_path)
    config.diarizer.out_dir = str(temp_dir)

    try:
        diarizer = ClusteringDiarizer(cfg=config)
        diarizer.diarize()
    except Exception as e:
        logger.error(f"❌ Ошибка диаризации: {e}")
        raise

    rttm_file = temp_dir / "pred_rttms" / (audio_path.stem + ".rttm")
    if not rttm_file.exists():
        raise FileNotFoundError(f"RTTM файл не создан: {rttm_file}")

    segments = []
    with open(rttm_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            start = float(parts[3])
            duration = float(parts[4])
            end = start + duration
            speaker = parts[7]
            segments.append((start, end, speaker))

    audio_data, sr = sf.read(audio_path)
    if sr != 16000:
        raise ValueError("Аудио должно быть 16kHz")

    full_text = []
    for start, end, speaker in segments:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        if start_sample >= len(audio_data):
            continue
        segment_audio = audio_data[start_sample:end_sample]
        if len(segment_audio) == 0:
            continue
        segment_tensor = torch.tensor(segment_audio, dtype=torch.float32).to(device)

        with torch.no_grad():
            hyp = asr_model.transcribe([segment_tensor], batch_size=1)
        text = hyp[0].text if hasattr(hyp[0], 'text') else hyp[0]
        if text:
            full_text.append(f"[{speaker}] {start:.2f}-{end:.2f}s: {text}")
        torch.cuda.empty_cache()

    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    return "\n".join(full_text)

def main():
    parser = argparse.ArgumentParser(description="Speech Recognition with optional Diarization")
    parser.add_argument("--diarization", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--language", choices=["ru", "en"], default="ru", help="Language: ru (default) or en")
    parser.add_argument("--transducer", action="store_true", help="Use transducer model (most accurate, heavy)")
    parser.add_argument("--ctc", action="store_true", help="Use CTC model (lighter, faster)")
    parser.add_argument("--model_name", required=True, help="Specific model name to use (e.g., stt_en_conformer_ctc_large)")
    args = parser.parse_args()

    if args.transducer and args.ctc:
        raise ValueError("Можно выбрать только один тип модели: --transducer или --ctc")
    if not args.transducer and not args.ctc:
        raise ValueError("Необходимо выбрать тип модели: --transducer или --ctc")

    model_type = "transducer" if args.transducer else "ctc"

    asr_model = load_asr_model(args.language, model_type, args.model_name)
    use_diarization = args.diarization
    mode_name = "ASR + Диаризация" if use_diarization else "Только ASR"
    logger.info(f">> Режим: {mode_name} | Язык: {args.language} | Тип модели: {model_type} | Модель: {args.model_name}")

    audio_files = [f for f in AUDIO_DIR.iterdir() if f.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg")]

    if not audio_files:
        logger.warning("❌ В папке audio нет аудиофайлов!")
        return

    for audio_file in audio_files:
        print(f"\n>> Обработка: {audio_file.name}")

        wav_path = convert_to_wav(audio_file)
        output_file = RESULTS_DIR / (wav_path.stem + ".txt")

        try:
            simple_asr = transcribe_audio(wav_path, asr_model)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(simple_asr)
            print(f"✅ Сохранено: {output_file.name}")

            if use_diarization:
                diar_result = diarize_and_transcribe(wav_path, asr_model)
                diar_output_file = RESULTS_DIR / (wav_path.stem + "-diarization.txt")
                with open(diar_output_file, "w", encoding="utf-8") as f:
                    f.write(diar_result)
                print(f"✅ Сохранено: {diar_output_file.name}")

        except Exception as e:
            logger.error(f"❌ Ошибка при обработке {audio_file.name}: {e}")

        torch.cuda.empty_cache()

    print(f"\n🎉 Все файлы обработаны! Результаты в папке '{RESULTS_DIR}'")

if __name__ == "__main__":
    main()