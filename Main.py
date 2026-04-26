import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import csv

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# ===== TOKEN =====
TOKEN = os.getenv("TOKEN")

if not TOKEN:
    raise ValueError("❌ TOKEN не найден в Railway Variables")

# ===== MODEL LOAD (с защитой) =====
print("Загружаю модель...")
try:
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    MODEL_OK = True
    print("✅ Модель загружена")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    MODEL_OK = False

# ===== CLASS MAP (локальный файл!) =====
class_names = []

try:
    with open("yamnet_class_map.csv", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            class_names.append(row[2])
    print("✅ Классы загружены")
except Exception as e:
    print(f"❌ Ошибка class_map: {e}")
    class_names = []

# ===== SOUND RULES =====
IMPORTANT_SOUNDS = [
    "siren", "alarm", "gunshot", "explosion",
    "emergency vehicle", "fire alarm",
    "car horn", "dog bark"
]

# ===== ANALYSIS =====
def analyze_audio(file_path):
    try:
        waveform, sr = librosa.load(file_path, sr=16000)
        waveform = waveform.astype(np.float32)

        scores, _, _ = model(waveform)
        scores_np = scores.numpy()

        mean_scores = np.mean(scores_np, axis=0)
        top_index = int(np.argmax(mean_scores))

        sound = class_names[top_index] if class_names else "unknown"
        confidence = float(mean_scores[top_index])

        return sound, confidence

    except Exception as e:
        return f"error: {e}", 0.0


# ===== HANDLER =====
async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🎧 Анализирую...")

    try:
        audio = update.message.voice or update.message.audio

        if not audio:
            await update.message.reply_text("❌ Отправь голосовое или аудио")
            return

        file = await audio.get_file()
        file_path = "audio.ogg"

        await file.download_to_drive(file_path)

        # если модель не загрузилась
        if not MODEL_OK:
            await update.message.reply_text("⚠️ Модель не загрузилась")
            return

        sound, confidence = analyze_audio(file_path)

        if "error" in sound:
            await update.message.reply_text(f"⚠️ Ошибка анализа: {sound}")
            return

        if confidence < 0.4:
            await update.message.reply_text("🔍 Не уверен в распознавании")
            return

        is_danger = any(x in sound.lower() for x in IMPORTANT_SOUNDS)

        if is_danger:
            text = f"🚨 ОПАСНО: {sound} ({confidence:.2f})"
        else:
            text = f"🟢 Не опасно: {sound} ({confidence:.2f})"

        await update.message.reply_text(text)

    except Exception as e:
        await update.message.reply_text(f"❌ Критическая ошибка: {e}")


# ===== MAIN =====
def main():
    try:
        app = ApplicationBuilder().token(TOKEN).build()

        app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

        print("🚀 Бот запущен")
        app.run_polling()

    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")


if __name__ == "__main__":
    main()