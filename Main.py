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
    raise ValueError("TOKEN не найден в Railway Variables")

# ===== MODEL =====
print("Загружаю модель...")
model = hub.load("https://tfhub.dev/google/yamnet/1")
print("Модель готова")

# ===== CLASS NAMES FIX =====
class_map_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://githubusercontent.com')
class_names = []
with open(class_map_path, encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        class_names.append(row[2])


IMPORTANT_SOUNDS = [
    "siren",
    "alarm",
    "gunshot",
    "explosion",
    "emergency vehicle",
    "fire alarm",
    "car horn",
    "dog bark"
]


# ===== AUDIO ANALYSIS =====
def analyze_audio(file_path):
    waveform, sr = librosa.load(file_path, sr=16000)

    waveform = waveform.astype(np.float32)

    scores, embeddings, spectrogram = model(waveform)

    scores_np = scores.numpy()
    mean_scores = np.mean(scores_np, axis=0)

    top_index = np.argmax(mean_scores)

    sound = class_names[top_index]
    confidence = float(mean_scores[top_index])

    return sound, confidence


# ===== HANDLER =====
async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🎧 Анализирую звук...")

    try:
        audio = update.message.voice or update.message.audio

        if not audio:
            await update.message.reply_text("Отправь голосовое или аудио")
            return

        file = await audio.get_file()
        file_path = "audio.wav"

        await file.download_to_drive(file_path)

        sound, confidence = analyze_audio(file_path)

        if confidence < 0.4:
            await update.message.reply_text("🔍 Не уверен в распознавании")
            return

        is_danger = any(x in sound.lower() for x in IMPORTANT_SOUNDS)

        if is_danger:
            text = f"🚨 ОПАСНО: {sound} ({confidence:.2f})"
        else:
            text = f"🟢 Нормально: {sound} ({confidence:.2f})"

        await update.message.reply_text(text)

    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")


# ===== MAIN =====
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

    print("Бот запущен...")
    app.run_polling()


if __name__ == "__main__":
    main()