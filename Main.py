import numpy as np
import tensorflow_hub as hub
import librosa
import csv

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

TOKEN = "8717894279:AAFQK2R6ZbW-kMofRU5iM88HiH6snUhTvFU"

print("Загружаю модель...")
model = hub.load('https://tfhub.dev/google/yamnet/1')

# ✅ правильная загрузка классов
class_map_path = model.class_map_path().numpy()

class_names = []
with open(class_map_path, encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        class_names.append(row[2])

print("Модель готова")


# 🎧 анализ
def analyze_audio(file_path):
    waveform, sr = librosa.load(file_path, sr=16000)

    scores, _, _ = model(waveform)
    scores_np = scores.numpy()

    mean_scores = np.mean(scores_np, axis=0)
    top_index = np.argmax(mean_scores)

    sound = class_names[top_index]
    confidence = float(mean_scores[top_index])

    return sound, confidence


# 🚨 важные звуки
IMPORTANT_SOUNDS = [
    "siren",
    "alarm",
    "gunshot",
    "explosion",
    "emergency vehicle",
    "fire alarm",
    "car horn"
]


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🎧 Анализирую звук...")

    try:
        audio = None

        if update.message.voice:
            audio = update.message.voice
        elif update.message.audio:
            audio = update.message.audio
        else:
            await update.message.reply_text("❌ Отправь аудио")
            return

        file = await audio.get_file()
        file_path = "audio.ogg"
        await file.download_to_drive(file_path)

        sound, confidence = analyze_audio(file_path)

        if confidence < 0.4:
            await update.message.reply_text("🔍 Звук не распознан уверенно")
            return

        IMPORTANT_SOUNDS = [
            "siren", "alarm", "gunshot", "explosion",
            "emergency vehicle", "fire alarm", "car horn"
        ]

        if any(word in sound.lower() for word in IMPORTANT_SOUNDS):
            text = f"🚨 ОПАСНО: {sound} ({confidence:.2f})"
        else:
            text = f"🔍 Не опасно: {sound} ({confidence:.2f})"

        await update.message.reply_text(text)

    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")

        file = await audio.get_file()
        file_path = "audio.ogg"
        await file.download_to_drive(file_path)

        sound, confidence = analyze_audio(file_path)

        if confidence < 0.4:
            await update.message.reply_text("🔍 Звук не распознан уверенно")
            return

        sound_lower = sound.lower()

        if any(word in sound_lower for word in IMPORTANT_SOUNDS):
            text = f"🚨 ОПАСНО: {sound} ({confidence:.2f})"
        else:
            text = f"🔍 Не опасно: {sound} ({confidence:.2f})"

        await update.message.reply_text(text)

    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")


def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

    print("Бот запущен...")
    app.run_polling()


if __name__ == "__main__":
    main()

    import os
TOKEN = os.getenv("TOKEN")