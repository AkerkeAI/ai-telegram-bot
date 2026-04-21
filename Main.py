import os
import csv
import asyncio
import numpy as np
import librosa
import tensorflow_hub as hub

from flask import Flask, request
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# ================== CONFIG ==================
TOKEN = os.getenv("TOKEN")

if not TOKEN:
    raise ValueError("TOKEN не найден!")

app = Flask(__name__)

# ================== MODEL (ленивая загрузка) ==================
model = None
class_names = None

def load_model():
    global model, class_names

    if model is None:
        print("Загружаю модель...")
        model = hub.load('https://tfhub.dev/google/yamnet/1')

        class_map_path = model.class_map_path().numpy()
        class_names = []

        with open(class_map_path, encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                class_names.append(row[2])

        print("Модель готова")

# ================== AUDIO ==================
def analyze_audio(file_path):
    load_model()

    waveform, sr = librosa.load(file_path, sr=16000)
    scores, _, _ = model(waveform)
    scores_np = scores.numpy()

    mean_scores = np.mean(scores_np, axis=0)
    top_index = np.argmax(mean_scores)

    sound = class_names[top_index]
    confidence = float(mean_scores[top_index])

    return sound, confidence

IMPORTANT_SOUNDS = [
    "siren", "alarm", "gunshot", "explosion",
    "emergency vehicle", "fire alarm", "car horn",
    "dog", "bark"  # 🐶 добавили лай собак
]

# ================== TELEGRAM ==================
telegram_app = ApplicationBuilder().token(TOKEN).build()

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🎧 Анализирую звук...")

    try:
        audio = update.message.voice or update.message.audio

        if not audio:
            await update.message.reply_text("❌ Отправь аудио")
            return

        file = await audio.get_file()
        file_path = "audio.ogg"
        await file.download_to_drive(file_path)

        sound, confidence = analyze_audio(file_path)

        if confidence < 0.4:
            await update.message.reply_text("🔍 Не уверен в звуке")
            return

        if any(word in sound.lower() for word in IMPORTANT_SOUNDS):
            text = f"🚨 ОПАСНО: {sound} ({confidence:.2f})"
        else:
            text = f"🔍 Не опасно: {sound} ({confidence:.2f})"

        await update.message.reply_text(text)

    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")

telegram_app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

# ================== ROUTES ==================
@app.route("/")
def home():
    return "AIQYN бот работает ✅"

@app.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), telegram_app.bot)
    asyncio.run(telegram_app.process_update(update))
    return "ok"

# ================== START ==================
if __name__ == "__main__":
    asyncio.run(telegram_app.initialize())  # ✅ исправлено
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)