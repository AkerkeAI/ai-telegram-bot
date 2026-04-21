import os
import numpy as np
import tensorflow_hub as hub
import librosa
import csv
import tempfile

from flask import Flask, request
from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters

# =========================
# ENV VARIABLES
# =========================
TOKEN = os.getenv("TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # Railway domain + /webhook

if not TOKEN:
    raise ValueError("TOKEN не найден в Railway Variables")

if not WEBHOOK_URL:
    raise ValueError("WEBHOOK_URL не найден в Railway Variables")

# =========================
# FLASK APP (WEB SERVER)
# =========================
app_flask = Flask(__name__)

# =========================
# TELEGRAM BOT
# =========================
print("Загружаю модель...")
model = hub.load("https://tfhub.dev/google/yamnet/1")

class_map_path = model.class_map_path().numpy()
class_names = []

with open(class_map_path, encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        class_names.append(row[2])

print("Модель готова")

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


def analyze_audio(file_path):
    waveform, sr = librosa.load(file_path, sr=16000)

    scores, _, _ = model(waveform)
    scores_np = scores.numpy()

    mean_scores = np.mean(scores_np, axis=0)
    top_index = np.argmax(mean_scores)

    sound = class_names[top_index]
    confidence = float(mean_scores[top_index])

    return sound, confidence


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("🎧 Анализирую звук...")

        audio = update.message.voice or update.message.audio
        if not audio:
            await update.message.reply_text("Отправь голосовое или аудио")
            return

        file = await audio.get_file()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
            file_path = tmp.name
            await file.download_to_drive(file_path)

        sound, confidence = analyze_audio(file_path)

        if confidence < 0.4:
            await update.message.reply_text("🔍 Звук не распознан уверенно")
            return

        is_danger = any(x in sound.lower() for x in IMPORTANT_SOUNDS)

        if is_danger:
            text = f"🚨 ОПАСНО: {sound} ({confidence:.2f})"
        else:
            text = f"🟢 Не опасно: {sound} ({confidence:.2f})"

        await update.message.reply_text(text)

    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")


# =========================
# CREATE APPLICATION
# =========================
telegram_app = Application.builder().token(TOKEN).build()
telegram_app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))


# =========================
# WEBHOOK ROUTE
# =========================
@app_flask.post("/webhook")
async def webhook():
    data = request.get_json(force=True)
    update = Update.de_json(data, telegram_app.bot)

    await telegram_app.process_update(update)
    return "ok"


# =========================
# START SERVER
# =========================
if __name__ == "__main__":
    print("Бот запущен (WEBHOOK MODE)")

    # ставим webhook в Telegram
    import requests
    requests.get(
        f"https://api.telegram.org/bot{TOKEN}/setWebhook?url={WEBHOOK_URL}"
    )

    # Railway сам даст PORT
    port = int(os.environ.get("PORT", 8080))

    app_flask.run(host="0.0.0.0", port=port)