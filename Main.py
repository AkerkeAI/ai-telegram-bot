import os
import numpy as np
import tensorflow_hub as hub
import librosa
import csv
from flask import Flask, request
import telegram

TOKEN = os.getenv("TOKEN")
URL = os.getenv("URL")  # Railway public URL

if not TOKEN:
    raise ValueError("TOKEN не найден")

if not URL:
    raise ValueError("URL не найден (Railway domain)")

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
    "siren", "alarm", "gunshot", "explosion",
    "fire", "car horn", "dog bark"
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


bot = telegram.Bot(token=TOKEN)
app = Flask(__name__)


@app.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    update = telegram.Update.de_json(request.get_json(force=True), bot)

    if update.message and (update.message.voice or update.message.audio):

        msg = update.message
        audio = msg.voice or msg.audio

        file = bot.get_file(audio.file_id)
        file_path = "audio.ogg"
        file.download_to_drive(file_path)

        sound, confidence = analyze_audio(file_path)

        if confidence < 0.4:
            msg.reply_text("🔍 Не удалось точно распознать звук")
            return "ok"

        if any(x in sound.lower() for x in IMPORTANT_SOUNDS):
            text = f"🚨 ОПАСНО: {sound} ({confidence:.2f})"
        else:
            text = f"🟢 Безопасно: {sound} ({confidence:.2f})"

        msg.reply_text(text)

    return "ok"


@app.route("/", methods=["GET"])
def index():
    return "Bot is running"


if __name__ == "__main__":
    bot.set_webhook(url=f"{URL}/{TOKEN}")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))