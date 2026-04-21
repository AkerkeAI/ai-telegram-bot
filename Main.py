import os
import numpy as np
import tensorflow_hub as hub
import librosa
import csv
import requests
from flask import Flask, request

TOKEN = os.getenv("TOKEN")
URL = os.getenv("URL")  # https://xxxx.up.railway.app

if not TOKEN:
    raise ValueError("TOKEN не найден")
if not URL:
    raise ValueError("URL не найден")

API = f"https://api.telegram.org/bot{TOKEN}"

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
    "fire", "emergency", "car horn", "dog"
]

app = Flask(__name__)


# ---------------- AI ----------------
def analyze_audio(file_path):
    waveform, sr = librosa.load(file_path, sr=16000)

    scores, _, _ = model(waveform)
    scores_np = scores.numpy()

    mean_scores = np.mean(scores_np, axis=0)
    top_index = np.argmax(mean_scores)

    sound = class_names[top_index]
    confidence = float(mean_scores[top_index])

    return sound, confidence


# ---------------- Telegram helpers ----------------
def send_message(chat_id, text):
    requests.post(f"{API}/sendMessage", json={
        "chat_id": chat_id,
        "text": text
    })


def download_file(file_id):
    r = requests.get(f"{API}/getFile?file_id={file_id}")
    file_path = r.json()["result"]["file_path"]

    file_url = f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"
    audio_data = requests.get(file_url)

    local_path = "audio.ogg"
    with open(local_path, "wb") as f:
        f.write(audio_data.content)

    return local_path


# ---------------- WEBHOOK ----------------
@app.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    data = request.get_json()

    if "message" not in data:
        return "ok"

    msg = data["message"]
    chat_id = msg["chat"]["id"]

    file_id = None

    if "voice" in msg:
        file_id = msg["voice"]["file_id"]
    elif "audio" in msg:
        file_id = msg["audio"]["file_id"]

    if not file_id:
        send_message(chat_id, "Отправь голосовое или аудио")
        return "ok"

    send_message(chat_id, "🎧 Анализирую звук...")

    try:
        path = download_file(file_id)
        sound, confidence = analyze_audio(path)

        if confidence < 0.4:
            send_message(chat_id, "🔍 Не удалось точно распознать звук")
            return "ok"

        is_danger = any(x in sound.lower() for x in IMPORTANT_SOUNDS)

        if is_danger:
            text = f"🚨 ОПАСНО: {sound} ({confidence:.2f})"
        else:
            text = f"🟢 Безопасно: {sound} ({confidence:.2f})"

        send_message(chat_id, text)

    except Exception as e:
        send_message(chat_id, f"Ошибка: {str(e)}")

    return "ok"


# ---------------- HEALTH CHECK ----------------
@app.route("/", methods=["GET"])
def home():
    return "Bot is running"


# ---------------- SET WEBHOOK ----------------
def set_webhook():
    url = f"{URL}/{TOKEN}"
    requests.get(f"{API}/setWebhook?url={url}")


if __name__ == "__main__":
    set_webhook()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))