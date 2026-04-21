import os
import asyncio
from flask import Flask, request
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

app = Flask(__name__)

# 🐶 опасные звуки (добавили лай собаки)
danger_keywords = ["signal", "alarm", "cry", "scream", "dog bark"]

# 🔊 обработка сообщений
async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Аудио получено, анализирую...")

    # ❗ тут позже будет реальный анализ
    detected_sound = "dog bark"  # пока тест

    if detected_sound in danger_keywords:
        await update.message.reply_text("⚠️ Опасный звук обнаружен!")

# 🤖 создаем приложение
telegram_app = ApplicationBuilder().token(TOKEN).build()
telegram_app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

# 🌐 webhook endpoint
@app.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), telegram_app.bot)
    asyncio.run(telegram_app.process_update(update))
    return "ok"

# 🟢 главная страница (для проверки)
@app.route("/")
def home():
    return "Айкын работает"

# 🚀 запуск
async def main():
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.bot.set_webhook(f"{WEBHOOK_URL}/{TOKEN}")

# ▶️ старт приложения
if __name__ == "__main__":
    asyncio.run(main())
    app.run(host="0.0.0.0", port=8080)