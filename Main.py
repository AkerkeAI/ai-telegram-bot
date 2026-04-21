import os
from flask import Flask, request
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

TOKEN = os.getenv("TOKEN")
URL = os.getenv("URL")  # https://your-app.up.railway.app

app = Flask(__name__)

# Telegram app (python-telegram-bot v20+)
telegram_app = Application.builder().token(TOKEN).build()


# --- handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Бот активен. Я на webhook 🚀")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(update.message.text)


telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))


# --- webhook endpoint ---
@app.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), telegram_app.bot)
    import asyncio
    asyncio.run(telegram_app.process_update(update))
    return "ok"


# --- startup ---
@app.route("/")
def home():
    return "Bot is running"


if __name__ == "__main__":
    import asyncio

    async def on_start():
        await telegram_app.initialize()
        await telegram_app.start()

        await telegram_app.bot.set_webhook(url=f"{URL}/{TOKEN}")

    asyncio.run(on_start())

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))