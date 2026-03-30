from __future__ import annotations
import os
import logging

from flask import Flask
from threading import Thread
from app.bot import build_application
from app.config import load_settings
from app.image_processor import MangaImageProcessor
from app.ocr import OCRService
from app.translators import build_translation_service



def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

app = Flask(__name__)


@app.route("/")
def home():
    return "Bot is alive"


@app.route("/healthz")
def healthz():
    return "ok"


def run_web() -> None:
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)


def keep_alive() -> None:
    Thread(target=run_web, daemon=True).start()
    

# main.py

def main() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    keep_alive()

    logger = logging.getLogger(__name__)
    logger.info("Starting manga translation bot")
    logger.info("Admins configured: %s", sorted(settings.admin_user_ids))

    logger.info("Before OCRService")
    ocr_service = OCRService(settings)
    logger.info("After OCRService")

    logger.info("Before translator")
    translator = build_translation_service(settings)
    logger.info("After translator")

    logger.info("Before processor")
    processor = MangaImageProcessor(
        settings=settings,
        ocr_service=ocr_service,
        translator=translator,
    )
    logger.info("After processor")

    logger.info("Before build_application")
    application = build_application(settings=settings, processor=processor)
    logger.info("Before run_polling")

    application.run_polling(drop_pending_updates=False)
    

if __name__ == "__main__":
    main()
