from __future__ import annotations

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
    app.run(host="0.0.0.0", port=PORT)


def keep_alive() -> None:
    Thread(target=run_web, daemon=True).start()
    

def main() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    keep_alive()

    logger = logging.getLogger(__name__)
    logger.info("Starting manga translation bot")
    logger.info("Admins configured: %s", sorted(settings.admin_user_ids))

    ocr_service = OCRService(settings)
    translator = build_translation_service(settings)
    processor = MangaImageProcessor(settings=settings, ocr_service=ocr_service, translator=translator)
    application = build_application(settings=settings, processor=processor)
    application.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    main()
