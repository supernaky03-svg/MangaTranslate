# Manga/Comic English ➜ Burmese Telegram Bot

Production-style Telegram bot that:
- accepts manga/comic pages from Telegram
- silently ignores non-admin users
- handles single images, multiple separate images, and Telegram albums/media groups
- OCRs English text regions
- translates dialogue into natural Burmese
- redraws Burmese into the original text region
- returns output images in the same order they were received

## Important quality note
For genuinely human-like Burmese manga dialogue, configure an LLM backend with `OPENAI_API_KEY` or `GEMINI_API_KEY`.

The built-in fallback translator is only a crash-safe backup. It keeps the bot working if the API is down, but it is **not** comparable to an LLM for natural manga dialogue.

## Recommended stack
- Python 3.11+
- `python-telegram-bot`
- `EasyOCR`
- `OpenCV`
- `Pillow`
- a Myanmar font such as **Noto Sans Myanmar** or **Padauk** installed on the host

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with your:
- `TELEGRAM_BOT_TOKEN`
- `ADMIN_USER_IDS`
- at least one translation backend key (`OPENAI_API_KEY` or `GEMINI_API_KEY`) for best results
- `MM_FONT_PATH` if your host does not already have a usable Myanmar font in one of the default system locations

## Run
```bash
python main.py
```

## Commands
- `/start` – short usage guide for admins only
- `/help` – concise instructions/limits for admins only

Non-admin users are ignored silently.

## How ordering works
Each admin gets an independent submission pipeline.

- single images are queued in arrival order
- albums reserve their queue position as soon as the first media-group message arrives
- pages inside the same album are processed in Telegram message order
- one failed image does not block the remaining pages

## How translation/redraw works
1. EasyOCR detects English text regions.
2. Regions are merged and sorted in reading order.
3. English text is translated into concise Burmese.
4. Original text is removed with inpainting.
5. The bot tries to fit Burmese inside the same region using:
   - line wrapping
   - font size reduction
   - optional shortening rewrite pass
6. Output image is sent back to Telegram.

## Deployment notes
- `easyocr` pulls in PyTorch and is heavier than a basic bot. Prefer a host with enough RAM.
- For CPU-only deployment, keep `OCR_GPU=false`.
- Install a Myanmar font on the server and set `MM_FONT_PATH` if needed.
- If you need higher quality bubble detection later, you can replace `OCRService.detect_regions()` with a stronger detector without changing the Telegram queueing layer.

## Known trade-offs
This implementation is real and automatic, but manga cleanup is still heuristic:
- speech bubble detection is approximated from OCR text regions plus padding
- solid background fill is only applied when the region looks like a bright bubble
- artistic restoration is intentionally secondary to readability

That trade-off is deliberate: readability, stability, and ordered batch processing were prioritized.
