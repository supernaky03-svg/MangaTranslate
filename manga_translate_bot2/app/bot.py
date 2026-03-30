from __future__ import annotations

import asyncio
import logging
import mimetypes
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from telegram import File, Message, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from .config import Settings
from .image_processor import MangaImageProcessor
from .models import ImageJob, Submission

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AlbumCollector:
    seq: int
    user_id: int
    chat_id: int
    media_group_id: str
    messages: list[Message] = field(default_factory=list)
    timer_task: asyncio.Task[Any] | None = None


class UserSubmissionState:
    def __init__(self, user_id: int) -> None:
        self.user_id = user_id
        self.next_seq = 1
        self.next_to_process = 1
        self.submissions: dict[int, Submission] = {}
        self.albums: dict[str, AlbumCollector] = {}
        self.condition = asyncio.Condition()
        self.worker_task: asyncio.Task[Any] | None = None

    def allocate_seq(self) -> int:
        seq = self.next_seq
        self.next_seq += 1
        return seq


class SubmissionCoordinator:
    def __init__(self, application: Application, settings: Settings, processor: MangaImageProcessor) -> None:
        self.application = application
        self.settings = settings
        self.processor = processor
        self._states: dict[int, UserSubmissionState] = {}

    def is_admin(self, user_id: int | None) -> bool:
        return bool(user_id) and int(user_id) in self.settings.admin_user_ids

    def _state_for(self, user_id: int) -> UserSubmissionState:
        state = self._states.get(user_id)
        if state is None:
            state = UserSubmissionState(user_id)
            self._states[user_id] = state
        if state.worker_task is None or state.worker_task.done():
            state.worker_task = asyncio.create_task(self._worker_loop(state), name=f"worker-user-{user_id}")
        return state

    async def enqueue_message(self, message: Message) -> None:
        user = message.from_user
        if user is None or not self.is_admin(user.id):
            return

        state = self._state_for(user.id)
        if message.media_group_id:
            await self._collect_album_message(state, message)
            return
        await self._create_single_submission(state, message)

    async def _create_single_submission(self, state: UserSubmissionState, message: Message) -> None:
        if not self._message_has_supported_image(message):
            return

        seq = state.allocate_seq()
        submission = Submission(seq=seq, user_id=state.user_id, chat_id=message.chat_id, ready=False)
        state.submissions[seq] = submission
        try:
            job = await self._download_job(
                message=message,
                user_id=state.user_id,
                seq=seq,
                index_in_submission=0,
                media_group_id=None,
            )
            submission.jobs.append(job)
        except Exception as exc:
            logger.exception("Failed to download single image from message %s", message.message_id)
            if self.settings.notify_on_image_failure:
                await self._safe_notify_failure(
                    chat_id=message.chat_id,
                    text=f"Image download failed for message {message.message_id}: {exc}",
                    reply_to_message_id=message.message_id,
                )
        finally:
            submission.ready = True
            await self._notify_state(state)

    async def _collect_album_message(self, state: UserSubmissionState, message: Message) -> None:
        media_group_id = message.media_group_id
        assert media_group_id is not None
        collector = state.albums.get(media_group_id)
        if collector is None:
            seq = state.allocate_seq()
            collector = AlbumCollector(
                seq=seq,
                user_id=state.user_id,
                chat_id=message.chat_id,
                media_group_id=media_group_id,
            )
            state.albums[media_group_id] = collector
            state.submissions[seq] = Submission(
                seq=seq,
                user_id=state.user_id,
                chat_id=message.chat_id,
                ready=False,
                media_group_id=media_group_id,
            )
        collector.messages.append(message)
        if collector.timer_task and not collector.timer_task.done():
            collector.timer_task.cancel()
        collector.timer_task = asyncio.create_task(self._finalize_album_after_delay(state, collector))

    async def _finalize_album_after_delay(self, state: UserSubmissionState, collector: AlbumCollector) -> None:
        try:
            try:
                await asyncio.sleep(self.settings.media_group_collect_delay)
            except asyncio.CancelledError:
                return

            submission = state.submissions.get(collector.seq)
            if submission is None:
                return

            ordered_messages = sorted(collector.messages, key=lambda msg: msg.message_id)
            jobs: list[ImageJob] = []
            for index, message in enumerate(ordered_messages):
                try:
                    job = await self._download_job(
                        message=message,
                        user_id=state.user_id,
                        seq=collector.seq,
                        index_in_submission=index,
                        media_group_id=collector.media_group_id,
                    )
                    jobs.append(job)
                except Exception as exc:
                    logger.exception(
                        "Failed to download media-group image %s/%s from message %s",
                        collector.media_group_id,
                        index,
                        message.message_id,
                    )
                    if self.settings.notify_on_image_failure:
                        await self._safe_notify_failure(
                            chat_id=message.chat_id,
                            text=f"Album image download failed for message {message.message_id}: {exc}",
                            reply_to_message_id=message.message_id,
                        )
            submission.jobs = jobs
        except Exception:
            logger.exception("Unexpected album finalization failure for media group %s", collector.media_group_id)
        finally:
            submission = state.submissions.get(collector.seq)
            if submission is not None:
                submission.ready = True
            state.albums.pop(collector.media_group_id, None)
            await self._notify_state(state)

    async def _download_job(
        self,
        message: Message,
        user_id: int,
        seq: int,
        index_in_submission: int,
        media_group_id: str | None,
    ) -> ImageJob:
        tg_file, original_filename, mime_type = await self._extract_image_info(message)
        suffix = self._preferred_suffix(original_filename, mime_type)
        safe_name = f"u{user_id}_s{seq}_i{index_in_submission}_{uuid.uuid4().hex}{suffix}"
        local_path = self.settings.incoming_dir / safe_name
        await tg_file.download_to_drive(custom_path=str(local_path))
        return ImageJob(
            user_id=user_id,
            chat_id=message.chat_id,
            source_message_id=message.message_id,
            input_path=local_path,
            original_filename=original_filename,
            mime_type=mime_type,
            submission_seq=seq,
            index_in_submission=index_in_submission,
            media_group_id=media_group_id,
            reply_to_message_id=message.message_id,
        )

    async def _worker_loop(self, state: UserSubmissionState) -> None:
        logger.info("Worker started for admin user_id=%s", state.user_id)
        while True:
            try:
                async with state.condition:
                    await state.condition.wait_for(
                        lambda: state.next_to_process in state.submissions
                        and state.submissions[state.next_to_process].ready
                    )
                    submission = state.submissions.pop(state.next_to_process)
                    state.next_to_process += 1
                await self._process_submission(submission)
            except asyncio.CancelledError:
                logger.info("Worker cancelled for admin user_id=%s", state.user_id)
                raise
            except Exception:
                logger.exception("Unexpected worker failure for user_id=%s", state.user_id)

    async def _process_submission(self, submission: Submission) -> None:
        if not submission.jobs:
            return
        for job in submission.jobs:
            await self._process_job(job)

    async def _process_job(self, job: ImageJob) -> None:
        output_name = self._output_name(job)
        output_path = self.settings.outgoing_dir / output_name
        try:
            result = await asyncio.to_thread(self.processor.process_image, job.input_path, output_path)
            if result.had_text or self.settings.send_original_when_no_text:
                await self._send_result(job, result.output_path, had_text=result.had_text)
        except Exception as exc:
            logger.exception("Image processing failed for %s", job.input_path)
            if self.settings.notify_on_image_failure:
                await self._safe_notify_failure(
                    chat_id=job.chat_id,
                    text=f"Processing failed for image {job.original_filename}: {exc}",
                    reply_to_message_id=job.reply_to_message_id,
                )
            self._move_to_failed(job.input_path)
            if output_path.exists():
                self._move_to_failed(output_path)
        else:
            self._safe_unlink(job.input_path)
            self._safe_unlink(output_path)

    async def _notify_state(self, state: UserSubmissionState) -> None:
        async with state.condition:
            state.condition.notify_all()

    async def _safe_notify_failure(self, chat_id: int, text: str, reply_to_message_id: int | None = None) -> None:
        try:
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_to_message_id=reply_to_message_id,
            )
        except Exception:
            logger.exception("Failed to send failure notification to chat_id=%s", chat_id)

    async def _send_result(self, job: ImageJob, path: Path, had_text: bool) -> None:
        caption = None
        if not had_text:
            caption = "No English text detected. Returning the original page."
        filename = path.name
        original_suffix = Path(job.original_filename).suffix.lower()
        if original_suffix and path.suffix.lower() != original_suffix:
            filename = f"{Path(path).stem}{original_suffix}"
        with path.open("rb") as handle:
            await self.application.bot.send_document(
                chat_id=job.chat_id,
                document=handle,
                filename=filename,
                caption=caption,
                reply_to_message_id=job.reply_to_message_id,
            )

    def _output_name(self, job: ImageJob) -> str:
        original = Path(job.original_filename)
        stem = original.stem if original.stem else f"page_{job.source_message_id}"
        ext = self._preferred_suffix(job.original_filename, job.mime_type)
        return f"translated_{job.submission_seq:05d}_{job.index_in_submission:03d}_{stem}{ext}"

    def _move_to_failed(self, path: Path) -> None:
        if not path.exists():
            return
        failed_path = self.settings.failed_dir / f"{uuid.uuid4().hex}_{path.name}"
        try:
            shutil.move(str(path), str(failed_path))
        except Exception:
            logger.exception("Failed to move %s to failed dir", path)

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            logger.exception("Failed to remove temp file %s", path)

    @staticmethod
    def _preferred_suffix(filename: str, mime_type: str) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
            return suffix
        guessed = mimetypes.guess_extension(mime_type or "")
        if guessed in {".jpg", ".jpeg", ".png", ".webp"}:
            return guessed
        return ".png"

    @staticmethod
    async def _extract_image_info(message: Message) -> tuple[File, str, str]:
        if message.photo:
            photo = message.photo[-1]
            return await photo.get_file(), f"photo_{message.message_id}.jpg", "image/jpeg"
        document = message.document
        if document and (document.mime_type or "").startswith("image/"):
            return await document.get_file(), document.file_name or f"document_{message.message_id}.png", document.mime_type
        raise ValueError("Message does not contain a supported image.")

    @staticmethod
    def _message_has_supported_image(message: Message) -> bool:
        if message.photo:
            return True
        document = message.document
        return bool(document and (document.mime_type or "").startswith("image/"))

    async def shutdown(self) -> None:
        tasks: list[asyncio.Task[Any]] = []
        for state in self._states.values():
            if state.worker_task and not state.worker_task.done():
                state.worker_task.cancel()
                tasks.append(state.worker_task)
            for album in state.albums.values():
                if album.timer_task and not album.timer_task.done():
                    album.timer_task.cancel()
                    tasks.append(album.timer_task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    coordinator: SubmissionCoordinator = context.application.bot_data["coordinator"]
    user = update.effective_user
    if user is None or not coordinator.is_admin(user.id):
        return
    await update.effective_message.reply_text(
        "Send one or more manga/comic images.\n"
        "The bot will detect English text, translate it into natural Burmese, redraw it in place, and return the edited pages in order.\n\n"
        "Supported: single images, multiple separate images, and albums/media groups.\n"
        "Use /help for tips and limits."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    coordinator: SubmissionCoordinator = context.application.bot_data["coordinator"]
    user = update.effective_user
    if user is None or not coordinator.is_admin(user.id):
        return
    await update.effective_message.reply_text(
        "How it works:\n"
        "• Best results come from clear English dialogue inside speech bubbles or text boxes.\n"
        "• You can send single pages, many pages one by one, or Telegram albums.\n"
        "• Images are processed sequentially in the same order they were received.\n"
        "• Only approved admin user IDs can use this bot.\n"
        "• If OCR or translation partially fails, the bot skips bad regions and keeps going."
    )


async def image_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    coordinator: SubmissionCoordinator = context.application.bot_data["coordinator"]
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None or not coordinator.is_admin(user.id):
        return
    await coordinator.enqueue_message(message)


async def post_shutdown(application: Application) -> None:
    coordinator: SubmissionCoordinator | None = application.bot_data.get("coordinator")
    if coordinator is not None:
        await coordinator.shutdown()


def build_application(settings: Settings, processor: MangaImageProcessor) -> Application:
    application = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .concurrent_updates(False)
        .post_shutdown(post_shutdown)
        .build()
    )

    coordinator = SubmissionCoordinator(application=application, settings=settings, processor=processor)
    application.bot_data["coordinator"] = coordinator

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(
        MessageHandler(filters.PHOTO | filters.Document.IMAGE, image_message_handler)
    )
    return application
