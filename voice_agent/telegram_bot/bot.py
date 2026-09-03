import asyncio
import logging
import uuid

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from .config import bot_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telegram_bot")

bot = Bot(token=bot_settings.bot_token)
dp = Dispatcher()

# chat_id -> текущий thread_id. In-memory: сбрасывается при перезапуске бота,
# для пет-проекта с одним процессом этого достаточно.
_active_threads: dict[int, str] = {}

MAX_TELEGRAM_FILE_SIZE = 20 * 1024 * 1024  # жёсткий лимит Bot API на скачивание файла
CHAT_API_TIMEOUT = 300  # analyze_document может сделать много последовательных вызовов LLM
SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
}


def get_thread_id(chat_id: int) -> str:
    return _active_threads.setdefault(chat_id, f"chat-{chat_id}")


async def call_chat_api(
    message: str,
    *,
    user_id: int,
    thread_id: str,
    file_path: str | None = None,
    file_id: str | None = None,
    mime_type: str | None = None,
) -> str:
    payload = {"message": message, "thread_id": thread_id, "user_id": user_id}
    if file_path is not None:
        payload["file_path"] = file_path
    if file_id is not None:
        payload["file_id"] = file_id
    if mime_type is not None:
        payload["mime_type"] = mime_type

    async with httpx.AsyncClient(timeout=CHAT_API_TIMEOUT) as client:
        response = await client.post(f"{bot_settings.api_base_url}/chat", json=payload)
        response.raise_for_status()
        return response.json()["response"]


@dp.message(CommandStart())
async def handle_start(message: Message) -> None:
    await message.answer(
        "Привет! Я ассистент с памятью. Пиши мне, проси что-то запомнить, "
        "или пришли PDF-документ, чтобы добавить его в базу знаний."
    )


@dp.message(Command("reset"))
async def handle_reset(message: Message) -> None:
    new_thread_id = f"chat-{message.chat.id}-{uuid.uuid4().hex[:8]}"
    _active_threads[message.chat.id] = new_thread_id
    await message.answer("Контекст диалога сброшен. Долговременная память при этом не затронута.")


@dp.message(F.document)
async def handle_document(message: Message) -> None:
    document = message.document

    if document.file_size and document.file_size > MAX_TELEGRAM_FILE_SIZE:
        await message.answer("Файл слишком большой — максимум 20 МБ для бота Telegram.")
        return

    if document.mime_type not in SUPPORTED_MIME_TYPES:
        await message.answer("Пока поддерживаются только PDF и Word (.docx) документы.")
        return

    bot_settings.upload_dir.mkdir(parents=True, exist_ok=True)
    destination = bot_settings.upload_dir / f"{document.file_unique_id}_{document.file_name}"
    await bot.download(document, destination=destination)

    caption = message.caption or "Сохрани этот документ в базу знаний."
    # Явный маркер без самого пути — путь подставляется отдельно через file_path
    # (см. inject_user_context), модель никогда не видит и не указывает его сама.
    text = f"{caption}\n\n[К этому сообщению прикреплён файл.]"

    thread_id = get_thread_id(message.chat.id)
    try:
        answer = await call_chat_api(
            text,
            user_id=message.from_user.id,
            thread_id=thread_id,
            file_path=str(destination.resolve()),
            file_id=document.file_id,
            mime_type=document.mime_type,
        )
    except Exception:
        logger.exception("Ошибка при обработке документа")
        answer = "Произошла ошибка при обработке документа, попробуйте позже."

    await message.answer(answer)


@dp.message(F.text)
async def handle_text(message: Message) -> None:
    thread_id = get_thread_id(message.chat.id)
    try:
        answer = await call_chat_api(message.text, user_id=message.from_user.id, thread_id=thread_id)
    except Exception:
        logger.exception("Ошибка при обращении к /chat")
        answer = "Произошла ошибка, попробуйте позже."

    await message.answer(answer)


async def main() -> None:
    bot_settings.upload_dir.mkdir(parents=True, exist_ok=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
