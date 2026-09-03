from shared.models import Memory
from shared.db import AsyncSessionLocal
from ..embeddings import embed_text

import logging

logger = logging.getLogger("save_memory")

async def save_memory(fact: str, thread_id: str | None = None, user_id: int = 0) -> str:
    """Сохранить факт о пользователе в долговременную память.
    
    :param fact: текст факта
    :param thread_id: идентификатор потока (для аудита, НЕ для фильтрации recall)    
    :param user_id: идентификатор пользователя (Telegram user id)
    """
    try:
        embedding = await embed_text(fact)
    except Exception as e:
        logger.error(f"Error creating embedding for fact: {e}")
        return f"Ошибка при создании эмбеддинга"

    memory = Memory(
        content=fact,
        embedding=embedding,
        user_id=user_id,
        thread_id=thread_id
    )

    try:
        async with AsyncSessionLocal() as session:
            session.add(memory)
            await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Error saving memory: {e}")
        return f"Ошибка при сохранении факта"
        
    return "Факт сохранён в долговременной памяти."