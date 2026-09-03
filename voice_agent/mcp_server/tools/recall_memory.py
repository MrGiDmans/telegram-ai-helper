from shared.db import AsyncSessionLocal
from shared.models import Memory
from sqlalchemy import select
import logging
from ..embeddings import embed_text

logger = logging.getLogger("recall_memory")

async def recall_memory(query: str, user_id: int = 0) -> str:
    """Найти ранее сохранённые факты о пользователе, релевантные запросу.

    :param user_id: идентификатор пользователя (Telegram user id)
    :param query: запрос пользователя
    """
    try:
        query_embeding = await embed_text(query)
    except Exception as e:
        logger.error(f"Error creating embedding for query: {e}")
        return f"Ошибка при создании эмбеддинга запроса"

    MAX_DISTANCE = 0.4
    stmt = (
        select(Memory)
        .where(
            Memory.user_id == user_id,
            Memory.embedding.cosine_distance(query_embeding) < MAX_DISTANCE
            )
        .order_by(Memory.embedding.cosine_distance(query_embeding))
        .limit(5)
    )

    try:
        async with AsyncSessionLocal() as session:
            result = await session.scalars(stmt)
            memories = result.all()
    except Exception as e:
        logger.error(f"Error querying memories: {e}")
        return f"Ошибка при поиске памяти"

    if not memories:
        return "Нет релевантных фактов в долговременной памяти."

    formatted_memories = "\n".join(
        [f"- {m.content}" for m in memories]
    )

    return f"Найденные релевантные факты:\n{formatted_memories}"
