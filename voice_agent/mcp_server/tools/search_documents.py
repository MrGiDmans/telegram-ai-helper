import logging

from sqlalchemy import select

from shared.db import AsyncSessionLocal
from shared.models import Document, DocumentChunk
from ..embeddings import embed_text

logger = logging.getLogger("search_documents")

MAX_DISTANCE = 0.4


async def search_documents(query: str, user_id: int = 0) -> str:
    """Найти информацию в загруженных документах, релевантную запросу."""
    try:
        query_embedding = await embed_text(query)
    except Exception as e:
        logger.error(f"Ошибка при создании эмбеддинга запроса: {e}")
        return "Ошибка при создании эмбеддинга запроса."

    stmt = (
        select(DocumentChunk, Document.title, Document.id)
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(
            Document.user_id == user_id,
            DocumentChunk.embedding.cosine_distance(query_embedding) < MAX_DISTANCE,
        )
        .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
        .limit(5)
    )

    try:
        async with AsyncSessionLocal() as session:
            rows = (await session.execute(stmt)).all()
    except Exception as e:
        logger.error(f"Ошибка при поиске по документам: {e}")
        return "Ошибка при поиске по документам."

    if not rows:
        return "Ничего релевантного не найдено в загруженных документах."

    formatted = "\n".join(
        f"- [{title}, id={doc_id}, стр. {chunk.page_number}] {chunk.content}"
        for chunk, title, doc_id in rows
    )
    return f"Найденные фрагменты:\n{formatted}"
