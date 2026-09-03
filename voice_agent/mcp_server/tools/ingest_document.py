import logging
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from shared.db import AsyncSessionLocal
from shared.models import Document, DocumentChunk

from ..document_parsing import extract_pages
from ..embeddings import embed_text, embed_texts
from ..llm_utils import extract_text
from ..text_splitter import extract_text_chunks

logger = logging.getLogger("ingest_document")

_llm = ChatOllama(model="qwen3:8b", temperature=0.1, reasoning=False)
DESCRIPTION_SAMPLE_CHARS = 3000


async def _describe_document(sample_text: str) -> str | None:
    """Короткое описание документа для последующего поиска НУЖНОГО документа
    среди нескольких — отдельно от поиска по содержимому чанков."""
    try:
        response = await _llm.ainvoke([
            HumanMessage(content=(
                "Опиши одним-двумя предложениями, что это за документ и о чём он. "
                "ПЕРВЫМ делом явно назови ТИП документа через несколько синонимов "
                "в скобках, если они есть — например 'Резюме (CV)' или 'Дипломная "
                "работа (диплом, выпускная квалификационная работа)' — это важно "
                "для последующего поиска, слово 'резюме' само по себе неоднозначно "
                "(CV vs просто 'сводка/итог'), поэтому пиши оба варианта.\n"
                "Пример: 'Резюме (CV) кандидата в области Computer Vision с опытом...' "
                "или 'Дипломная работа (диплом) по классификации астрономических "
                "изображений...'. Только описание, без вступлений.\n\n"
                f"Текст документа:\n{sample_text}"
            ))
        ])
        return extract_text(response.content).strip() or None
    except Exception as e:
        logger.error(f"Ошибка при генерации описания документа: {e}")
        return None


async def ingest_document(
        file_path: str | None = None,
        thread_id: str | None = None,
        mime_type: str | None = None,
        file_id: str | None = None,
        user_id: int = 0
        ) -> str:
    """Сохранить прикреплённый к сообщению файл в базу знаний (разбить на чанки и проиндексировать)."""
    if file_path is None:
        return "К этому сообщению не прикреплён файл."

    pages_data = extract_pages(file_path)
    chunks = extract_text_chunks(pages_data)

    description = None
    description_embedding = None
    sample_text = " ".join(page["text"] for page in pages_data)[:DESCRIPTION_SAMPLE_CHARS]
    if sample_text.strip():
        description = await _describe_document(sample_text)
        if description is not None:
            try:
                description_embedding = await embed_text(description)
            except Exception as e:
                logger.error(f"Ошибка при создании эмбеддинга описания документа: {e}")
                description = None

    async with AsyncSessionLocal() as session:
        document = Document(
            file_id=file_id,
            thread_id=thread_id,
            mime_type=mime_type,
            user_id=user_id,
            title=Path(file_path).name,
            storage_path=file_path,
            description=description,
            description_embedding=description_embedding,
        )
        session.add(document)
        await session.flush()  # Получаем ID документа

        if chunks:
            try:
                embeddings = await embed_texts([chunk["content"] for chunk in chunks])
            except Exception as e:
                logger.error(f"Ошибка при создании эмбеддингов для документа {document.id}: {e}")
                await session.rollback()
                return f"Ошибка при создании эмбеддингов для документа"
            for chunk, embedding in zip(chunks, embeddings):
                document_chunk = DocumentChunk(
                    document_id=document.id,
                    chunk_index=chunk["chunk_index"],
                    page_number=chunk["page_number"],
                    content=chunk["content"],
                    embedding=embedding
                )
                session.add(document_chunk)

        try:
            await session.commit()
        except Exception as e:
            logger.error(f"Ошибка при сохранении документа {document.id} в базу данных: {e}")
            await session.rollback()
            return f"Ошибка при сохранении документа в базу данных"

    return f"Документ успешно обработан и сохранён в базу данных."
