import uuid

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from sqlalchemy import select

from ..document_parsing import extract_pages
from ..embeddings import embed_text
from ..llm_utils import extract_text
from ..text_splitter import extract_text_chunks
from shared.db import AsyncSessionLocal
from shared.models import Document

_llm = ChatOllama(model="qwen3:8b", temperature=0.1, reasoning=False)
SUMMARY_BATCH_SIZE = 5

async def analyze_document(
    file_path: str | None = None,
    document_id: str | None = None,
    question: str | None = None,
    user_id: int = 0,
) -> str:
    """Проанализировать документ и ответить на вопрос или сделать сводку, ничего не сохраняя в базу знаний.

    Укажи ЛИБО file_path (документ, только что прикреплённый к сообщению),
    ЛИБО document_id (документ, ранее найденный через search_documents).
    """
    document = None

    if document_id is not None:
        try:
            document_uuid = uuid.UUID(document_id)
        except ValueError:
            document_uuid = None
        if document_uuid is not None:
            async with AsyncSessionLocal() as session:
                candidate = await session.get(Document, document_uuid)
            if candidate is not None and candidate.user_id == user_id:
                document = candidate

    # document_id часто оказывается придуманным моделью (не результатом реального
    # search_documents) — вместо жёсткого отказа подстраховываемся сами: ищем документ
    # по сходству его короткого описания (Document.description_embedding) с вопросом —
    # описание генерируется при ingest_document специально под эту задачу, отдельно
    # от поиска по содержимому чанков. "Последний загруженный" был плохим фолбэком —
    # при нескольких документах он подсовывал не тот файл (например, дипломную работу
    # вместо резюме); используем его только как самый последний запасной вариант.
    if document is None and file_path is None:
        async with AsyncSessionLocal() as session:
            if question is not None:
                query_embedding = await embed_text(question)
                stmt = (
                    select(Document)
                    .where(
                        Document.user_id == user_id,
                        Document.description_embedding.is_not(None),
                    )
                    .order_by(Document.description_embedding.cosine_distance(query_embedding))
                    .limit(1)
                )
                document = (await session.scalars(stmt)).first()

            if document is None:
                stmt = (
                    select(Document)
                    .where(Document.user_id == user_id)
                    .order_by(Document.uploaded_at.desc())
                    .limit(1)
                )
                document = (await session.scalars(stmt)).first()

    if document is not None:
        file_path = document.storage_path

    if file_path is None:
        return "У вас пока нет загруженных документов."

    pages_data = extract_pages(file_path)

    # чанки крупнее, чем для search_documents — тут важна экономия вызовов LLM, а не точность попадания
    chunks = extract_text_chunks(pages_data, chunk_size=3000, chunk_overlap=0)
    if not chunks:
        return "Не удалось извлечь текст из документа."

    instruction = f"Ответь на вопрос по тексту ниже: {question}" if question else "Сделай краткую сводку текста ниже."

    current = []
    for chunk in chunks:
        response = await _llm.ainvoke([HumanMessage(content=f"{instruction}\n\nТекст:\n{chunk['content']}")])
        current.append(extract_text(response.content))

    while len(current) > 1:
        next_round = []
        for i in range(0, len(current), SUMMARY_BATCH_SIZE):
            batch = "\n\n---\n\n".join(current[i:i + SUMMARY_BATCH_SIZE])
            response = await _llm.ainvoke([HumanMessage(content=f"Сведи эти сводки в одну связную:\n\n{batch}")])
            next_round.append(extract_text(response.content))
        current = next_round

    return current[0]
