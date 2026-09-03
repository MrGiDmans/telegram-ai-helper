import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, MetaData, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Размерность эмбеддинга должна совпадать с выбранной embedding-моделью.
# nomic-embed-text (через Ollama) отдаёт 768-мерные вектора.
EMBEDDING_DIM = 768

POSTGRES_NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=POSTGRES_NAMING_CONVENTION)


class Memory(Base):
    """
    Постоянная память: отдельные факты о пользователе (save_memory/recall_memory).
    
    Attributes:
        id: уникальный идентификатор памяти (UUID)
        user_id: идентификатор пользователя (Telegram user id)
        thread_id: идентификатор потока (для аудита, НЕ для фильтрации recall)
        content: текст факта
        embedding: векторное представление факта (эмбеддинг)
        created_at: дата и время создания записи (автоматически устанавливается при создании)
    """

    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int] = mapped_column(index=True)  # Telegram user id
    thread_id: Mapped[str | None] = mapped_column(String)  # для аудита, НЕ для фильтрации recall
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Document(Base):
    """Метаданные загруженного документа (один раз на документ).
    
    Attributes:
        id: уникальный идентификатор документа (UUID)
        user_id: идентификатор пользователя (Telegram user id)
        file_id: идентификатор файла в Telegram (file_id)
        title: название документа (может быть пустым)
        storage_path: путь к файлу на сервере (может быть пустым)
        uploaded_at: дата и время загрузки документа (автоматически устанавливается при создании)
        chunks: список чанков текста документа (DocumentChunk)
    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int] = mapped_column(index=True)
    file_id: Mapped[str | None] = mapped_column(String)  # Telegram file_id, для справки/пересылки
    thread_id: Mapped[str | None] = mapped_column(String)  # для аудита, как у Memory
    mime_type: Mapped[str | None] = mapped_column(String)  # напр. application/pdf
    title: Mapped[str] = mapped_column(String)
    storage_path: Mapped[str] = mapped_column(String)
    # Короткое LLM-сгенерированное описание "о чём документ" — для поиска НУЖНОГО
    # документа среди нескольких, отдельно от поиска по содержимому чанков.
    description: Mapped[str | None] = mapped_column(Text)
    description_embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    uploaded_at: Mapped[datetime] = mapped_column(server_default=func.now())

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    """Один чанк текста документа со своим эмбеддингом (search_documents).
    
    Attributes:
        id: уникальный идентификатор чанка (UUID)
        document_id: идентификатор документа (Document.id)
        chunk_index: порядковый номер чанка внутри документа
        page_number: номер страницы PDF, если известен (может быть None)
        content: текст чанка
        embedding: векторное представление чанка (эмбеддинг)
        created_at: дата и время создания записи (автоматически устанавливается при создании)
    """

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int]  # порядковый номер чанка внутри документа
    page_number: Mapped[int | None]  # номер страницы PDF, если известен
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    document: Mapped["Document"] = relationship(back_populates="chunks")
