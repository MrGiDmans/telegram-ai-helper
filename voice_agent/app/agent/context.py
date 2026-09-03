from dataclasses import dataclass


@dataclass
class AgentContext:
    user_id: int
    thread_id: str | None = None
    # Абсолютный путь к файлу, прикреплённому к ТЕКУЩЕМУ сообщению (если есть).
    # Подставляется интерцептором напрямую в ingest_document/analyze_document —
    # модель никогда не видит и не должна сама указывать путь к файлу.
    file_path: str | None = None
    # Реальный Telegram file_id вложения (для справки/пересылки), той же природы,
    # что file_path — не должен придумываться моделью.
    file_id: str | None = None
    # MIME-тип вложения — как file_path/file_id, определяется системой (Telegram),
    # а не придумывается моделью.
    mime_type: str | None = None
