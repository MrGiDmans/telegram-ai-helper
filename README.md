# Voice Agent — LLM-агент с памятью и RAG для Telegram

Self-hosted LLM-агент на базе LangGraph и Ollama с долговременной памятью о
пользователе, базой знаний по загружаемым документам (RAG) и собственным
MCP-сервером инструментов — обёрнутый в FastAPI-сервис, с Telegram-ботом как
клиентским слоем поверх него.

Подробный разбор архитектуры и ключевых решений — в
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Что не реализовано и почему —
в [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Стек

- **LLM** — `qwen3:8b` через Ollama, локально, без внешних API, tool calling
- **Оркестрация** — LangGraph, граф собран вручную (не `create_react_agent`)
- **Инструменты** — собственный MCP-сервер (STDIO-транспорт), 5 тулов, подключается через `langchain-mcp-adapters`
- **Память** — PostgreSQL + pgvector, раздельно: сессионная история (LangGraph checkpointer) и постоянная память/база знаний (свои MCP-тулы)
- **API** — FastAPI
- **Клиент** — Telegram-бот, aiogram v3
- **Миграции** — Alembic
- **Тесты** — pytest, интеграционные, без моков, на реальном стеке (Ollama + Postgres + MCP)

## Быстрый старт

### Требования

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Ollama](https://ollama.com) установлена и запущена
- Токен Telegram-бота от [@BotFather](https://t.me/BotFather)

### Установка

```bash
git clone <repo-url>
cd telegram-ai-helper

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt

cp .env.example .env
# заполнить BOT_TOKEN; остальные значения по умолчанию подходят для локальной разработки

ollama pull qwen3:8b
ollama pull nomic-embed-text

docker compose up -d            # поднимает Postgres + pgvector

cd voice_agent
alembic upgrade head            # накатить схему БД
```

### Запуск

```bash
# терминал 1 — API-сервис (агент живёт здесь)
cd voice_agent
uvicorn app.main:app --reload --port 8000

# терминал 2 — Telegram-бот (просто HTTP-клиент /chat)
cd voice_agent
python -m telegram_bot.bot
```

Дальше просто пишите боту в Telegram — обычным текстом, с просьбой что-то
запомнить, или прикрепив PDF/`.docx` документ.

### Тесты

```bash
cd voice_agent
pytest tests/ -v
```

Тесты реальные (не моки) — поднимают полный граф агента и гоняют его через
живую Ollama, Postgres и MCP-сервер, поэтому требуют поднятой инфраструктуры
из шагов установки выше.
