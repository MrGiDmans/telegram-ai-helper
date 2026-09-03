"""
Изолированная проверка tool calling выбранной модели (qwen3:8b) поверх
реального стека: Ollama + Postgres/pgvector + MCP-сервер по stdio.

Не мокает ничего — это осознанный интеграционный тест на "система вообще
работает", а не unit-тест на изолированную логику.
"""

import uuid

import pytest
from fpdf import FPDF

from app.agent.checkpointer import build_checkpointer
from app.agent.context import AgentContext
from app.agent.graph import build_graph
from app.agent.llm import build_llm
from app.mcp_client.client import get_mcp_tools


@pytest.fixture
def user_id() -> int:
    # Свой user_id на каждый запуск теста, чтобы не пересекаться с данными
    # от предыдущих прогонов и других тестов в той же базе.
    return uuid.uuid4().int % 1_000_000_000


@pytest.fixture
async def graph():
    llm = build_llm("qwen3:8b", reasoning=False, num_predict=512, temperature=0.1)
    tools = await get_mcp_tools()
    checkpointer = build_checkpointer()
    return build_graph(llm, tools, checkpointer)


FILE_MARKER = "[К этому сообщению прикреплён файл.]"


async def ask(
    graph, message: str, *, user_id: int, thread_id: str, file_path: str | None = None
) -> str:
    result = await graph.ainvoke(
        {"messages": [("user", message)]},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 10},
        context=AgentContext(user_id=user_id, thread_id=thread_id, file_path=file_path),
    )
    return result["messages"][-1].content


async def test_save_and_recall_memory(graph, user_id):
    fact = "мой любимый цвет — сиреневый с блёстками"

    await ask(graph, f"Запомни: {fact}", user_id=user_id, thread_id=f"t-{user_id}-save")

    # Другой thread_id намеренно — постоянная память обязана быть видна
    # в НОВОЙ сессии, а не только внутри той же ветки диалога.
    answer = await ask(
        graph, "Какой мой любимый цвет?", user_id=user_id, thread_id=f"t-{user_id}-recall"
    )

    assert "сирен" in answer.lower()


async def test_ingest_and_search_document(graph, user_id, tmp_path):
    secret = "Zefir-9"
    pdf_path = tmp_path / "test_doc.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, text=f"Codename of the project: {secret}. This is a test document.")
    pdf.output(str(pdf_path))

    await ask(
        graph,
        f"Сохрани этот документ в базу знаний.\n\n{FILE_MARKER}",
        user_id=user_id,
        thread_id=f"t-{user_id}-ingest",
        file_path=str(pdf_path),
    )

    # Запрос намеренно на том же языке, что и документ (английский) — кросс-языковой
    # RU-запрос/EN-документ у nomic-embed-text даёт заметно большее косинусное расстояние
    # (проверено: ~0.53 против порога MAX_DISTANCE=0.4) и не находится — это отдельный
    # вопрос настройки порога/модели эмбеддингов, а не баг пайплайна ingest/search.
    answer = await ask(
        graph,
        "What is the project's codename according to the uploaded documents?",
        user_id=user_id,
        thread_id=f"t-{user_id}-search",
    )

    assert secret.lower() in answer.lower()


async def test_analyze_document(graph, user_id, tmp_path):
    pdf_path = tmp_path / "analyze_doc.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, text="This document describes a fictional animal called the Grumpolo.")
    pdf.add_page()
    pdf.cell(0, 10, text="The Grumpolo lives exclusively in cold mountain caves.")

    pdf.output(str(pdf_path))

    answer = await ask(
        graph,
        f"Сделай краткую сводку этого документа.\n\n{FILE_MARKER}",
        user_id=user_id,
        thread_id=f"t-{user_id}-analyze",
        file_path=str(pdf_path),
    )

    # Модель отвечает по-русски и вправе транслитерировать "Grumpolo" -> "Грамполо",
    # поэтому проверяем по содержанию факта, а не по конкретному написанию имени.
    assert "пещер" in answer.lower()
