def extract_text(content: str | list) -> str:
    """Достать текст из content ответа LLM.

    По контракту BaseMessage.content это `str | list[str | dict]` — обычно
    просто строка, но иногда приходит списком блоков (мы это уже наблюдали
    у ToolMessage от MCP-тулов и держим в уме, что не-MCP вызовы ChatOllama
    тоже вправе так вернуть). Не-текстовые блоки просто дают пустую строку,
    а не падение.
    """
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
        )
    return content
