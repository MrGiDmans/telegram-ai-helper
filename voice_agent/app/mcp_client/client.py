import sys
from collections.abc import Sequence
from typing import Annotated, cast

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from langchain_core.tools import BaseTool, InjectedToolArg
from pydantic import BaseModel, create_model
from pydantic_core import PydanticUndefined

from ..agent.context import AgentContext

# Поля, которые не должна заполнять LLM — подставляются интерцептором из
# доверенного контекста запроса (см. inject_user_context), а не моделью.
SENSITIVE_ARGS: Sequence[str] = ("user_id", "thread_id", "file_path", "file_id", "mime_type")


async def inject_user_context(request: MCPToolCallRequest, handler):
    # ToolNode инжектит langgraph.prebuilt.ToolRuntime, а не сам langgraph.runtime.Runtime —
    # это НЕ подклассы друг друга (разные иерархии), общий у них только атрибут .context,
    # поэтому проверяем через duck typing, а не isinstance по конкретному классу.
    runtime = request.runtime
    context = getattr(runtime, "context", None)
    if isinstance(context, AgentContext):
        overrides = {
            **request.args,
            "user_id": context.user_id,
            "thread_id": context.thread_id,
        }
        # file_path/file_id подставляем, только если к ЭТОМУ сообщению реально прикреплён
        # файл — иначе затрём document_id-based вызов analyze_document пустым значением.
        if context.file_path is not None:
            overrides["file_path"] = context.file_path
        if context.file_id is not None:
            overrides["file_id"] = context.file_id
        if context.mime_type is not None:
            overrides["mime_type"] = context.mime_type
        request = request.override(args=overrides)
    return await handler(request)


def hide_sensitive_args(tool: BaseTool, hidden_args: Sequence[str]) -> BaseTool:
    """Убирает поля из схемы, видимой LLM, — их подставляет интерцептор, не модель.

    Тулы из langchain-mcp-adapters хранят `args_schema` как сырой JSON-словарь
    (не pydantic-класс) — `tool_call_schema` для dict-схем просто возвращает
    тот же словарь как есть, без какой-либо фильтрации InjectedToolArg (та
    фильтрация работает только для pydantic-классов). Поэтому тут чистим сам
    словарь напрямую, а не пересобираем pydantic-модель.
    """
    schema = tool.args_schema

    if isinstance(schema, dict):
        to_hide = set(hidden_args) & set(schema.get("properties", {}))
        if not to_hide:
            return tool
        still_required = to_hide & set(schema.get("required", []))
        if still_required:
            raise ValueError(
                f"Нельзя скрыть обязательные поля {still_required} у тула '{tool.name}' без значения по умолчанию"
            )
        tool.args_schema = {
            **schema,
            "properties": {k: v for k, v in schema["properties"].items() if k not in to_hide},
            "required": [r for r in schema.get("required", []) if r not in to_hide],
        }
        return tool

    if not isinstance(schema, type):
        return tool
    schema = cast(type[BaseModel], schema)

    to_hide = set(hidden_args) & set(schema.model_fields)
    if not to_hide:
        return tool

    fields = {}
    for name, field in schema.model_fields.items():
        default = ... if field.default is PydanticUndefined else field.default
        if name in to_hide:
            if default is ...:
                raise ValueError(
                    f"Нельзя скрыть обязательное поле '{name}' у тула '{tool.name}' без значения по умолчанию"
                )
            fields[name] = (Annotated[field.annotation, InjectedToolArg()], default)
        else:
            fields[name] = (field.annotation, default)

    tool.args_schema = create_model(schema.__name__, **fields)
    return tool


async def get_mcp_tools() -> list[BaseTool]:
    client = MultiServerMCPClient(
        {
        "memory_server": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "mcp_server.server"],
            # Без этого дочерний процесс на Windows читает/пишет stdio в системной
            # кодировке консоли (обычно не UTF-8), и кириллица бьётся при передаче.
            "env": {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        }
        },
        tool_interceptors=[inject_user_context],
    )

    tools = await client.get_tools()
    return [hide_sensitive_args(tool, SENSITIVE_ARGS) for tool in tools]