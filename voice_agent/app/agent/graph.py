from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, AIMessage, trim_messages

from langchain_ollama import ChatOllama

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from .context import AgentContext
from .my_prompts import BASE_SYSTEM_PROMPT, VOICE_SUFFIX

import logging

logger = logging.getLogger("agent")


class TelegramMessagesState(MessagesState):
    """
    State for the Telegram messages.

    Attributes:
        voice (bool): Whether the message is a voice message.
    """
    voice: bool

def build_graph(
        llm: ChatOllama, 
        tools: list[BaseTool], 
        checkpointer: BaseCheckpointSaver,
        ) -> CompiledStateGraph[TelegramMessagesState, AgentContext]:
    llm_with_tools = llm.bind_tools(tools)

    async def agent_node(state: TelegramMessagesState) -> dict:
        try:
            # max_tokens должен вмещать не только диалог, но и результаты тулов —
            # analyze_document может вернуть многотысячесимвольную сводку целиком;
            # при allow_partial=False (дефолт) сообщение, которое не влезает целиком
            # в бюджет, выбрасывается целиком, а не обрезается — 1000 обрезал всё до нуля.
            trimmed = trim_messages(
                state["messages"], max_tokens=8000, strategy="last", token_counter="approximate",
            )
            prompt_text = BASE_SYSTEM_PROMPT + (VOICE_SUFFIX if state.get("voice") else "")
            response = await llm_with_tools.ainvoke([SystemMessage(content=prompt_text), *trimmed])
            return {"messages": [response]}
        except Exception:
            logger.exception("LLM invocation failed")
            return {"messages": [AIMessage(content="Произошла ошибка, попробуйте позже.")]}


    builder = StateGraph(TelegramMessagesState, context_schema=AgentContext)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)