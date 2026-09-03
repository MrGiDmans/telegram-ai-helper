from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

from .agent.graph import build_graph
from .agent.llm import build_llm
from .agent.checkpointer import build_checkpointer
from .schemas.models import ChatRequest, ChatResponse
from .mcp_client.client import get_mcp_tools
from .agent.context import AgentContext
from .config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for the application's lifespan.
    """
    llm = build_llm(settings.name_model, 
                    reasoning=settings.reasoning, 
                    num_predict=settings.num_predict, 
                    temperature=settings.temperature)
    
    tools = await get_mcp_tools()
    checkpointer = build_checkpointer()

    app.state.graph = build_graph(llm, tools, checkpointer)

    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest, request: Request):
    graph = request.app.state.graph
    result = await graph.ainvoke(
        {"messages": [("user", chat_request.message)]},
        config={"configurable": {"thread_id": chat_request.thread_id}, "recursion_limit": 10},
        context=AgentContext(
            user_id=chat_request.user_id,
            thread_id=chat_request.thread_id,
            file_path=chat_request.file_path,
            file_id=chat_request.file_id,
            mime_type=chat_request.mime_type,
        ),
    )
    return ChatResponse(response=result["messages"][-1].content, thread_id=chat_request.thread_id)

