from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

def build_checkpointer() -> BaseCheckpointSaver:
    return MemorySaver()