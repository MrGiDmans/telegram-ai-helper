from mcp.server.fastmcp import FastMCP

from .tools.save_memory import save_memory
from .tools.recall_memory import recall_memory
from .tools.search_documents import search_documents
from .tools.ingest_document import ingest_document
from .tools.analyze_document import analyze_document

mcp = FastMCP("voice_agent_memory")

mcp.tool()(save_memory)
mcp.tool()(recall_memory)
mcp.tool()(search_documents)
mcp.tool()(ingest_document)
mcp.tool()(analyze_document)

if __name__ == "__main__":
    mcp.run(transport="stdio")
