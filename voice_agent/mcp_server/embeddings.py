from langchain_ollama import OllamaEmbeddings

_embeddings = OllamaEmbeddings(model="nomic-embed-text", keep_alive=30 * 60)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    return await _embeddings.aembed_documents(texts)


async def embed_text(text: str) -> list[float]:
    return (await embed_texts([text]))[0]
