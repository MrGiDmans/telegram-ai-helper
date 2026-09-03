from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_chunks(
    pages_data: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []
    chunk_index = 0

    for page in pages_data:
        page_chunks = splitter.split_text(page["text"])
        for chunk_text in page_chunks:
            chunks.append({
                "chunk_index": chunk_index,
                "page_number": page["page_number"],
                "content": chunk_text
            })
            chunk_index += 1

    return chunks