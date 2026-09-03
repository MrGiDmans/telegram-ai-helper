from pydantic import BaseModel

class ChatRequest(BaseModel):
    """
    Schema for chat request.
    
    :param: message: The message from the user.
    :param: thread_id: The thread ID for the conversation.
    :param: user_id: The user ID of the user sending the message.
    :param: file_path: Absolute path to a file attached to THIS message, if any.
        Not relayed through conversation text — injected straight into
        ingest_document/analyze_document calls so it can't be lost to
        history trimming or mistyped by the model.
    """
    message: str
    thread_id: str
    user_id: int
    file_path: str | None = None
    file_id: str | None = None
    mime_type: str | None = None

class ChatResponse(BaseModel):
    response: str
    thread_id: str