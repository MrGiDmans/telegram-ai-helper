from langchain_ollama import ChatOllama

def build_llm(
        name_model: str, 
        reasoning: bool = False, 
        num_predict: int = 512, 
        temperature: float = 0.1
        ) -> ChatOllama:
    
    return ChatOllama(
        model=name_model,
        reasoning=reasoning,
        num_predict=num_predict,
        temperature=temperature
        )