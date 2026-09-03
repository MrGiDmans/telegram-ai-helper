from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    
    - name_model: The name of the model to be used for the language model.
    - reasoning: A boolean indicating whether reasoning is enabled for the model.
    - num_predict: The number of predictions to generate.
    - temperature: The temperature setting for the model, affecting randomness in predictions.
    """

    name_model: str = "qwen3:8b"
    reasoning: bool = False
    num_predict: int = 512
    temperature: float = 0.1

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()