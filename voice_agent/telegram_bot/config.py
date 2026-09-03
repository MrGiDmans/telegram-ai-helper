from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"


class BotSettings(BaseSettings):
    bot_token: str
    api_base_url: str = "http://127.0.0.1:8000"
    upload_dir: Path = BASE_DIR / "voice_agent" / "data" / "uploads"

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )


bot_settings = BotSettings()
