from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    
    gemini_api_key: str = Field(..., validation_alias="GEMINI_API_KEY")
    model_name: str = "gemini-3.1-flash-lite"
    
    # Hybrid Search Settings
    hybrid_top_k: int = 4
    dense_weight: float = 0.5
    sparse_weight: float = 0.5

    # LangSmith
    langchain_tracing_v2: bool = Field(True, validation_alias="LANGCHAIN_TRACING_V2")
    langchain_endpoint: str = Field("https://api.smith.langchain.com", validation_alias="LANGCHAIN_ENDPOINT")
    langchain_api_key: str | None = Field(None, validation_alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field("rag-assistant", validation_alias="LANGCHAIN_PROJECT")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


def load_policy(policy_path: str = "policy.yaml") -> dict[str, Any]:
    path = Path(policy_path)
    if not path.exists():
        raise FileNotFoundError(f"❌ Помилка: Файл політик {policy_path} не знайдено!")
    
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


try:
    settings = Settings()
    policy = load_policy()
except Exception as e:
    print(f"❌ Помилка ініціалізації конфігурації: {e}")
    raise e