"""Runtime settings, overridable with SHOPFLOOR_* environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SHOPFLOOR_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    # OpenAI-compatible endpoint of the local model server (llama.cpp's llama-server)
    llm_base_url: str = "http://127.0.0.1:8081/v1"
    llm_model: str = "local"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_s: float = 600.0

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def plant_db(self) -> Path:
        return self.data_dir / "plant.db"


def get_settings() -> Settings:
    return Settings()
