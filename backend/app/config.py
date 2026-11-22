"""Application configuration using Pydantic Settings."""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(_ENV_FILE, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application settings
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=True, alias="DEBUG")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    # PostgreSQL & pgvector settings
    postgres_user: str = Field(default="postgres", alias="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", alias="POSTGRES_PASSWORD")
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5433, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="tark_studio", alias="POSTGRES_DB")

    # Explicit Database URL override (supports external or container databases)
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")

    # External AI Provider settings
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")

    # Reranker settings (baseline, onnx)
    reranker_type: str = Field(default="baseline", alias="RERANKER_TYPE")
    reranker_model_onnx: str = Field(default="Xenova/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL_ONNX")
    reranker_top_k: int = Field(default=5, alias="RERANKER_TOP_K")
    reranker_candidate_k: int = Field(default=20, alias="RERANKER_CANDIDATE_K")

    # Conversation Memory settings (Rolling summary + Token-budget buffer)
    memory_token_budget: int = Field(default=2048, alias="MEMORY_TOKEN_BUDGET")
    memory_summary_threshold_tokens: int = Field(default=1024, alias="MEMORY_SUMMARY_THRESHOLD_TOKENS")
    memory_summary_min_messages: int = Field(default=4, alias="MEMORY_SUMMARY_MIN_MESSAGES")
    memory_summary_max_tokens: int = Field(default=250, alias="MEMORY_SUMMARY_MAX_TOKENS")

    # Security & CORS settings
    cors_origins: str = Field(default="http://localhost:5173,http://localhost:3000", alias="CORS_ORIGINS")

    @property
    def allowed_cors_origins(self) -> list[str]:
        """Parse comma-separated CORS origins or allow all in debug mode."""
        if self.debug:
            return ["*"]
        if not self.cors_origins:
            return ["http://localhost:5173", "http://localhost:3000"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def async_database_url(self) -> str:
        """Resolve async SQLAlchemy database connection string."""
        if self.database_url:
            url = self.database_url
            if url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            return url
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
