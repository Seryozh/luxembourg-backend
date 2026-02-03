from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    host: str = "0.0.0.0"
    port: int = 8000
    poll_timeout: int = 30
    session_ttl: int = 3600  # Session TTL in seconds (default: 1 hour)
    cleanup_interval: int = 300  # Cleanup interval in seconds (default: 5 minutes)
    max_json_retries: int = 3  # Max retries for JSON parsing in worker
    openrouter_key: str = ""  # Optional: server-side OpenRouter key (not recommended)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
