from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    poll_timeout: int = 30  # seconds plugin waits before empty poll response
    openrouter_key: str = ""  # default key from .env, plugin can override with BYOK

    class Config:
        env_file = ".env"


settings = Settings()
