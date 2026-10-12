"""Central settings (override with SPECINFER_* env vars)."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SPECINFER_")

    backend: str = "hf"  # "hf" | "mlx"
    draft_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    target_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lookahead_k: int = 4
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 256

    cache_enabled: bool = True
    cache_path: str = "data/cache_store"
    cache_distance_threshold: float = 0.05
    embed_model: str = "nomic-ai/nomic-embed-text-v1.5"
    embed_dim: int = 768

    host: str = "127.0.0.1"
    port: int = 8000


settings = Settings()
