from pydantic_settings import BaseSettings
from pydantic import Field


class EmbeddingConfig(BaseSettings):
    product_dim: int = Field(default=128)
    user_dim: int = Field(default=128)
    model_name: str = Field(default="all-MiniLM-L6-v2")
    model_config = {"env_prefix": "EMBEDDING_"}


class RetrievalConfig(BaseSettings):
    index_type: str = Field(default="IVF256,Flat")
    num_candidates: int = Field(default=50)
    nprobe: int = Field(default=16)
    model_config = {"env_prefix": "RETRIEVAL_"}


class RankingConfig(BaseSettings):
    learning_rate: float = Field(default=0.05)
    num_trees: int = Field(default=300)
    max_depth: int = Field(default=6)
    model_config = {"env_prefix": "RANKING_"}


class ServingConfig(BaseSettings):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    default_k: int = Field(default=10)
    max_k: int = Field(default=100)
    model_config = {"env_prefix": "SERVING_"}


class RedisConfig(BaseSettings):
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    user_profile_ttl: int = Field(default=86400)
    model_config = {"env_prefix": "REDIS_"}


class AppConfig:
    def __init__(self) -> None:
        self.embedding = EmbeddingConfig()
        self.retrieval = RetrievalConfig()
        self.ranking = RankingConfig()
        self.serving = ServingConfig()
        self.redis = RedisConfig()
