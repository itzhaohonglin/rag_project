from dataclasses import dataclass, field


@dataclass
class EmbeddingVector:
    vector: list[float]
    dimension: int
    model_name: str

    def __post_init__(self):
        self.dimension = len(self.vector)


@dataclass
class EmbeddingConfig:
    model_name: str = "text-embedding-3-small"
    dimension: int = 1536
    batch_size: int = 32
    max_retries: int = 3
    timeout: int = 60
    provider: str = "openai"  # "openai" or "local"
    local_model_path: str = ""
