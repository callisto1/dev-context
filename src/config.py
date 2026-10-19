from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://devctx:devctx@localhost:5432/dev_context_engine"

    llm_base_url: str = "http://ollama:11434/v1"
    llm_model: str = "qwen2.5-coder:14b"
    llm_code_model: str = "qwen2.5-coder:14b"
    llm_docs_model: str = "qwen2.5:14b"
    llm_vision_model: str = "llava:13b"
    llm_api_key: str = "not-needed"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.2

    embedding_mode: str = "voyage"  # "local", "voyage", or "remote"
    embedding_model: str = "voyage-code-3"
    embedding_dimension: int = 1024
    embedding_api_key: str = ""  # Voyage API key (or generic remote key)
    embedding_base_url: str = ""  # only for mode=remote (OpenAI-compatible endpoint)

    rag_top_k: int = 10
    rag_similarity_threshold: float = 0.3
    rag_max_context_tokens: int = 12000

    git_clone_dir: str = "/tmp/dev-context-engine/repos"
    git_token: str = ""
    git_user: str = ""

    confluence_base_url: str = ""  # e.g. https://yourcompany.atlassian.net
    confluence_user_email: str = ""
    confluence_api_token: str = ""

    reranker_enabled: bool = True
    reranker_mode: str = "local"  # "local" or "remote"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 5  # final number of chunks after reranking
    reranker_base_url: str = ""  # only for remote mode
    reranker_api_key: str = ""  # only for remote mode

    rag_retrieval_k: int = 30  # broad retrieval before reranking

    reindex_enabled: bool = True
    reindex_default_cron: str = "0 0 * * *"  # daily at midnight UTC

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
