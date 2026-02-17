class RagAppError(Exception):
    """Base error for the RAG app."""


class IngestionError(RagAppError):
    pass


class ChunkingError(RagAppError):
    pass


class EmbeddingError(RagAppError):
    pass


class VectorStoreError(RagAppError):
    pass


class RetrievalError(RagAppError):
    pass


class RerankError(RagAppError):
    pass


class GenerationError(RagAppError):
    pass


class IndexIncompatibleError(RagAppError):
    pass


class AdamsApiError(RagAppError):
    """NRC ADAMS API failure."""


class AdamsRateLimitError(RagAppError):
    """Rate limit exceeded (HTTP 429)."""


class AdamsAuthError(RagAppError):
    """Authentication failed (HTTP 401/403)."""


class AdamsNotFoundError(RagAppError):
    """Document not found (HTTP 404)."""
