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
