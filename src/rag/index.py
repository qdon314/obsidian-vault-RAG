from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import chromadb

def build_or_load_index(docs: list[Document] | None, chroma_path: str) -> VectorStoreIndex:
    """
    Build or load an index from a list of documents or an existing Chroma collection.
    If `docs` is provided, build a new index from the documents.
    If `docs` is None, load an existing index from the Chroma collection.
    
    Args:
        docs (list[Document] | None): List of documents to build the index from, or None to load an existing index.
        chroma_path (str): Path to the Chroma database.
    
    
    Returns:
        VectorStoreIndex: The built or loaded index.
    """
    # Local embedding model (fast, good enough to start)
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection("obsidian_notes")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if docs is None:
        # Load existing index from persisted vector store
        return VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # Build index
    return VectorStoreIndex.from_documents(docs, storage_context=storage_context)
