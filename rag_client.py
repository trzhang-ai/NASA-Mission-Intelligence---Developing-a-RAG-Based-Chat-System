import chromadb
import os
from chromadb.utils.embedding_functions import (
    OpenAIEmbeddingFunction,
)
from typing import Dict, List, Optional
from pathlib import Path

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends."""
    backends = {}

    # TODO: Implement backend discovery later.

    return backends

def initialize_rag_system(
    chroma_dir: str,
    collection_name: str,
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
):
    """Open an existing Chroma collection for retrieval."""
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    base_url = (
        openai_base_url
        or os.getenv("OPENAI_BASE_URL")
    )

    if not api_key or not api_key.strip():
        raise ValueError("OpenAI API key must not be empty")

    embedding_function = OpenAIEmbeddingFunction(
        api_key=api_key,
        api_base=base_url,
        model_name=embedding_model,
    )

    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )

    return collection, True, None

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    if (
        not isinstance(n_results, int)
        or isinstance(n_results, bool)
        or n_results <= 0
    ):
        raise ValueError("n_results must be a positive integer")
    where_filter = None
    if (
        mission_filter
        and mission_filter.strip().casefold()
        not in {"all", "all missions"}
    ):
        mission_key = "".join(
            character
            for character in mission_filter.casefold()
            if character.isalnum()
        )
        mission_aliases = {
            "apollo11": "apollo11",
            "apollo13": "apollo13",
            "challenger": "challenger",
            "sts51l": "challenger",
        }

        if mission_key not in mission_aliases:
            raise ValueError(
                f"Unsupported mission filter: {mission_filter!r}"
            )
        where_filter = {
            "mission": {
                "$eq": mission_aliases[mission_key]
            }
        }
    return collection.query(
        query_texts=[query.strip()],
        n_results=n_results,
        where=where_filter,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

def format_context(
    documents: List[str],
    metadatas: List[Dict],
) -> str:
    """Build cited LLM context from retrieved chunks."""
    if not documents:
        return ""
    if len(documents) != len(metadatas):
        raise ValueError(
            "documents and metadatas must have equal lengths"
        )
    mission_labels = {
        "apollo11": "Apollo 11",
        "apollo13": "Apollo 13",
        "challenger": "Challenger",
        "sts51l": "Challenger",
    }
    context_blocks = []
    seen_documents = set()
    for document, metadata in zip(documents, metadatas):
        if not isinstance(document, str) or not document.strip():
            continue
        if not isinstance(metadata, dict):
            raise ValueError("Each metadata value must be a dictionary")
        cleaned_document = document.strip()
        deduplication_key = " ".join(
            cleaned_document.split()
        ).casefold()
        if deduplication_key in seen_documents:
            continue
        seen_documents.add(deduplication_key)
        mission_value = str(
            metadata.get("mission", "unknown")
        )
        mission_key = "".join(
            character
            for character in mission_value.casefold()
            if character.isalnum()
        )
        mission = mission_labels.get(
            mission_key,
            mission_value,
        )
        source = (
            metadata.get("source_file")
            or metadata.get("source")
            or "Unknown source"
        )
        filepath = (
            metadata.get("filepath")
            or metadata.get("source_path")
            or "Unknown filepath"
        )
        source_type = (
            metadata.get("source_type")
            or metadata.get("document_category")
            or "unknown"
        )
        header_lines = [
            f"[DOCUMENT {len(context_blocks) + 1}]",
            f"MISSION = {mission}",
            f"SOURCE = {source}",
            f"FILEPATH = {filepath}",
            f"TYPE = {source_type}",
        ]
        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")
        if page_start is not None:
            header_lines.append(
                f"PAGES = {page_start}-{page_end or page_start}"
            )
        line_start = metadata.get("source_line_start")
        line_end = metadata.get("source_line_end")
        if line_start is not None:
            header_lines.append(
                f"LINES = {line_start}-{line_end or line_start}"
            )
        context_blocks.append(
            "\n".join(header_lines)
            + "\n\n"
            + cleaned_document
        )
    if not context_blocks:
        return ""
    return (
        "RETRIEVED DOCUMENTS\n\n"
        + "\n\n---\n\n".join(context_blocks)
    )
