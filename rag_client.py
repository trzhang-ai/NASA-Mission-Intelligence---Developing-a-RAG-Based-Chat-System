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

    # TODO: Initialize filter variable to None (represents no filtering)

    # TODO: Check if filter parameter exists and is not set to "all" or equivalent
    # TODO: If filter conditions are met, create filter dictionary with appropriate field-value pairs

    # TODO: Execute database query with the following parameters:
        # TODO: Pass search query in the required format
        # TODO: Set maximum number of results to return
        # TODO: Apply conditional filter (None for no filtering, dictionary for specific filtering)

    # TODO: Return query results to caller

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    # TODO: Initialize list with header text for context section

    # TODO: Loop through paired documents and their metadata using enumeration
        # TODO: Extract mission information from metadata with fallback value
        # TODO: Clean up mission name formatting (replace underscores, capitalize)
        # TODO: Extract source information from metadata with fallback value  
        # TODO: Extract category information from metadata with fallback value
        # TODO: Clean up category name formatting (replace underscores, capitalize)
        
        # TODO: Create formatted source header with index number and extracted information
        # TODO: Add source header to context parts list
        
        # TODO: Check document length and truncate if necessary
        # TODO: Add truncated or full document content to context parts list

    # TODO: Join all context parts with newlines and return formatted string