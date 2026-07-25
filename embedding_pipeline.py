#!/usr/bin/env python3
"""
ChromaDB Embedding Pipeline for NASA Space Mission Data - Text Files Only

This script reads parsed text data from various NASA space mission folders and creates
a permanent ChromaDB collection with OpenAI embeddings for RAG applications.
Optimized to process only text files to avoid duplication with JSON versions.

Supported data sources:
- Apollo 11 extracted data (text files only)
- Apollo 13 extracted data (text files only)
- Apollo 11 Textract extracted data (text files only)
- Challenger transcribed audio data (text files only)
"""

from dotenv import load_dotenv
from nasa_text_cleaners import build_all_nasa_dataframes
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import chromadb
import openai
from openai import OpenAI
import hashlib
import time
import re
from datetime import datetime
import argparse
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.utils import get_tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chroma_embedding_text_only.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChromaEmbeddingPipelineTextOnly:
    """Pipeline for creating ChromaDB collections with OpenAI embeddings - Text files only"""
    
    def __init__(
        self,
        openai_api_key: str,
        openai_base_url: Optional[str] = None,
        chroma_persist_directory: str = "./chroma_db",
        collection_name: str = "nasa_space_missions_text",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the embedding pipeline
        
        Args:
            openai_api_key: OpenAI API key
            openai_base_url: Optional base URL for an OpenAI-compatible API
            chroma_persist_directory: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
            embedding_model: OpenAI embedding model to use
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks
        """
        if not openai_api_key or not openai_api_key.strip():
            raise ValueError("openai_api_key must not be empty")
        self.openai_client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_base_url,
        )
        self.embedding_function = OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            api_base=openai_base_url,
            model_name=embedding_model,
        )
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size"
            )
        self.openai_base_url = openai_base_url
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_persist_directory,
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Text to chunk
            metadata: Base metadata for the text
            
        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """
        cleaned_text = text.strip()
        if not cleaned_text:
            return []
        tokenizer = get_tokenizer()
        token_count = len(tokenizer(cleaned_text))
        if token_count <= self.chunk_size:
            chunk_metadata = metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "token_count": token_count,
                }
            )
            return [(cleaned_text, chunk_metadata)]
        source_type = metadata.get("source_type")
        if source_type not in {"report", "transcript"}:
            raise ValueError(
                "metadata['source_type'] must be 'report' or 'transcript'"
            )
        paragraph_separator = "\n" if source_type == "transcript" else "\n\n"
        token_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=" ",
            backup_separators=[paragraph_separator],
            keep_whitespaces=True,
        )
        chunks = token_splitter.split_text(cleaned_text)
        chunk_records = []
        total_chunks = len(chunks)
        for chunk_index, chunk in enumerate(chunks):
            chunk_token_count = len(tokenizer(chunk))

            if chunk_token_count > self.chunk_size:
                raise RuntimeError(
                    f"Chunk {chunk_index} has {chunk_token_count} tokens, "
                    f"exceeding chunk_size={self.chunk_size}"
                )
            chunk_metadata = metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "token_count": chunk_token_count,
                }
            )
            chunk_records.append((chunk, chunk_metadata))
        return chunk_records
    
    def check_document_exists(self, doc_id: str) -> bool:
        """
        Check if a document with the given ID already exists in the collection
        
        Args:
            doc_id: Document ID to check
            
        Returns:
            True if document exists, False otherwise
        """
        if not doc_id or not doc_id.strip():
            raise ValueError("doc_id must not be empty")
        try:
            result = self.collection.get(
                ids=[doc_id],
                include=[],
            )
        except Exception:
            logger.exception(
                "Failed to check whether document %s exists",
                doc_id,
            )
            raise
        return bool(result["ids"])
    
    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """
        Update an existing document in the collection
        
        Args:
            doc_id: Document ID to update
            text: New text content
            metadata: New metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get new embedding
            embedding = self.get_embedding(text)
            
            # Update the document
            self.collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            logger.debug(f"Updated document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False
    
    def delete_documents_by_source(self, source_pattern: str) -> int:
        """
        Delete all documents from a specific source (useful for re-processing files)
        
        Args:
            source_pattern: Pattern to match source names
            
        Returns:
            Number of documents deleted
        """
        try:
            # Get all documents
            all_docs = self.collection.get()
            
            # Find documents matching the source pattern
            ids_to_delete = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if source_pattern in metadata.get('source', ''):
                    ids_to_delete.append(all_docs['ids'][i])
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} documents matching source pattern: {source_pattern}")
                return len(ids_to_delete)
            else:
                logger.info(f"No documents found matching source pattern: {source_pattern}")
                return 0
                
        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            return 0
    
    def get_file_documents(self, file_path: Path) -> List[str]:
        """
        Get all document IDs for a specific file
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of document IDs for the file
        """
        try:
            source = file_path.stem
            mission = self.extract_mission_from_path(file_path)
            
            # Get all documents
            all_docs = self.collection.get()
            
            # Find documents from this file
            file_doc_ids = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if (metadata.get('source') == source and 
                    metadata.get('mission') == mission):
                    file_doc_ids.append(all_docs['ids'][i])
            
            return file_doc_ids
            
        except Exception as e:
            logger.error(f"Error getting file documents: {e}")
            return []
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get OpenAI embedding for text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
        except Exception:
            logger.exception(
                "Failed to create embedding with model %s",
                self.embedding_model,
            )
            raise
        if not response.data:
            raise RuntimeError("The embedding API returned no embedding data")
        return response.data[0].embedding

    def generate_document_id(self, file_path: Path, metadata: Dict[str, Any]) -> str:
        """
        Generate stable document ID based on file path and chunk position
        This allows for document updates without changing IDs
        """
        mission = str(metadata.get("mission", "")).strip()
        source_value = (
            metadata.get("source")
            or metadata.get("source_file")
            or file_path.stem
        )
        source = Path(str(source_value)).stem.strip()
        chunk_index = metadata.get("chunk_index")
        if not mission:
            raise ValueError("metadata['mission'] must not be empty")
        if not source:
            raise ValueError("metadata must contain a non-empty source")
        if (
            not isinstance(chunk_index, int)
            or isinstance(chunk_index, bool)
            or chunk_index < 0
        ):
            raise ValueError(
                "metadata['chunk_index'] must be a non-negative integer"
            )

        def slugify(value: str) -> str:
            slug = re.sub(
                r"[^a-z0-9]+",
                "_",
                value.casefold(),
            ).strip("_")
            if not slug:
                raise ValueError("ID component must contain a letter or digit")
            return slug
        return (
            f"{slugify(mission)}::{slugify(source)}::"
            f"chunk_{chunk_index:04d}"
        )
    
    def process_text_file(self, file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Process plain text files with enhanced metadata extraction
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of (text, metadata) tuples
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return []
            
            # Enhanced metadata extraction
            metadata = {
                'source': file_path.stem,
                'file_path': str(file_path),
                'file_type': 'text',
                'content_type': 'full_text',
                'mission': self.extract_mission_from_path(file_path),
                'data_type': self.extract_data_type_from_path(file_path),
                'document_category': self.extract_document_category_from_filename(file_path.name),
                'file_size': len(content),
                'processed_timestamp': datetime.now().isoformat()
            }
            
            return self.chunk_text(content, metadata)
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return []
    
    def extract_mission_from_path(self, file_path: Path) -> str:
        """Extract mission name from file path"""
        path_str = str(file_path).lower()
        if 'apollo11' in path_str or 'apollo_11' in path_str:
            return 'apollo_11'
        elif 'apollo13' in path_str or 'apollo_13' in path_str:
            return 'apollo_13'
        elif 'challenger' in path_str:
            return 'challenger'
        else:
            return 'unknown'
    
    def extract_data_type_from_path(self, file_path: Path) -> str:
        """Extract data type from file path"""
        path_str = str(file_path).lower()
        if 'transcript' in path_str:
            return 'transcript'
        elif 'textract' in path_str:
            return 'textract_extracted'
        elif 'audio' in path_str:
            return 'audio_transcript'
        elif 'flight_plan' in path_str:
            return 'flight_plan'
        else:
            return 'document'
    
    def extract_document_category_from_filename(self, filename: str) -> str:
        """Extract document category from filename for better organization"""
        filename_lower = filename.lower()
        
        # Apollo transcript types
        if 'pao' in filename_lower:
            return 'public_affairs_officer'
        elif 'cm' in filename_lower:
            return 'command_module'
        elif 'tec' in filename_lower:
            return 'technical'
        elif 'flight_plan' in filename_lower:
            return 'flight_plan'
        
        # Challenger audio segments
        elif 'mission_audio' in filename_lower:
            return 'mission_audio'
        
        # NASA archive documents
        elif 'ntrs' in filename_lower:
            return 'nasa_archive'
        elif '19900066485' in filename_lower:
            return 'technical_report'
        elif '19710015566' in filename_lower:
            return 'mission_report'
        
        # General categories
        elif 'full_text' in filename_lower:
            return 'complete_document'
        else:
            return 'general_document'
    
    def scan_text_files_only(self, base_path: str) -> List[Path]:
        """
        Scan data directories for text files only (avoiding JSON duplicates)
        
        Args:
            base_path: Base directory path
            
        Returns:
            List of text file paths to process
        """
        base_path = Path(base_path)
        files_to_process = []
        
        # Define directories to scan
        data_dirs = [
            'apollo11',
            'apollo13',
            'challenger'
        ]
        
        for data_dir in data_dirs:
            dir_path = base_path / data_dir
            if dir_path.exists():
                logger.info(f"Scanning directory: {dir_path}")
                
                # Find only text files
                text_files = list(dir_path.glob('**/*.txt'))
                files_to_process.extend(text_files)
                logger.info(f"Found {len(text_files)} text files in {data_dir}")
        
        # Filter out unwanted files
        filtered_files = []
        for file_path in files_to_process:
            # Skip system files and summaries
            if (file_path.name.startswith('.') or 
                'summary' in file_path.name.lower() or
                file_path.suffix.lower() != '.txt'):
                continue
            filtered_files.append(file_path)
        
        logger.info(f"Total text files to process: {len(filtered_files)}")
        
        # Log file breakdown by mission
        mission_counts = {}
        for file_path in filtered_files:
            mission = self.extract_mission_from_path(file_path)
            mission_counts[mission] = mission_counts.get(mission, 0) + 1
        
        logger.info("Files by mission:")
        for mission, count in mission_counts.items():
            logger.info(f"  {mission}: {count} files")
        
        return filtered_files
    
    def chunk_cleaned_records_by_file(
        self,
        cleaned_df: Any,
    ) -> Dict[Path, List[Tuple[str, Dict[str, Any]]]]:
        """
        Chunk validated cleaner records and group chunks by source file.

        This method performs no OpenAI calls and no ChromaDB writes.
        """
        documents_by_file: Dict[
            Path,
            List[Tuple[str, Dict[str, Any]]],
        ] = {}

        for position, row in cleaned_df.iterrows():
            document = row["document"]
            metadata = row["metadata"]

            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Metadata at position {position} must be a dictionary"
                )

            source_path = metadata.get("source_path")
            if not source_path:
                raise ValueError(
                    f"Metadata at position {position} has no source_path"
                )

            file_path = Path(source_path)
            chunks = self.chunk_text(document, metadata)

            documents_by_file.setdefault(file_path, []).extend(chunks)

        return documents_by_file

    def add_documents_to_collection(self, documents: List[Tuple[str, Dict[str, Any]]], 
                                   file_path: Path, batch_size: int = 50, 
                                   update_mode: str = 'skip') -> Dict[str, int]:
        """
        Add documents to ChromaDB collection in batches with update handling
        
        Args:
            documents: List of (text, metadata) tuples
            file_path: Path to the source file
            batch_size: Number of documents to process in each batch
            update_mode: How to handle existing documents:
                        'skip' - skip existing documents
                        'update' - update existing documents
                        'replace' - delete all existing documents from file and re-add
            
        Returns:
            Dictionary with counts of added, updated, and skipped documents
        """
        valid_update_modes = {"skip", "update", "replace"}
        if update_mode not in valid_update_modes:
            raise ValueError(
                "update_mode must be 'skip', 'update', or 'replace'"
            )
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if not documents:
            return {'added': 0, 'updated': 0, 'skipped': 0}
        stats = {'added': 0, 'updated': 0, 'skipped': 0}
        prepared_documents = []
        for position, item in enumerate(documents):
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(
                    f"Document at position {position} must be a "
                    "(text, metadata) tuple"
                )
            text, metadata = item
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Document text at position {position} must not be empty"
                )
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Metadata at position {position} must be a dictionary"
                )
            prepared_metadata = metadata.copy()
            mission_value = (
                prepared_metadata.get("collection")
                or prepared_metadata.get("mission")
            )
            mission = re.sub(
                r"[^a-z0-9]+",
                "",
                str(mission_value or "").casefold(),
            )
            valid_missions = {"apollo11", "apollo13", "challenger"}
            if mission not in valid_missions:
                raise ValueError(
                    f"Metadata at position {position} has an unsupported "
                    f"mission: {mission_value!r}"
                )
            prepared_metadata["mission"] = mission
            source = (
                prepared_metadata.get("source")
                or prepared_metadata.get("source_file")
                or file_path.name
            )
            filepath = (
                prepared_metadata.get("filepath")
                or prepared_metadata.get("file_path")
                or prepared_metadata.get("source_path")
                or str(file_path)
            )
            prepared_metadata["source"] = str(source).strip()
            prepared_metadata["filepath"] = str(filepath).strip()
            if not prepared_metadata["source"]:
                raise ValueError(
                    f"Metadata at position {position} has an empty source"
                )
            if not prepared_metadata["filepath"]:
                raise ValueError(
                    f"Metadata at position {position} has an empty filepath"
                )
            document_id = self.generate_document_id(
                file_path,
                prepared_metadata,
            )
            prepared_documents.append(
                (document_id, text, prepared_metadata)
            )
        document_ids = [
            document_id
            for document_id, _, _ in prepared_documents
        ]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError(
                "Generated document IDs must be unique within one file"
            )
        try:
            existing_result = self.collection.get(
                ids=document_ids,
                include=[],
            )
        except Exception:
            logger.exception(
                "Failed to inspect existing document IDs"
            )
            raise
        existing_ids = set(existing_result["ids"])
        existing_documents = [
            record
            for record in prepared_documents
            if record[0] in existing_ids
        ]
        new_documents = [
            record
            for record in prepared_documents
            if record[0] not in existing_ids
        ]

        def write_batches(records, write_method):
            for start in range(0, len(records), batch_size):
                batch = records[start:start + batch_size]
                write_method(
                    ids=[
                        document_id
                        for document_id, _, _ in batch
                    ],
                    documents=[
                        text
                        for _, text, _ in batch
                    ],
                    metadatas=[
                        metadata
                        for _, _, metadata in batch
                    ],
                )

        if update_mode == "skip":
            write_batches(
                new_documents,
                self.collection.add,
            )
            stats["added"] = len(new_documents)
            stats["skipped"] = len(existing_documents)
            return stats
        if update_mode == "update":
            write_batches(
                existing_documents,
                self.collection.update,
            )
            write_batches(
                new_documents,
                self.collection.add,
            )
            stats["updated"] = len(existing_documents)
            stats["added"] = len(new_documents)
            return stats
        if update_mode == "replace":
            file_scopes = {
                (
                    metadata["mission"],
                    metadata["filepath"],
                )
                for _, _, metadata in prepared_documents
            }
            if len(file_scopes) != 1:
                raise ValueError(
                    "replace mode requires documents from exactly one file"
                )
            mission, filepath = next(iter(file_scopes))
            file_filter = {
                "$and": [
                    {"mission": {"$eq": mission}},
                    {"filepath": {"$eq": filepath}},
                ]
            }
            try:
                existing_file_result = self.collection.get(
                    where=file_filter,
                    include=[],
                )
            except Exception:
                logger.exception(
                    "Failed to inspect existing documents for %s",
                    filepath,
                )
                raise
            existing_file_ids = set(
                existing_file_result["ids"]
            )
            write_batches(
                prepared_documents,
                self.collection.upsert,
            )
            incoming_ids = set(document_ids)
            stale_document_ids = sorted(
                existing_file_ids - incoming_ids
            )
            for start in range(
                0,
                len(stale_document_ids),
                batch_size,
            ):
                batch_ids = stale_document_ids[
                    start:start + batch_size
                ]
                self.collection.delete(ids=batch_ids)
            stats["updated"] = len(existing_documents)
            stats["added"] = len(new_documents)
            return stats
    
    def process_all_text_data(
        self,
        base_path: str,
        update_mode: str = "skip",
        batch_size: int = 50,
    ) -> Dict[str, int]:
        """
        Process all text files and add to ChromaDB
        
        Args:
            base_path: Base directory containing data folders
            update_mode: How to handle existing documents:
                        'skip' - skip existing documents (default)
                        'update' - update existing documents
                        'replace' - delete all existing documents from file and re-add
            batch_size: Number of chunks written in each ChromaDB batch
        Returns:
            Statistics about processed files
        """
        stats = {
            'files_processed': 0,
            'documents_added': 0,
            'documents_updated': 0,
            'documents_skipped': 0,
            'errors': 0,
            'total_chunks': 0,
            'missions': {}
        }
        valid_update_modes = {"skip", "update", "replace"}
        if update_mode not in valid_update_modes:
            raise ValueError(
                "update_mode must be 'skip', 'update', or 'replace'"
            )
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        # Build validated semantic records from the cleaned NASA corpus.
        _, _, cleaned_df = build_all_nasa_dataframes(base_path)
        # Chunk each record while preserving its provenance, then group by source file.
        documents_by_file = self.chunk_cleaned_records_by_file(cleaned_df)
        stats["total_chunks"] = sum(
            len(documents)
            for documents in documents_by_file.values()
        )
        logger.info(
            "Prepared %d chunks from %d source files",
            stats["total_chunks"],
            len(documents_by_file),
        )
        for file_path, documents in documents_by_file.items():
            try:
                file_stats = self.add_documents_to_collection(
                    documents=documents,
                    file_path=file_path,
                    batch_size=batch_size,
                    update_mode=update_mode,
                )
                stats["files_processed"] += 1
                stats["documents_added"] += file_stats["added"]
                stats["documents_updated"] += file_stats["updated"]
                stats["documents_skipped"] += file_stats["skipped"]
                mission = documents[0][1].get("mission", "unknown")
                mission_stats = stats["missions"].setdefault(
                    mission,
                    {
                        "files": 0,
                        "chunks": 0,
                        "added": 0,
                        "updated": 0,
                        "skipped": 0,
                    },
                )
                mission_stats["files"] += 1
                mission_stats["chunks"] += len(documents)
                mission_stats["added"] += file_stats["added"]
                mission_stats["updated"] += file_stats["updated"]
                mission_stats["skipped"] += file_stats["skipped"]
            except Exception:
                stats["errors"] += 1
                logger.exception(
                    "Failed to process cleaned records from %s",
                    file_path,
                )
        return stats
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection"""
        return {
            "collection_name": self.collection.name,
            "document_count": self.collection.count(),
            "metadata": self.collection.metadata or {},
        }
    
    def query_collection(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the collection for testing
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            
        Returns:
            Query results
        """
        if (
            not isinstance(query_text, str)
            or not query_text.strip()
        ):
            raise ValueError("query_text must not be empty")
        if (
            not isinstance(n_results, int)
            or isinstance(n_results, bool)
            or n_results <= 0
        ):
            raise ValueError(
                "n_results must be a positive integer"
            )
        try:
            return self.collection.query(
                query_texts=[query_text.strip()],
                n_results=n_results,
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )
        except Exception:
            logger.exception(
                "Failed to query the collection"
            )
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the collection"""
        try:
            all_docs = self.collection.get(
                include=["metadatas"]
            )
            metadatas = all_docs.get("metadatas") or []
            stats = {
                "collection_name": self.collection.name,
                "total_documents": self.collection.count(),
                "source_files": 0,
                "missions": {},
                "data_types": {},
                "document_categories": {},
                "file_types": {},
            }
            if not metadatas:
                return stats
            # Analyze metadata
            source_files = set()
            for metadata in metadatas:
                metadata = metadata or {}
                filepath = (
                    metadata.get("filepath")
                    or metadata.get("file_path")
                    or metadata.get("source_path")
                    or metadata.get("source")
                )
                if filepath:
                    source_files.add(str(filepath))
                mission = (
                    metadata.get("mission")
                    or "unknown"
                )
                data_type = (
                    metadata.get("data_type")
                    or metadata.get("source_type")
                    or metadata.get("filetype")
                    or "unknown"
                )
                doc_category = (
                    metadata.get("document_category")
                    or "unknown"
                )
                file_type = (
                    metadata.get("filetype")
                    or metadata.get("file_type")
                    or metadata.get("source_type")
                    or "unknown"
                )
                # Count by mission
                stats['missions'][mission] = stats['missions'].get(mission, 0) + 1
                # Count by data type
                stats['data_types'][data_type] = stats['data_types'].get(data_type, 0) + 1
                # Count by document category
                stats['document_categories'][doc_category] = stats['document_categories'].get(doc_category, 0) + 1
                # Count by file type
                stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1
            stats["source_files"] = len(source_files)
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}

def main():
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path)
    parser = argparse.ArgumentParser(
        description='ChromaDB Embedding Pipeline for NASA Data'
    )
    parser.add_argument(
        "--data-path",
        default="data_text",
        help=(
            "Directory containing the Apollo 11, Apollo 13, "
            "and Challenger data folders"
        ),
    )
    parser.add_argument(
        "--openai-key",
        default=os.getenv("OPENAI_API_KEY"),
        help=(
            "OpenAI API key; defaults to OPENAI_API_KEY "
            "from the environment or .env"
        ),
    )
    parser.add_argument('--chroma-dir', default='./chroma_db_openai', help='ChromaDB persist directory')
    parser.add_argument('--collection-name', default='nasa_space_missions_text', help='Collection name')
    parser.add_argument('--embedding-model', default='text-embedding-3-small', help='OpenAI embedding model')
    parser.add_argument('--chunk-size', type=int, default=500, help='Text chunk size')
    parser.add_argument('--chunk-overlap', type=int, default=100, help='Chunk overlap size')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--update-mode', choices=['skip', 'update', 'replace'], default='skip',
                       help='How to handle existing documents: skip, update, or replace')
    parser.add_argument('--test-query', help='Test query after processing')
    parser.add_argument('--stats-only', action='store_true', help='Only show collection statistics')
    parser.add_argument('--delete-source', help='Delete all documents from a specific source pattern')
    args = parser.parse_args()
    if not args.openai_key:
        parser.error(
            "OpenAI API key not found. Set OPENAI_API_KEY "
            "in .env or pass --openai-key."
        )
    # Initialize pipeline
    logger.info("Initializing ChromaDB Embedding Pipeline...")
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    # Handle delete source operation
    if args.delete_source:
        deleted_count = pipeline.delete_documents_by_source(args.delete_source)
        logger.info(f"Deleted {deleted_count} documents matching source pattern: {args.delete_source}")
        return
    # If stats only, show collection statistics and exit
    if args.stats_only:
        logger.info("Collection Statistics:")
        stats = pipeline.get_collection_stats()
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        return
    # Process all data
    logger.info(f"Starting text data processing with update mode: {args.update_mode}")
    start_time = time.time()
    stats = pipeline.process_all_text_data(
        args.data_path,
        update_mode=args.update_mode,
        batch_size=args.batch_size,
    )
    end_time = time.time()
    processing_time = end_time - start_time
    # Print results
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Total chunks created: {stats['total_chunks']}")
    logger.info(f"Documents added to collection: {stats['documents_added']}")
    logger.info(f"Documents updated in collection: {stats['documents_updated']}")
    logger.info(f"Documents skipped (already exist): {stats['documents_skipped']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Processing time: {processing_time:.2f} seconds")
    # Mission breakdown
    logger.info("\nMission breakdown:")
    for mission, mission_stats in stats['missions'].items():
        logger.info(f"  {mission}: {mission_stats['files']} files, {mission_stats['chunks']} chunks")
        logger.info(f"    Added: {mission_stats['added']}, Updated: {mission_stats['updated']}, Skipped: {mission_stats['skipped']}")
    # Collection info
    collection_info = pipeline.get_collection_info()
    logger.info(f"\nCollection: {collection_info.get('collection_name', 'N/A')}")
    logger.info(f"Total documents in collection: {collection_info.get('document_count', 'N/A')}")
    # Test query if provided
    if args.test_query:
        logger.info(f"\nTesting query: '{args.test_query}'")
        results = pipeline.query_collection(args.test_query)
        if results and 'documents' in results:
            logger.info(f"Found {len(results['documents'][0])} results:")
            for i, doc in enumerate(results['documents'][0][:3]):  # Show top 3
                logger.info(f"Result {i+1}: {doc[:200]}...")
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
