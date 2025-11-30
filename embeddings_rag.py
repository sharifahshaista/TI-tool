"""
JSON to Embeddings Processor with S3 Storage
Processes JSON files, creates embeddings with structured metadata,
and stores them in S3 for RAG applications.
"""

import json
import hashlib
import tiktoken
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import for embeddings
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("OpenAI not available")

# Import S3 storage
try:
    from aws_storage import get_storage
    HAS_S3 = True
except ImportError:
    get_storage = None  # type: ignore
    HAS_S3 = False
    logger.warning("S3 storage not available - embeddings will only be stored locally")


@dataclass
class EmbeddingChunk:
    """Represents a single chunk with its embedding and metadata."""
    chunk_id: str  # Format: {record_id}_chunk_{index}
    record_id: str  # Unique identifier for the source JSON record
    text: str  # The text content that was embedded
    embedding: List[float]  # The embedding vector
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any]  # Structured metadata from JSON


@dataclass
class EmbeddingIndex:
    """Container for all embeddings from a JSON file."""
    index_name: str
    source_file: str
    created_at: str
    embedding_model: str
    num_documents: int
    num_chunks: int
    chunks: List[EmbeddingChunk]


class JSONEmbeddingProcessor:
    """
    Process JSON files and create embeddings with structured metadata.
    Supports chunking for long indicators and S3 storage.
    """
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-large",
        max_chunk_tokens: int = 500,
        chunk_overlap: int = 50,
        enable_s3_sync: bool = True,
        local_storage_dir: str = "./embeddings_storage"
    ):
        """
        Initialize the embedding processor.
        
        Args:
            embedding_model: OpenAI embedding model name
            max_chunk_tokens: Maximum tokens per chunk
            chunk_overlap: Token overlap between chunks
            enable_s3_sync: Enable automatic S3 sync
            local_storage_dir: Directory for local storage
        """
        self.embedding_model = embedding_model
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_overlap = chunk_overlap
        self.enable_s3_sync = enable_s3_sync and HAS_S3
        self.local_storage_dir = Path(local_storage_dir)
        self.local_storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Initialize OpenAI client
        if not HAS_OPENAI:
            raise ImportError("OpenAI package required. Install with: pip install openai")
        
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
        
        self.client = OpenAI(api_key=api_key)
        
        # Initialize S3 storage
        self.s3_storage = None
        if self.enable_s3_sync:
            try:
                if get_storage is None:
                    raise ImportError("get_storage not available")
                self.s3_storage = get_storage()
                logger.info("✓ S3 sync enabled for embeddings")
            except Exception as e:
                logger.warning(f"S3 storage initialization failed: {e}")
                self.enable_s3_sync = False
    
    def generate_record_id(self, url: str) -> str:
        """Generate unique record ID from URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def chunk_text(self, text: str, record_id: str) -> List[Dict[str, Any]]:
        """
        Chunk long text into smaller pieces with overlap.
        
        Args:
            text: Text to chunk
            record_id: Unique record identifier
            
        Returns:
            List of chunk dictionaries
        """
        tokens = self.encoding.encode(text)
        chunks = []
        
        # If text fits in one chunk, don't split
        if len(tokens) <= self.max_chunk_tokens:
            return [{
                "text": text,
                "chunk_index": 0,
                "total_chunks": 1,
                "record_id": record_id
            }]
        
        # Split into chunks with overlap
        start = 0
        chunk_index = 0
        
        while start < len(tokens):
            end = min(start + self.max_chunk_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunks.append({
                "text": chunk_text,
                "chunk_index": chunk_index,
                "total_chunks": 0,  # Will update after
                "record_id": record_id
            })
            
            chunk_index += 1
            start = end - self.chunk_overlap
        
        # Update total chunks
        for chunk in chunks:
            chunk["total_chunks"] = len(chunks)
        
        return chunks
    
    def create_embedding_text(self, entry: Dict[str, Any], chunk_text: Optional[str] = None) -> str:
        """
        Create formatted text for embedding from JSON entry.
        
        Args:
            entry: JSON entry dictionary
            chunk_text: Optional chunk of indicator text (if chunking is used)
            
        Returns:
            Formatted text string for embedding
        """
        indicator = chunk_text if chunk_text else entry.get("Indicator", "")
        
        return f"""Title: {entry.get('title', '')}
Categories: {entry.get('categories', '')}
Technology: {entry.get('Tech', '')}
Dimension: {entry.get('Dimension', '')}
TRL: {entry.get('TRL', '')}
Start-up: {entry.get('Start-up', '')}
Date: {entry.get('publication_date', '')}

Content: {indicator}"""
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create( # type: ignore
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def process_entry(self, entry: Dict[str, Any]) -> List[EmbeddingChunk]:
        """
        Process a single JSON entry and create embedding chunks.
        
        Args:
            entry: JSON entry dictionary
            
        Returns:
            List of EmbeddingChunk objects
        """
        record_id = self.generate_record_id(entry["url"])
        indicator = entry.get("Indicator", "")
        
        # Chunk the indicator text
        chunks = self.chunk_text(indicator, record_id)
        
        embedding_chunks = []
        for chunk in chunks:
            # Create embedding text
            embedding_text = self.create_embedding_text(entry, chunk["text"])
            
            # Generate embedding
            embedding_vector = self.create_embedding(embedding_text)
            
            # Create structured metadata
            metadata = {
                "record_id": record_id,
                "url": entry.get("url", ""),
                "title": entry.get("title", ""),
                "publication_date": entry.get("publication_date", ""),
                "categories": entry.get("categories", ""),
                "dimension": entry.get("Dimension", ""),
                "tech": entry.get("Tech", ""),
                "trl": float(entry.get("TRL", 0.0)) if entry.get("TRL") else 0.0,
                "startup": entry.get("Start-up", "") or entry.get("URL to start-ups", ""),
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                "full_indicator": indicator  # Store complete indicator
            }
            
            # Create embedding chunk
            embedding_chunk = EmbeddingChunk(
                chunk_id=f"{record_id}_chunk_{chunk['chunk_index']}",
                record_id=record_id,
                text=embedding_text,
                embedding=embedding_vector,
                chunk_index=chunk["chunk_index"],
                total_chunks=chunk["total_chunks"],
                metadata=metadata
            )
            
            embedding_chunks.append(embedding_chunk)
        
        return embedding_chunks
    
    def process_json_file(
        self,
        json_path: str,
        progress_callback=None
    ) -> EmbeddingIndex:
        """
        Process entire JSON file and create embeddings.
        
        Args:
            json_path: Path to JSON file
            progress_callback: Optional callback for progress updates
            
        Returns:
            EmbeddingIndex object containing all embeddings
        """
        json_path_obj = Path(json_path)
        index_name = json_path_obj.stem
        
        if progress_callback:
            progress_callback("Loading JSON file...", 0, 100)
        
        # Load JSON entries
        with open(json_path_obj, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        
        total = len(entries)
        all_chunks = []
        
        if progress_callback:
            progress_callback(f"Processing {total} entries...", 0, total)
        
        # Process each entry
        for i, entry in enumerate(entries, 1):
            try:
                chunks = self.process_entry(entry)
                all_chunks.extend(chunks)
                
                if progress_callback and i % 10 == 0:
                    progress_callback(
                        f"Processing entry {i}/{total} ({len(all_chunks)} chunks created)",
                        i,
                        total
                    )
            except Exception as e:
                logger.error(f"Error processing entry {i}: {e}")
                continue
        
        # Create embedding index
        embedding_index = EmbeddingIndex(
            index_name=index_name,
            source_file=str(json_path),
            created_at=datetime.now().isoformat(),
            embedding_model=self.embedding_model,
            num_documents=total,
            num_chunks=len(all_chunks),
            chunks=all_chunks
        )
        
        if progress_callback:
            progress_callback(
                f"Created {len(all_chunks)} embeddings from {total} documents",
                total,
                total
            )
        
        return embedding_index
    
    def save_index_locally(self, embedding_index: EmbeddingIndex) -> str:
        """
        Save embedding index to local storage.
        
        Args:
            embedding_index: EmbeddingIndex to save
            
        Returns:
            Path to saved file
        """
        file_path = self.local_storage_dir / f"{embedding_index.index_name}.pkl"
        
        with open(file_path, 'wb') as f:
            pickle.dump(embedding_index, f)
        
        logger.info(f"✓ Saved embeddings locally: {file_path}")
        return str(file_path)
    
    def load_index_locally(self, index_name: str) -> EmbeddingIndex:
        """
        Load embedding index from local storage.
        
        Args:
            index_name: Name of the index
            
        Returns:
            EmbeddingIndex object
        """
        file_path = self.local_storage_dir / f"{index_name}.pkl"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Index not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            embedding_index = pickle.load(f)
        
        logger.info(f"✓ Loaded embeddings from: {file_path}")
        return embedding_index
    
    def upload_index_to_s3(self, embedding_index: EmbeddingIndex) -> bool:
        """
        Upload embedding index to S3.
        
        Args:
            embedding_index: EmbeddingIndex to upload
            
        Returns:
            True if successful
        """
        if not self.enable_s3_sync or not self.s3_storage:
            return False
        
        try:
            # First save locally
            local_path = self.save_index_locally(embedding_index)
            
            # Upload to S3
            s3_key = f"rag_embeddings/{embedding_index.index_name}.pkl"
            success = self.s3_storage.upload_file(
                local_path,
                s3_key,
                metadata={
                    'index_name': embedding_index.index_name,
                    'created_at': embedding_index.created_at,
                    'num_documents': str(embedding_index.num_documents),
                    'num_chunks': str(embedding_index.num_chunks),
                    'embedding_model': embedding_index.embedding_model
                }
            )
            
            if success:
                logger.info(f"✓ Uploaded embeddings to S3: {s3_key}")
            else:
                logger.error(f"✗ Failed to upload embeddings to S3")
            
            return success
            
        except Exception as e:
            logger.error(f"Error uploading to S3: {e}")
            return False
    
    def download_index_from_s3(self, index_name: str) -> Optional[EmbeddingIndex]:
        """
        Download embedding index from S3.
        
        Args:
            index_name: Name of the index
            
        Returns:
            EmbeddingIndex object or None if not found
        """
        if not self.enable_s3_sync or not self.s3_storage:
            return None
        
        try:
            s3_key = f"rag_embeddings/{index_name}.pkl"
            local_path = self.local_storage_dir / f"{index_name}.pkl"
            
            # Download from S3
            success = self.s3_storage.download_file(s3_key, str(local_path))
            if not success:
                logger.error(f"Failed to download from S3: {s3_key}")
                return None
            
            # Load from local file
            return self.load_index_locally(index_name)
            
        except Exception as e:
            logger.error(f"Error downloading from S3: {e}")
            return None
    
    def list_s3_indexes(self) -> List[str]:
        """
        List all available indexes in S3.
        
        Returns:
            List of index names
        """
        if not self.enable_s3_sync or not self.s3_storage:
            return []
        
        try:
            files = self.s3_storage.list_files(prefix="rag_embeddings/", suffix=".pkl")
            # Extract index names from file paths
            index_names = [Path(f).stem for f in files]
            return index_names
        except Exception as e:
            logger.error(f"Error listing S3 indexes: {e}")
            return []
    
    def list_local_indexes(self) -> List[str]:
        """
        List all available indexes in local storage.
        
        Returns:
            List of index names
        """
        if not self.local_storage_dir.exists():
            return []
        
        index_files = list(self.local_storage_dir.glob("*.pkl"))
        return [f.stem for f in index_files]
    
    def query_similar(
        self,
        embedding_index: EmbeddingIndex,
        query_text: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find most similar chunks to query using cosine similarity.
        
        Args:
            embedding_index: EmbeddingIndex to query
            query_text: Query text
            top_k: Number of top results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of result dictionaries with scores and metadata
        """
        import numpy as np
        
        # Create embedding for query
        query_embedding = self.create_embedding(query_text)
        query_vector = np.array(query_embedding)
        
        # Calculate cosine similarity for all chunks
        results = []
        for chunk in embedding_index.chunks:
            # Apply metadata filters if specified
            if filter_metadata:
                match = True
                for key, value in filter_metadata.items():
                    if chunk.metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            # Calculate cosine similarity
            chunk_vector = np.array(chunk.embedding)
            similarity = np.dot(query_vector, chunk_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(chunk_vector)
            )
            
            results.append({
                'chunk_id': chunk.chunk_id,
                'record_id': chunk.record_id,
                'score': float(similarity),
                'text': chunk.text,
                'metadata': chunk.metadata
            })
        
        # Sort by similarity score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:top_k]
    
    def get_full_record(
        self,
        embedding_index: EmbeddingIndex,
        record_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific record.
        
        Args:
            embedding_index: EmbeddingIndex to search
            record_id: Record ID to retrieve
            
        Returns:
            Dictionary with full record information
        """
        chunks = [c for c in embedding_index.chunks if c.record_id == record_id]
        
        if not chunks:
            return None
        
        # Sort chunks by index
        chunks.sort(key=lambda x: x.chunk_index)
        
        # Get metadata from first chunk (all chunks have same metadata)
        metadata = chunks[0].metadata
        
        return {
            'record_id': record_id,
            'url': metadata['url'],
            'title': metadata['title'],
            'publication_date': metadata['publication_date'],
            'categories': metadata['categories'],
            'dimension': metadata['dimension'],
            'tech': metadata['tech'],
            'trl': metadata['trl'],
            'startup': metadata['startup'],
            'full_indicator': metadata['full_indicator'],
            'total_chunks': len(chunks),
            'chunks': [{'chunk_index': c.chunk_index, 'text': c.text} for c in chunks]
        }


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = JSONEmbeddingProcessor(
        embedding_model="text-embedding-3-large",
        max_chunk_tokens=500,
        chunk_overlap=50,
        enable_s3_sync=True
    )
    
    # Process JSON file
    print("Processing JSON file...")
    embedding_index = processor.process_json_file(
        "path/to/your/data.json",
        progress_callback=lambda msg, current, total: print(f"{msg} [{current}/{total}]")
    )
    
    print(f"\n✓ Created {embedding_index.num_chunks} embeddings from {embedding_index.num_documents} documents")
    
    # Save locally
    processor.save_index_locally(embedding_index)
    
    # Upload to S3
    if processor.enable_s3_sync:
        processor.upload_index_to_s3(embedding_index)
    
    # Query example
    print("\n--- Query Example ---")
    results = processor.query_similar(
        embedding_index,
        "reforestation projects in Brazil",
        top_k=3
    )
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Title: {result['metadata']['title']}")
        print(f"   URL: {result['metadata']['url']}")
        print(f"   Chunk: {result['metadata']['chunk_index']+1}/{result['metadata']['total_chunks']}")
    
    # Get full record
    if results:
        record_id = results[0]['record_id']
        full_record = processor.get_full_record(embedding_index, record_id)
        print(f"\n--- Full Record ---")
        if full_record:
            print(json.dumps({k: v for k, v in full_record.items() if k != 'chunks'}, indent=2))
    
    # List available indexes
    print("\n--- Available Indexes ---")
    print("Local:", processor.list_local_indexes())
    if processor.enable_s3_sync:
        print("S3:", processor.list_s3_indexes())


import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import shutil
import tempfile
import logging

from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Import Azure OpenAI embeddings
try:
    from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
except ImportError:
    AzureOpenAIEmbedding = None
    print("Warning: AzureOpenAIEmbedding import failed")

# Try to import AzureOpenAI - handle both old and new module paths
try:
    from llama_index.llms.azure_openai import AzureOpenAI
except ImportError:
    # Fallback - will use OpenAI instead
    AzureOpenAI = None  # type: ignore
    print("Warning: AzureOpenAI import failed, will use OpenAI instead")

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

# Import S3 storage (optional - gracefully handle if not configured)
try:
    from aws_storage import S3Storage
    HAS_S3 = True
except ImportError:
    HAS_S3 = False
    logger.warning("S3 storage not available - embeddings will only be stored locally")


class LlamaIndexRAG:
    """RAG system using LlamaIndex with persistent storage.
    
    Supports Azure OpenAI for both embeddings and text generation.
    """
    
    def __init__(
        self,
        persist_dir: str = "rag_storage",
        embedding_model: str = "text-embedding-3-large",
        llm_provider: Optional[str] = None,  # "azure_openai", "openai", or "lm_studio" (auto-detect from env)
        llm_model: str = "gpt-4o-mini",
        lm_studio_base_url: str = "http://127.0.0.1:1234/v1",
        azure_deployment: Optional[str] = None,  # Azure deployment name for LLM
        azure_api_version: str = "2024-02-15-preview",
        embedding_provider: Optional[str] = None,  # "azure" or "openai" (auto-detect from env if None)
        enable_s3_sync: bool = True,  # Enable automatic S3 sync for embeddings
        use_chunking: bool = False,  # Enable text chunking (default: False - one embedding per record)
        chunk_size: int = 2048,  # Size of text chunks for embedding (only if use_chunking=True)
        chunk_overlap: int = 400  # Overlap between consecutive chunks (only if use_chunking=True)
    ):
        """
        Initialize RAG system with LlamaIndex.
        
        Args:
            persist_dir: Directory to persist vector index
            embedding_model: Embedding model name (Azure deployment name if using Azure, or OpenAI model name)
            llm_provider: LLM provider - "azure_openai", "openai", or "lm_studio" (auto-detect from LLM_PROVIDER env if None)
            llm_model: LLM model name
            lm_studio_base_url: Base URL for LM Studio API (if using lm_studio)
            azure_deployment: Azure OpenAI deployment name for LLM (if using azure_openai)
            azure_api_version: Azure OpenAI API version
            embedding_provider: Embedding provider - "azure" or "openai" (auto-detect from EMBEDDING_PROVIDER env if None)
            enable_s3_sync: Enable automatic S3 sync for embeddings (default: True)
            use_chunking: Enable text chunking - if False, one embedding per JSON record (default: False)
            chunk_size: Size of text chunks for embedding (default: 2048 tokens, only if use_chunking=True)
            chunk_overlap: Overlap between consecutive chunks (default: 400 tokens, only if use_chunking=True)
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect providers from environment if not specified
        if llm_provider is None:
            llm_provider = os.getenv("LLM_PROVIDER", "azure_openai").lower()
            # Normalize provider names
            if llm_provider == "azure":
                llm_provider = "azure_openai"
        
        if embedding_provider is None:
            # First check for explicit embedding provider setting
            embedding_provider = os.getenv("EMBEDDING_PROVIDER", "").lower()
            # If not set, use same as LLM provider by default
            if not embedding_provider:
                if llm_provider == "azure_openai":
                    embedding_provider = "azure"
                elif llm_provider == "openai":
                    embedding_provider = "openai"
                else:
                    # For lm_studio or others, default to openai embeddings
                    embedding_provider = "openai"
        
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.enable_s3_sync = enable_s3_sync and HAS_S3
        self.use_chunking = use_chunking
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize S3 storage if enabled
        self.s3_storage = None
        if self.enable_s3_sync:
            try:
                from aws_storage import S3Storage
                self.s3_storage = S3Storage()
                logger.info("✓ S3 sync enabled for RAG embeddings")
            except Exception as e:
                logger.warning(f"S3 storage initialization failed: {e}. Falling back to local-only storage.")
                self.enable_s3_sync = False
        
        # Configure embeddings based on provider
        if self.embedding_provider == "azure":
            # Use Azure OpenAI for embeddings
            if not AzureOpenAIEmbedding:
                raise ImportError(
                    "AzureOpenAIEmbedding is not available. "
                    "Please install: pip install llama-index-embeddings-azure-openai"
                )
            
            azure_embedding_api_key = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
            azure_embedding_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
            azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", embedding_model)
            azure_embedding_version = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-02-15-preview")
            
            if not azure_embedding_api_key or not azure_embedding_endpoint:
                raise ValueError(
                    "Azure OpenAI embeddings configuration missing. "
                    "Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT (or specific AZURE_OPENAI_EMBEDDING_* variants) in your .env file."
                )
            
            Settings.embed_model = AzureOpenAIEmbedding(
                model=azure_embedding_deployment,
                deployment_name=azure_embedding_deployment,
                api_key=azure_embedding_api_key,
                azure_endpoint=azure_embedding_endpoint,
                api_version=azure_embedding_version
            )
            logger.info(f"✓ Using Azure OpenAI embeddings: {azure_embedding_deployment}")
            logger.info(f"  Endpoint: {azure_embedding_endpoint}")
            logger.info(f"  API Version: {azure_embedding_version}")
        else:
            # Use standard OpenAI for embeddings
            openai_api_key = os.getenv("OPENAI_API_KEY")
            
            if not openai_api_key or (not openai_api_key.startswith("sk-") and not openai_api_key.startswith("sk-proj-")):
                raise ValueError(
                    "Valid OPENAI_API_KEY is required for embeddings. "
                    "Please set a real OpenAI API key in your .env file."
                )
            
            Settings.embed_model = OpenAIEmbedding(
                model=embedding_model,
                api_key=openai_api_key
            )
            logger.info(f"✓ Using OpenAI embeddings: {embedding_model}")
        
        # Configure LLM based on provider
        if llm_provider == "azure_openai":
            # Use Azure OpenAI for generation
            if not AzureOpenAI:
                raise ImportError("AzureOpenAI not available. Please install llama-index-llms-azure-openai")
            
            # Get Azure configuration - prioritize chat deployment, then general deployment
            azure_llm_deployment = (
                azure_deployment or 
                os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME") or 
                os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or
                llm_model
            )
            azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            
            if not azure_api_key or not azure_endpoint:
                raise ValueError(
                    "Azure OpenAI configuration missing. "
                    "Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in your .env file."
                )
            
            Settings.llm = AzureOpenAI(
                model=llm_model,
                deployment_name=azure_llm_deployment,
                api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
                temperature=0.7
            )
            self.llm_model_name = azure_llm_deployment
            logger.info(f"✓ Using Azure OpenAI LLM: {self.llm_model_name}")
            logger.info(f"  Endpoint: {azure_endpoint}")
            logger.info(f"  API Version: {azure_api_version}")
        elif llm_provider == "lm_studio":
            # Use LM Studio for generation
            # Store original key
            original_openai_key = os.getenv("OPENAI_API_KEY", "")
            
            # LM Studio uses OpenAI-compatible API, so we use OpenAI class
            # but point it to local server
            # Temporarily set a dummy API key for LM Studio (won't be validated by local server)
            os.environ["OPENAI_API_KEY"] = "sk-111111111111111111111111111111111111111111111111"
            
            Settings.llm = OpenAI(
                model=llm_model,
                api_base=lm_studio_base_url,
                temperature=0.7,
                request_timeout=120.0,
                max_retries=0  # Don't retry on LM Studio
            )
            
            # Restore original key
            if original_openai_key:
                os.environ["OPENAI_API_KEY"] = original_openai_key
            
            self.llm_model_name = llm_model
            logger.info(f"✓ Using LM Studio LLM: {self.llm_model_name}")
        else:
            # Use OpenAI for generation
            openai_api_key = os.getenv("OPENAI_API_KEY")
            
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI LLM provider")
            
            Settings.llm = OpenAI(
                model=llm_model,
                api_key=openai_api_key,
                temperature=0.7
            )
            self.llm_model_name = llm_model
            logger.info(f"✓ Using OpenAI LLM: {self.llm_model_name}")
        
        # Configure text splitter for chunking (only if enabled)
        if self.use_chunking:
            Settings.text_splitter = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            logger.info(f"✓ Text chunking enabled: chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
        else:
            # Disable chunking - use very large chunk size to keep documents whole
            # This ensures one embedding per JSON record (entire document)
            Settings.text_splitter = SentenceSplitter(
                chunk_size=1000000,  # Very large to never split
                chunk_overlap=0
            )
            logger.info("✓ Text chunking disabled - one embedding per JSON record")
        
        self.embedding_model_name = embedding_model
        self.index: Optional[VectorStoreIndex] = None
        self.current_index_name: Optional[str] = None
        self.loaded_indexes: Dict[str, VectorStoreIndex] = {}  # Support multiple loaded indexes
    
    def _detect_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Detect if query requires special handling (date sorting, filtering, etc.).
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with detected intent: {'sort_by_date': bool, 'filter_source': str or None, 'filter_metadata': dict}
        """
        query_lower = query.lower()
        
        intent = {
            'sort_by_date': False,
            'filter_source': None,
            'filter_metadata': {}
        }
        
        # Detect temporal queries
        temporal_keywords = [
            'recent', 'latest', 'newest', 'new', 'last', 'current',
            'up to date', 'up-to-date', 'this year', 'this month',
            'timeline', 'chronological', 'when'
        ]
        
        if any(keyword in query_lower for keyword in temporal_keywords):
            intent['sort_by_date'] = True
        
        # Detect high-impact queries
        if 'high impact' in query_lower or 'high-impact' in query_lower:
            intent['filter_metadata']['impact'] = 'high'
        
        # Detect TRL level queries (e.g., "commercial", "deployed", "production", "research")
        if 'commercial' in query_lower or 'deployed' in query_lower or 'production' in query_lower:
            intent['filter_metadata']['trl'] = '8'  # High TRL
        elif 'research' in query_lower or 'poc' in query_lower or 'proof of concept' in query_lower:
            intent['filter_metadata']['trl'] = '3'  # Low TRL (research)
        
        # Detect sentiment filters
        if 'positive' in query_lower:
            intent['filter_metadata']['sentiment'] = 'positive'
        elif 'negative' in query_lower:
            intent['filter_metadata']['sentiment'] = 'negative'
        
        # Detect source-specific queries
        # Extract source names from loaded indexes
        if self.current_index_name:
            source_name = self.current_index_name.rsplit('_', 1)[0]
            # Check for source name in query (exact or partial match)
            source_name_lower = source_name.lower()
            if source_name_lower in query_lower:
                intent['filter_source'] = source_name
                logger.info(f"Detected source filter for: {source_name}")
            # Also check with underscores replaced by spaces
            elif source_name_lower.replace('_', ' ') in query_lower or source_name_lower.replace('-', ' ') in query_lower:
                intent['filter_source'] = source_name
                logger.info(f"Detected source filter for: {source_name}")
        
        return intent

    def _upload_index_to_s3(self, index_name: str) -> bool:
        """
        Upload persisted index directory to S3.
        
        Args:
            index_name: Name of the index (folder name)
            
        Returns:
            bool: True if successful
        """
        if not self.enable_s3_sync or not self.s3_storage:
            return False
        
        try:
            index_dir = self.persist_dir / index_name
            if not index_dir.exists():
                logger.error(f"Index directory not found: {index_dir}")
                return False
            
            # Create a temporary zip file of the index
            temp_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
            temp_zip_path = temp_zip.name
            temp_zip.close()
            
            # Zip the index directory
            shutil.make_archive(temp_zip_path.replace('.zip', ''), 'zip', index_dir)
            
            # Upload to S3
            s3_key = f"rag_embeddings/{index_name}.zip"
            success = self.s3_storage.upload_file(
                temp_zip_path,
                s3_key,
                metadata={
                    'index_name': index_name,
                    'created_at': datetime.now().isoformat()
                }
            )
            
            # Cleanup temp file
            Path(temp_zip_path).unlink(missing_ok=True)
            
            if success:
                logger.info(f"✓ Index '{index_name}' uploaded to S3: {s3_key}")
            else:
                logger.error(f"✗ Failed to upload index '{index_name}' to S3")
            
            return success
            
        except Exception as e:
            logger.error(f"Error uploading index to S3: {e}")
            return False
    
    def _download_index_from_s3(self, index_name: str) -> bool:
        """
        Download persisted index from S3 to local storage.
        
        Args:
            index_name: Name of the index (folder name)
            
        Returns:
            bool: True if successful
        """
        if not self.enable_s3_sync or not self.s3_storage:
            return False
        
        try:
            s3_key = f"rag_embeddings/{index_name}.zip"
            
            # Check if index exists in S3
            if not self.s3_storage.file_exists(s3_key):
                logger.info(f"Index '{index_name}' not found in S3")
                return False
            
            # Create temp file for download
            temp_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
            temp_zip_path = temp_zip.name
            temp_zip.close()
            
            # Download from S3
            success = self.s3_storage.download_file(s3_key, temp_zip_path)
            if not success:
                logger.error(f"Failed to download index from S3")
                Path(temp_zip_path).unlink(missing_ok=True)
                return False
            
            # Extract zip to local persist directory
            index_dir = self.persist_dir / index_name
            index_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.unpack_archive(temp_zip_path, index_dir, 'zip')
            
            # Cleanup temp file
            Path(temp_zip_path).unlink(missing_ok=True)
            
            logger.info(f"✓ Index '{index_name}' downloaded from S3 and extracted")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading index from S3: {e}")
            return False
    
    def list_s3_indexes(self) -> List[str]:
        """
        List all available indexes in S3.
        
        Returns:
            List of index names
        """
        if not self.enable_s3_sync or not self.s3_storage:
            return []
        
        try:
            files = self.s3_storage.list_files(prefix="rag_embeddings/", suffix=".zip")
            # Extract index names from file paths
            index_names = [Path(f).stem for f in files]
            return index_names
        except Exception as e:
            logger.error(f"Error listing S3 indexes: {e}")
            return []

    
    def build_index_from_json(
        self,
        json_path: str,
        force_rebuild: bool = False,
        progress_callback=None
    ) -> int:
        """
        Build or load vector index from JSON file.
        Automatically syncs with S3 if enabled.
        
        Args:
            json_path: Path to summarized JSON file
            force_rebuild: If True, rebuild even if index exists
            progress_callback: Optional callback for progress updates
            
        Returns:
            Number of documents indexed
        """
        json_path_obj = Path(json_path)
        index_name = json_path_obj.stem  # Use JSON filename as index identifier
        index_dir = self.persist_dir / index_name
        
        # Check if index already exists locally
        if index_dir.exists() and not force_rebuild:
            try:
                # Load existing local index
                if progress_callback:
                    progress_callback("Loading existing local index...", 0, 1)
                
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(index_dir)
                )
                self.index = load_index_from_storage(storage_context)  # type: ignore
                self.current_index_name = index_name
                
                # Count documents by loading docstore
                doc_count = len(storage_context.docstore.docs)
                
                if progress_callback:
                    progress_callback(f"Loaded existing index with {doc_count} documents", doc_count, doc_count)
                
                return doc_count
            except Exception as e:
                logger.warning(f"Failed to load local index: {e}. Checking S3...")
        
        # Try to download from S3 if not found locally
        if not index_dir.exists() and self.enable_s3_sync and not force_rebuild:
            if progress_callback:
                progress_callback("Checking S3 for existing index...", 0, 1)
            
            if self._download_index_from_s3(index_name):
                try:
                    # Load downloaded index
                    storage_context = StorageContext.from_defaults(
                        persist_dir=str(index_dir)
                    )
                    self.index = load_index_from_storage(storage_context)  # type: ignore
                    self.current_index_name = index_name
                    
                    doc_count = len(storage_context.docstore.docs)
                    
                    if progress_callback:
                        progress_callback(f"Loaded index from S3 with {doc_count} documents", doc_count, doc_count)
                    
                    return doc_count
                except Exception as e:
                    logger.warning(f"Failed to load index from S3: {e}. Rebuilding...")
        
        # Build new index
        if progress_callback:
            progress_callback("Loading JSON file...", 0, 100)
        
        with open(json_path_obj, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        documents = []
        total = len(articles)
        
        if progress_callback:
            progress_callback(f"Processing {total} documents...", 0, total)
        
        for i, article in enumerate(articles, 1):
            # Extract metadata from JSON structure (case-insensitive)
            title = article.get('Title') or article.get('title') or article.get('headline') or ''
            url = article.get('URL') or article.get('url') or article.get('link') or ''
            publication_date = article.get('Published Date') or article.get('publication_date') or article.get('date') or article.get('Date') or article.get('pubDate') or ''
            
            metadata = {
                'title': title,
                'url': url,
                'filename': article.get('filename') or article.get('file') or article.get('file_name') or '',
                'publication_date': publication_date,
                'dimension': article.get('Dimension') or article.get('dimension') or '',
                'tech': article.get('Tech') or article.get('tech') or '',
                'trl': str(article.get('TRL') or article.get('trl') or ''),
                'startup': article.get('Start-up') or article.get('Start_up') or article.get('start_up') or '',
                'indicator': article.get('Indicator') or article.get('indicator') or '',
                'impact': article.get('Impact') or article.get('impact') or '',
                'sentiment': article.get('Sentiment') or article.get('sentiment') or '',
                'company': article.get('Company') or article.get('company') or '',
                'categories': article.get('categories') or article.get('Categories') or '',
            }
            
            # Build rich text content from JSON structure
            text_parts = []
            
            # Title
            if title:
                text_parts.append(f"# {title}")
            
            # Metadata section
            text_parts.append("\n**Metadata:**")
            if url:
                text_parts.append(f"- URL: {url}")
            if publication_date:
                text_parts.append(f"- Published: {publication_date}")
            if metadata['company']:
                text_parts.append(f"- Company: {metadata['company']}")
            if metadata['tech']:
                text_parts.append(f"- Technology: {metadata['tech']}")
            if metadata['trl']:
                text_parts.append(f"- TRL: {metadata['trl']}")
            if metadata['impact']:
                text_parts.append(f"- Impact: {metadata['impact']}")
            if metadata['sentiment']:
                text_parts.append(f"- Sentiment: {metadata['sentiment']}")
            if metadata['dimension']:
                text_parts.append(f"- Dimension: {metadata['dimension']}")
            
            # Indicator (tech intelligence summary)
            indicator = article.get('Indicator') or article.get('indicator') or ''
            if indicator:
                text_parts.append(f"\n**Summary (Indicator):**\n{indicator}")
            
            # Key insights
            key_insights = article.get('Key Insights') or article.get('key_insights') or []
            if key_insights:
                if isinstance(key_insights, list):
                    insights_text = "\n".join(f"- {insight}" for insight in key_insights)
                else:
                    insights_text = str(key_insights)
                text_parts.append(f"\n**Key Insights:**\n{insights_text}")
            
            # Relevance/Analysis
            relevance = article.get('Relevance') or article.get('relevance') or ''
            if relevance:
                text_parts.append(f"\n**Relevance:**\n{relevance}")
            
            # Raw content
            raw_content = article.get('content') or article.get('Content') or ''
            if raw_content:
                text_parts.append(f"\n**Full Content:**\n{raw_content}")
            
            # Combine all text
            full_text = "\n".join(text_parts)
            
            # Create Document with rich metadata for filtering
            doc = Document(
                text=full_text,
                metadata=metadata,
                excluded_llm_metadata_keys=['filename'],  # Don't show filename to LLM
            )
            
            documents.append(doc)
            
            if progress_callback and i % 10 == 0:  # Update every 10 docs
                progress_callback(f"Processing document {i}/{total}", i, total)
        
        if progress_callback:
            progress_callback("Creating vector embeddings...", total, total)
        
        # Create index (this generates embeddings)
        self.index = VectorStoreIndex.from_documents(
            documents,
            show_progress=False  # We're handling progress ourselves
        )
        
        # Persist to disk
        index_dir.mkdir(parents=True, exist_ok=True)
        self.index.storage_context.persist(persist_dir=str(index_dir))
        self.current_index_name = index_name
        
        # Save metadata about this index
        metadata_file = index_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'source_file': str(json_path),
                'num_documents': len(documents),
                'created_at': datetime.now().isoformat(),
                'embedding_model': self.embedding_model_name,
                'llm_provider': self.llm_provider,
                'llm_model': self.llm_model_name,
            }, f, indent=2)
        
        # Upload to S3 if enabled
        if self.enable_s3_sync:
            if progress_callback:
                progress_callback("Uploading index to S3...", total, total)
            
            upload_success = self._upload_index_to_s3(index_name)
            if upload_success:
                logger.info(f"✓ Index successfully backed up to S3")
            else:
                logger.warning(f"⚠ Index built locally but S3 upload failed")
        
        if progress_callback:
            progress_callback(f"Index built and saved with {len(documents)} documents", total, total)
        
        return len(documents)
    
    def query(
        self,
        query_text: str,
        top_k: int = 3,
        sort_by_date: Optional[bool] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the vector index and return results with metadata.
        
        Args:
            query_text: User query
            top_k: Number of documents to retrieve
            sort_by_date: If True, sort results by date (most recent first). If None, auto-detect.
            filter_metadata: Optional metadata filters (e.g., {'source': 'canarymedia'})
            
        Returns:
            Dictionary with 'response' (generated answer) and 'retrieved_docs' (source documents)
        """
        if not self.index:
            raise ValueError("Index not built. Call build_index_from_json first.")
        
        # Auto-detect query intent if not explicitly specified
        if sort_by_date is None:
            intent = self._detect_query_intent(query_text)
            sort_by_date = intent['sort_by_date']
            if not filter_metadata and intent['filter_metadata']:
                filter_metadata = intent['filter_metadata']  # Use detected metadata filters
        
        # Import PromptTemplate for later use
        from llama_index.core.prompts import PromptTemplate
        
        # Build source mapping for citations (website names from index)
        source_name = self.current_index_name.rsplit('_', 1)[0] if self.current_index_name else "Source"
        
        # If we need date sorting or filtering, retrieve more documents first
        # Retrieve 5x more documents so we have enough after sorting/filtering
        retrieval_top_k = max(top_k * 5, 15) if (sort_by_date or filter_metadata) else top_k
        
        logger.info(f"Query: '{query_text}' | Retrieving {retrieval_top_k} documents (sort_by_date={sort_by_date}, filters={bool(filter_metadata)})")
        
        # Create retriever
        retriever = self.index.as_retriever(similarity_top_k=retrieval_top_k)
        
        # Retrieve nodes
        nodes = retriever.retrieve(query_text)
        logger.info(f"Retrieved {len(nodes)} nodes from index")
        
        # Apply metadata filtering if specified
        if filter_metadata:
            filtered_nodes = []
            for node in nodes:
                match = True
                for key, value in filter_metadata.items():
                    node_value = node.node.metadata.get(key, '')
                    if isinstance(value, str):
                        # Use fuzzy matching - check if value is a substring (case-insensitive)
                        if value.lower() not in str(node_value).lower():
                            match = False
                            break
                    elif node_value != value:
                        match = False
                        break
                if match:
                    filtered_nodes.append(node)
            
            # Only use filtered nodes if we have matches, otherwise fall back to all retrieved nodes
            if filtered_nodes:
                nodes = filtered_nodes
                logger.info(f"Metadata filtering: {len(filtered_nodes)} of {retrieval_top_k} nodes matched filters {filter_metadata}")
            else:
                logger.info(f"No nodes matched metadata filters {filter_metadata}, using all retrieved nodes")
        
        # Sort by date if requested (check both 'publication_date' and 'date' fields)
        if sort_by_date:
            from dateutil import parser
            
            def parse_date(date_str):
                if not date_str:
                    return datetime.min
                try:
                    # Try to parse the date string
                    return parser.parse(str(date_str))
                except:
                    return datetime.min
            
            nodes.sort(
                key=lambda x: parse_date(
                    x.node.metadata.get('publication_date') or x.node.metadata.get('date') or ''
                ),
                reverse=True  # Most recent first
            )
        
        # Take top K after filtering/sorting
        nodes = nodes[:top_k]
        
        # Build list of available URLs from retrieved documents
        available_urls = []
        for node in nodes:
            url = node.node.metadata.get('url', '')
            title = node.node.metadata.get('title', 'Article')
            if url:
                available_urls.append(f"- {title}: {url}")
        
        available_urls_str = "\n".join(available_urls) if available_urls else "No URLs available"
        
        # Update prompt with actual available URLs
        qa_prompt_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, answer the question.\n\n"
            "IMPORTANT CITATION RULES:\n"
            "1. You MUST cite sources inline within your answer using markdown hyperlink format: [text](URL)\n"
            "2. ONLY use URLs from this list of retrieved documents:\n"
            f"{available_urls_str}\n\n"
            "3. DO NOT create, modify, or hallucinate URLs - use ONLY the exact URLs listed above\n"
            "4. Cite sources immediately after stating facts or information from them\n"
            "5. Format: 'According to [article title](exact_url), the technology...'\n"
            f"6. Example: 'Studies show that solar efficiency has improved ([{source_name}](https://example.com/article))'\n\n"
            "Do NOT just list sources at the end. Integrate citations throughout your answer where you use information.\n\n"
            "Question: {query_str}\n"
            "Answer: "
        )
        
        qa_prompt = PromptTemplate(qa_prompt_str)
        
        # Synthesize response from the filtered/sorted nodes using our custom prompt
        from llama_index.core import get_response_synthesizer
        from llama_index.core.response_synthesizers import ResponseMode
        
        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            text_qa_template=qa_prompt
        )
        response = response_synthesizer.synthesize(query_text, nodes)
        
        # Extract source nodes (retrieved documents)
        retrieved_docs = []
        for idx, node in enumerate(nodes, 1):
            # Get metadata
            metadata = node.node.metadata
            # Add source_index to metadata for single-index queries
            if self.current_index_name:
                metadata['source_index'] = self.current_index_name
            
            retrieved_docs.append({
                'id': idx - 1,
                'score': node.score if hasattr(node, 'score') else 0.0,
                'text': node.node.text,  # type: ignore
                'metadata': metadata,
                'doc_id': node.node.id_
            })
        
        return {
            'response': str(response),
            'retrieved_docs': retrieved_docs,
        }
    
    def get_available_indexes(self) -> List[Dict[str, Any]]:
        """
        Get list of available persisted indexes from local storage and S3.
        
        Returns:
            List of index metadata dictionaries
        """
        indexes = []
        seen_names = set()
        
        # Get local indexes
        if self.persist_dir.exists():
            for index_dir in self.persist_dir.iterdir():
                if index_dir.is_dir():
                    metadata_file = index_dir / "metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            metadata['index_name'] = index_dir.name
                            metadata['index_path'] = str(index_dir)
                            metadata['location'] = 'local'
                            indexes.append(metadata)
                            seen_names.add(index_dir.name)
                        except Exception as e:
                            logger.warning(f"Failed to load metadata for {index_dir.name}: {e}")
        
        # Get S3 indexes (only those not already in local)
        if self.enable_s3_sync:
            s3_index_names = self.list_s3_indexes()
            for s3_name in s3_index_names:
                if s3_name not in seen_names:
                    # Get metadata from S3 if available
                    metadata = {
                        'index_name': s3_name,
                        'location': 's3',
                        'created_at': 'unknown',
                        'num_documents': 'unknown'
                    }
                    
                    # Try to get S3 file metadata
                    try:
                        if self.s3_storage:
                            s3_key = f"rag_embeddings/{s3_name}.zip"
                            file_meta = self.s3_storage.get_file_metadata(s3_key)
                            if file_meta:
                                metadata['created_at'] = file_meta['last_modified'].isoformat()
                                metadata['size'] = file_meta['size']
                    except Exception as e:
                        logger.debug(f"Could not get S3 metadata for {s3_name}: {e}")
                    
                    indexes.append(metadata)
        
        return sorted(indexes, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def load_index(self, index_name: str) -> int:
        """
        Load a specific persisted index.
        Automatically downloads from S3 if not found locally.
        
        Args:
            index_name: Name of the index directory
            
        Returns:
            Number of documents in the index
        """
        index_dir = self.persist_dir / index_name
        
        # Try local first
        if not index_dir.exists():
            # Try downloading from S3
            if self.enable_s3_sync:
                logger.info(f"Index '{index_name}' not found locally. Attempting S3 download...")
                if not self._download_index_from_s3(index_name):
                    raise ValueError(f"Index '{index_name}' not found locally or in S3")
            else:
                raise ValueError(f"Index '{index_name}' not found")
        
        storage_context = StorageContext.from_defaults(
            persist_dir=str(index_dir)
        )
        self.index = load_index_from_storage(storage_context)  # type: ignore
        self.current_index_name = index_name
        
        # Count documents
        doc_count = len(storage_context.docstore.docs)
        
        return doc_count
    
    def delete_index(self, index_name: str, delete_from_s3: bool = True):
        """
        Delete a persisted index from local storage and optionally S3.
        
        Args:
            index_name: Name of the index to delete
            delete_from_s3: If True, also delete from S3 (default: True)
        """
        index_dir = self.persist_dir / index_name
        
        # Delete local copy
        if index_dir.exists():
            shutil.rmtree(index_dir)
            logger.info(f"✓ Deleted local index: {index_name}")
            
            if self.current_index_name == index_name:
                self.index = None
                self.current_index_name = None
            if index_name in self.loaded_indexes:
                del self.loaded_indexes[index_name]
        
        # Delete from S3
        if delete_from_s3 and self.enable_s3_sync and self.s3_storage:
            s3_key = f"rag_embeddings/{index_name}.zip"
            if self.s3_storage.delete_file(s3_key):
                logger.info(f"✓ Deleted S3 index: {index_name}")
            else:
                logger.warning(f"⚠ Failed to delete S3 index: {index_name}")
    
    def load_multiple_indexes(self, index_names: List[str]) -> Dict[str, int]:
        """
        Load multiple indexes simultaneously.
        
        Args:
            index_names: List of index names to load
            
        Returns:
            Dictionary mapping index names to document counts
        """
        results = {}
        
        for index_name in index_names:
            try:
                doc_count = self.load_index(index_name)
                results[index_name] = doc_count
            except Exception as e:
                results[index_name] = f"Error: {e}"
        
        return results
    
    def query_multiple_indexes(
        self,
        query: str,
        index_names: List[str],
        top_k: int = 3,
        sort_by_date: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Query multiple indexes and merge results.
        
        Args:
            query: The query string
            index_names: List of index names to query
            top_k: Number of top results per index
            sort_by_date: If True, sort by date. If None, auto-detect from query.
            
        Returns:
            Dictionary with merged results and metadata
        """
        # Auto-detect query intent if not explicitly specified
        if sort_by_date is None:
            intent = self._detect_query_intent(query)
            sort_by_date = intent['sort_by_date']
        
        all_results = []
        
        for index_name in index_names:
            # Load index if not already loaded
            if index_name not in self.loaded_indexes:
                try:
                    index_dir = self.persist_dir / index_name
                    storage_context = StorageContext.from_defaults(
                        persist_dir=str(index_dir)
                    )
                    self.loaded_indexes[index_name] = load_index_from_storage(storage_context)  # type: ignore
                except Exception as e:
                    continue
            
            # Query this index
            try:
                index = self.loaded_indexes[index_name]
                retriever = index.as_retriever(similarity_top_k=top_k * 2 if sort_by_date else top_k)
                nodes = retriever.retrieve(query)
                
                # Add source information to each node
                for node in nodes:
                    node.node.metadata['source_index'] = index_name
                    all_results.append(node)
            except Exception as e:
                continue
        
        # Sort by date if requested
        if sort_by_date:
            from dateutil import parser
            
            def parse_date(date_str):
                if not date_str:
                    return datetime.min
                try:
                    return parser.parse(str(date_str))
                except:
                    return datetime.min
            
            all_results.sort(
                key=lambda x: parse_date(
                    x.node.metadata.get('publication_date') or x.node.metadata.get('date') or ''
                ),
                reverse=True  # Most recent first
            )
        else:
            # Sort all results by score
            all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Take top results overall
        top_results = all_results[:top_k * len(index_names)]
        
        # Generate response using the merged results
        if top_results:
            # Build source name mapping for citations
            source_citations = {}
            available_urls = []
            
            for node in top_results:
                source_index = node.node.metadata.get('source_index', 'Unknown')
                # Extract website name (remove date suffix)
                website_name = source_index.rsplit('_', 1)[0]
                if website_name not in source_citations:
                    source_citations[website_name] = source_index
                
                # Collect URLs
                url = node.node.metadata.get('url', '')
                title = node.node.metadata.get('title', 'Article')
                if url:
                    available_urls.append(f"- [{website_name}] {title}: {url}")
            
            # Create custom prompt with actual URLs from retrieved documents
            from llama_index.core.prompts import PromptTemplate
            
            source_list = ", ".join([f"[{name}]" for name in source_citations.keys()])
            available_urls_str = "\n".join(available_urls) if available_urls else "No URLs available"
            
            qa_prompt_str = (
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information and not prior knowledge, answer the question.\n\n"
                "IMPORTANT CITATION RULES:\n"
                "1. You MUST cite sources inline within your answer using markdown hyperlink format: [text](URL)\n"
                f"2. Available sources: {source_list}\n"
                "3. ONLY use URLs from this list of retrieved documents:\n"
                f"{available_urls_str}\n\n"
                "4. DO NOT create, modify, or hallucinate URLs - use ONLY the exact URLs listed above\n"
                "5. Cite sources immediately after stating facts or information from them\n"
                "6. Format: 'According to [source name](exact_url), the development...'\n"
                "7. Example: 'Recent research shows ([canarymedia](https://example.com/article)) that carbon capture...'\n\n"
                "Do NOT just list sources at the end. Integrate citations throughout your answer where you use information.\n\n"
                "Question: {query_str}\n"
                "Answer: "
            )
            
            qa_prompt = PromptTemplate(qa_prompt_str)
            
            # Create a query engine from the primary index (or first loaded)
            first_index_name = index_names[0] if index_names else None
            primary_index = self.loaded_indexes.get(first_index_name) if first_index_name else None
            if not primary_index:
                primary_index = self.index
            
            if primary_index:
                query_engine = primary_index.as_query_engine(
                    similarity_top_k=len(top_results)
                )
                query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})
                response = query_engine.query(query)
                
                return {
                    'response': str(response),
                    'source_nodes': top_results,
                    'indexes_queried': index_names
                }
        
        return {
            'response': "No results found across the selected indexes.",
            'source_nodes': [],
            'indexes_queried': index_names
        }
