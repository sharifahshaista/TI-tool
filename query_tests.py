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
    from aws_storage import S3Storage
    HAS_S3 = True
except ImportError:
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
        
        self.client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1") # type: ignore
        
        # Initialize S3 storage
        self.s3_storage = None
        if self.enable_s3_sync:
            try:
                self.s3_storage = S3Storage() # type: ignore
                logger.info("✓ S3 sync enabled for embeddings")
            except Exception as e:
                logger.warning(f"S3 storage initialization failed: {e}")
                self.enable_s3_sync = False
    
    def _parse_trl(self, trl_value) -> float:
        """
        Parse TRL value, handling 'N/A', ranges like '6-7', and other non-numeric values.
        
        Args:
            trl_value: TRL value from JSON (string or numeric)
            
        Returns:
            Float TRL value, or 0.0 if invalid
        """
        if not trl_value:
            return 0.0
        
        if isinstance(trl_value, (int, float)):
            return float(trl_value)
        
        if isinstance(trl_value, str):
            # Handle common non-numeric values
            if trl_value.upper() in ['N/A', 'NA', 'NULL', 'NONE', '']:
                return 0.0
            
            # Handle ranges like '6-7'
            if '-' in trl_value:
                parts = trl_value.split('-')
                if len(parts) == 2:
                    try:
                        start = float(parts[0].strip())
                        end = float(parts[1].strip())
                        # Return average of range
                        return (start + end) / 2.0
                    except ValueError:
                        pass
            
            # Try direct conversion
            try:
                return float(trl_value)
            except ValueError:
                return 0.0
        
        return 0.0
    
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
    
    def create_embedding_text(self, entry: Dict[str, Any], chunk_text: str = None) -> str: # type: ignore
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
Start-up: {entry.get('URL to start-up(s)', '')}
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
        response = self.client.embeddings.create(
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
                "trl": self._parse_trl(entry.get("TRL", 0.0)),
                "startup": entry.get("URL to start-up(s)", ""),
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
        source_name = self.extract_source_name(json_path_obj.name)
        index_name = f"{source_name}_embeddings"
        
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
    
    def extract_source_name(self, filename: str) -> str:
        """
        Extract and format source name from filename.
        
        Args:
            filename: Filename like "techcrunch_com_20251127.json"
            
        Returns:
            Formatted source name like "TechCrunch"
        """
        # Remove .json extension and split by underscores
        stem = Path(filename).stem
        parts = stem.split('_')
        
        # Take first part and capitalize it
        if parts:
            source = parts[0].replace('com', '').replace('org', '').replace('net', '')
            # Capitalize first letter
            return source.capitalize()
        
        return stem
    
    def process_summarised_content_directory(
        self,
        directory_path: str = "./summarised_content",
        progress_callback=None
    ) -> Dict[str, EmbeddingIndex]:
        """
        Process all JSON files in the summarised_content directory.
        
        Args:
            directory_path: Path to directory containing JSON files
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping source names to EmbeddingIndex objects
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        json_files = list(directory.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {directory_path}")
        
        embedding_indexes = {}
        
        for json_file in json_files:
            try:
                source_name = self.extract_source_name(json_file.name)
                
                if progress_callback:
                    progress_callback(f"Processing {source_name}...", 0, 100)
                
                # Process the JSON file
                embedding_index = self.process_json_file(
                    str(json_file),
                    progress_callback=progress_callback
                )
                
                # Save locally
                self.save_index_locally(embedding_index)
                
                # Upload to S3 if enabled
                if self.enable_s3_sync:
                    self.upload_index_to_s3(embedding_index)
                
                embedding_indexes[source_name] = embedding_index
                
                logger.info(f"✓ Processed {source_name}: {embedding_index.num_chunks} chunks from {embedding_index.num_documents} documents")
                
            except Exception as e:
                logger.error(f"Error processing {json_file.name}: {e}")
                continue
        
        return embedding_indexes


def run_test_queries(processor: JSONEmbeddingProcessor, embedding_index: EmbeddingIndex):
    """
    Run comprehensive test queries to validate retrieval accuracy.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST QUERIES FOR RETRIEVAL VALIDATION")
    print("="*80)
    
    test_cases = [
        {
            "name": "1. Exact Company Match",
            "query": "MORFO reforestation technology",
            "description": "Should retrieve articles specifically mentioning MORFO startup",
            "expected": "High scores for MORFO-related articles"
        },
        {
            "name": "2. Technology-Specific Query",
            "query": "AI drone technology for forest restoration",
            "description": "Should match technology field and indicator content",
            "expected": "Articles about AI/drone restoration tech"
        },
        {
            "name": "3. Geographic Focus",
            "query": "Amazon rainforest conservation projects",
            "description": "Should retrieve geographically relevant articles",
            "expected": "Amazon/Brazil-focused projects"
        },
        {
            "name": "4. Categorical Query",
            "query": "carbon markets and carbon credits trading",
            "description": "Should match articles in carbon markets category",
            "expected": "Carbon market related articles"
        },
        {
            "name": "5. Multi-Concept Query",
            "query": "large scale reforestation with carbon offset programs",
            "description": "Should balance multiple concepts (reforestation + carbon)",
            "expected": "Articles covering both concepts"
        },
        {
            "name": "6. Technical Detail Query",
            "query": "revegetation projects covering thousands of hectares",
            "description": "Should match specific technical details in indicators",
            "expected": "Large-scale project articles"
        },
        {
            "name": "7. Stakeholder Query",
            "query": "partnerships between companies and carbon buyers",
            "description": "Should find business relationship mentions",
            "expected": "Articles about partnerships/buyers"
        },
        {
            "name": "8. Innovation Query",
            "query": "novel approaches to biodiversity restoration",
            "description": "Should retrieve innovative/unique solutions",
            "expected": "Innovative restoration methods"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'─'*80}")
        print(f"TEST: {test_case['name']}")
        print(f"Query: '{test_case['query']}'")
        print(f"Description: {test_case['description']}")
        print(f"Expected: {test_case['expected']}")
        print(f"{'─'*80}")
        
        results = processor.query_similar(
            embedding_index,
            test_case['query'],
            top_k=3
        )
        
        if not results:
            print("⚠ WARNING: No results found!")
            continue
        
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i} (Score: {result['score']:.4f})")
            print(f"  ├─ Title: {result['metadata']['title'][:80]}...")
            print(f"  ├─ URL: {result['metadata']['url']}")
            print(f"  ├─ Tech: {result['metadata']['tech']}")
            print(f"  ├─ Dimension: {result['metadata']['dimension']}")
            print(f"  ├─ Startup: {result['metadata']['startup']}")
            print(f"  ├─ Categories: {result['metadata']['categories']}")
            print(f"  └─ Chunk: {result['metadata']['chunk_index']+1}/{result['metadata']['total_chunks']}")
            
            # Show snippet of matched text
            text_snippet = result['text'][:200].replace('\n', ' ')
            print(f"     Preview: {text_snippet}...")
        
        # Validate score distribution
        scores = [r['score'] for r in results]
        print(f"\n  📊 Score Analysis:")
        print(f"     Highest: {max(scores):.4f}")
        print(f"     Lowest: {min(scores):.4f}")
        print(f"     Range: {max(scores) - min(scores):.4f}")
        
        if max(scores) < 0.5:
            print(f"  ⚠ WARNING: Low similarity scores - query may not match content well")
        elif max(scores) - min(scores) < 0.05:
            print(f"  ⚠ WARNING: Small score range - results may not be well differentiated")
        else:
            print(f"  ✓ Good score distribution")


def run_metadata_filter_tests(processor: JSONEmbeddingProcessor, embedding_index: EmbeddingIndex):
    """
    Test metadata filtering functionality.
    """
    print("\n" + "="*80)
    print("METADATA FILTER TESTS")
    print("="*80)
    
    filter_tests = [
        {
            "name": "Filter by TRL Level",
            "query": "carbon capture technology",
            "filter": {"trl": 7.0},
            "description": "Only TRL 7 technologies"
        },
        {
            "name": "Filter by Dimension",
            "query": "environmental technology innovations",
            "filter": {"dimension": "Tech"},
            "description": "Only 'Tech' dimension articles"
        },
        {
            "name": "Filter by Tech Category",
            "query": "market developments",
            "filter": {"tech": "Carbon Markets"},
            "description": "Only Carbon Markets category"
        }
    ]
    
    for test in filter_tests:
        print(f"\n{'─'*80}")
        print(f"TEST: {test['name']}")
        print(f"Query: '{test['query']}'")
        print(f"Filter: {test['filter']}")
        print(f"Description: {test['description']}")
        print(f"{'─'*80}")
        
        results = processor.query_similar(
            embedding_index,
            test['query'],
            top_k=5,
            filter_metadata=test['filter']
        )
        
        print(f"\n  Found {len(results)} results matching filter")
        
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i} (Score: {result['score']:.4f})")
            print(f"  ├─ Title: {result['metadata']['title'][:70]}...")
            print(f"  ├─ Tech: {result['metadata']['tech']}")
            print(f"  ├─ TRL: {result['metadata']['trl']}")
            print(f"  └─ Dimension: {result['metadata']['dimension']}")
            
            # Verify filter was applied
            for key, expected_value in test['filter'].items():
                actual_value = result['metadata'].get(key)
                if actual_value == expected_value:
                    print(f"     ✓ Filter matched: {key}={expected_value}")
                else:
                    print(f"     ✗ Filter FAILED: {key}={actual_value} (expected {expected_value})")


def run_edge_case_tests(processor: JSONEmbeddingProcessor, embedding_index: EmbeddingIndex):
    """
    Test edge cases and potential issues.
    """
    print("\n" + "="*80)
    print("EDGE CASE TESTS")
    print("="*80)
    
    edge_cases = [
        {
            "name": "Very Short Query",
            "query": "AI",
            "description": "Single word query"
        },
        {
            "name": "Very Long Query",
            "query": "I am looking for comprehensive information about large-scale reforestation initiatives that leverage artificial intelligence and drone technology for rapid deployment across degraded landscapes in tropical biomes, particularly focusing on projects that generate carbon credits and have partnerships with corporate buyers interested in nature-based solutions",
            "description": "Long descriptive query"
        },
        {
            "name": "Technical Jargon",
            "query": "ARR projects with high additionality and permanence guarantees",
            "description": "Domain-specific terminology"
        },
        {
            "name": "Ambiguous Query",
            "query": "green technology",
            "description": "Vague/broad query"
        },
        {
            "name": "Non-Existent Concept",
            "query": "underwater solar panel installations in Antarctica",
            "description": "Query unlikely to match any content"
        }
    ]
    
    for test in edge_cases:
        print(f"\n{'─'*80}")
        print(f"TEST: {test['name']}")
        print(f"Query: '{test['query'][:100]}{'...' if len(test['query']) > 100 else ''}'")
        print(f"Description: {test['description']}")
        print(f"{'─'*80}")
        
        results = processor.query_similar(
            embedding_index,
            test['query'],
            top_k=3
        )
        
        if results:
            print(f"\n  Found {len(results)} results")
            print(f"  Top score: {results[0]['score']:.4f}")
            print(f"  Top result: {results[0]['metadata']['title'][:80]}...")
            
            if results[0]['score'] < 0.3:
                print(f"  ⚠ Very low similarity - query may not match content")
        else:
            print(f"\n  ⚠ No results found")


def run_chunking_validation(processor: JSONEmbeddingProcessor, embedding_index: EmbeddingIndex):
    """
    Validate that chunking is working correctly for long indicators.
    """
    print("\n" + "="*80)
    print("CHUNKING VALIDATION")
    print("="*80)
    
    # Find records with multiple chunks
    multi_chunk_records = {}
    for chunk in embedding_index.chunks:
        if chunk.total_chunks > 1:
            if chunk.record_id not in multi_chunk_records:
                multi_chunk_records[chunk.record_id] = []
            multi_chunk_records[chunk.record_id].append(chunk)
    
    print(f"\nFound {len(multi_chunk_records)} records with multiple chunks")
    print(f"Total chunks across all records: {embedding_index.num_chunks}")
    print(f"Single-chunk records: {embedding_index.num_documents - len(multi_chunk_records)}")
    
    if multi_chunk_records:
        # Show example of chunked record
        example_record_id = list(multi_chunk_records.keys())[0]
        example_chunks = sorted(multi_chunk_records[example_record_id], key=lambda x: x.chunk_index)
        
        print(f"\n{'─'*80}")
        print(f"EXAMPLE CHUNKED RECORD")
        print(f"Record ID: {example_record_id}")
        print(f"Total Chunks: {len(example_chunks)}")
        print(f"{'─'*80}")
        
        for chunk in example_chunks:
            print(f"\n  Chunk {chunk.chunk_index + 1}/{chunk.total_chunks}")
            print(f"  Title: {chunk.metadata['title'][:70]}...")
            print(f"  Text length: {len(chunk.text)} chars")
            print(f"  Embedding dim: {len(chunk.embedding)}")
            print(f"  Preview: {chunk.text[:150].replace(chr(10), ' ')}...")
        
        # Test querying specific chunk content
        print(f"\n{'─'*80}")
        print(f"QUERY CHUNKS TEST")
        print(f"{'─'*80}")
        
        # Extract a distinctive phrase from middle chunk
        if len(example_chunks) > 1:
            middle_chunk = example_chunks[len(example_chunks)//2]
            # Get a snippet from the chunk content
            words = middle_chunk.metadata['full_indicator'].split()
            if len(words) > 10:
                query_phrase = ' '.join(words[5:15])  # Extract middle section
                
                print(f"\nQuery from chunk {middle_chunk.chunk_index + 1}: '{query_phrase}'")
                
                results = processor.query_similar(
                    embedding_index,
                    query_phrase,
                    top_k=5
                )
                
                # Check if correct chunks are retrieved
                retrieved_record_ids = [r['record_id'] for r in results]
                if example_record_id in retrieved_record_ids:
                    rank = retrieved_record_ids.index(example_record_id) + 1
                    print(f"✓ Correct record found at rank {rank}")
                else:
                    print(f"✗ WARNING: Correct record not in top 5 results")


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
    
    # RUN COMPREHENSIVE TESTS
    print("\n\n" + "="*80)
    print("STARTING COMPREHENSIVE RETRIEVAL TESTS")
    print("="*80)
    
    # 1. Basic query tests
    run_test_queries(processor, embedding_index)
    
    # 2. Metadata filter tests
    run_metadata_filter_tests(processor, embedding_index)
    
    # 3. Edge case tests
    run_edge_case_tests(processor, embedding_index)
    
    # 4. Chunking validation
    run_chunking_validation(processor, embedding_index)
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)
    
    # List available indexes
    print("\n--- Available Indexes ---")
    print("Local:", processor.list_local_indexes())
    if processor.enable_s3_sync:
        print("S3:", processor.list_s3_indexes())