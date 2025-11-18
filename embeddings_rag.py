"""
LlamaIndex RAG with Persistent Vector Storage
Handles embedding generation, storage, and retrieval using LlamaIndex.
Supports S3 backup for embeddings persistence.
"""

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
    try:
        # Try newer import path
        from llama_index.llms.openai import AzureOpenAI
    except ImportError:
        # Fallback - will use OpenAI instead
        AzureOpenAI = None
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
        llm_provider: str = "azure_openai",  # "azure_openai", "openai", or "lm_studio"
        llm_model: str = "gpt-4o-mini",
        lm_studio_base_url: str = "http://127.0.0.1:1234/v1",
        azure_deployment: str = None,  # Azure deployment name for LLM
        azure_api_version: str = "2024-02-15-preview",
        use_azure_embeddings: bool = True,  # Use Azure OpenAI for embeddings
        enable_s3_sync: bool = True  # Enable automatic S3 sync for embeddings
    ):
        """
        Initialize RAG system with LlamaIndex.
        
        Args:
            persist_dir: Directory to persist vector index
            embedding_model: Embedding model name (Azure deployment name if use_azure_embeddings=True)
            llm_provider: LLM provider - "azure_openai", "openai", or "lm_studio"
            llm_model: LLM model name
            lm_studio_base_url: Base URL for LM Studio API (if using lm_studio)
            azure_deployment: Azure OpenAI deployment name for LLM (if using azure_openai)
            azure_api_version: Azure OpenAI API version
            use_azure_embeddings: Use Azure OpenAI for embeddings (default: True)
            enable_s3_sync: Enable automatic S3 sync for embeddings (default: True)
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.llm_provider = llm_provider
        self.use_azure_embeddings = use_azure_embeddings
        self.enable_s3_sync = enable_s3_sync and HAS_S3
        
        # Initialize S3 storage if enabled
        self.s3_storage = None
        if self.enable_s3_sync:
            try:
                self.s3_storage = S3Storage()
                logger.info("✓ S3 sync enabled for RAG embeddings")
            except Exception as e:
                logger.warning(f"S3 storage initialization failed: {e}. Falling back to local-only storage.")
                self.enable_s3_sync = False
        
        # Configure embeddings
        if use_azure_embeddings:
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
                    "Please set AZURE_OPENAI_EMBEDDING_API_KEY and AZURE_OPENAI_EMBEDDING_ENDPOINT in your .env file."
                )
            
            # Use AzureOpenAIEmbedding class for Azure
            Settings.embed_model = AzureOpenAIEmbedding(
                model=azure_embedding_deployment,
                deployment_name=azure_embedding_deployment,
                api_key=azure_embedding_api_key,
                azure_endpoint=azure_embedding_endpoint,
                api_version=azure_embedding_version
            )
            print(f"✓ Using Azure OpenAI embeddings: {azure_embedding_deployment}")
            print(f"  Endpoint: {azure_embedding_endpoint}")
            print(f"  API Version: {azure_embedding_version}")
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
            print(f"✓ Using OpenAI embeddings: {embedding_model}")
        
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
            print(f"✓ Using Azure OpenAI LLM: {self.llm_model_name}")
            print(f"  Endpoint: {azure_endpoint}")
            print(f"  API Version: {azure_api_version}")
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
            print(f"✓ Using LM Studio LLM: {self.llm_model_name}")
        else:
            # Use OpenAI for generation
            Settings.llm = OpenAI(
                model=llm_model,
                api_key=original_openai_key
            )
            self.llm_model_name = llm_model
        
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
            Dictionary with detected intent: {'sort_by_date': bool, 'filter_source': str or None}
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
        
        # Detect source-specific queries
        # Extract source names from loaded indexes
        if self.current_index_name:
            source_name = self.current_index_name.rsplit('_', 1)[0]
            if source_name in query_lower:
                intent['filter_source'] = source_name
        
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
        json_path = Path(json_path)
        index_name = json_path.stem  # Use JSON filename as index identifier
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
                self.index = load_index_from_storage(storage_context)
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
                    self.index = load_index_from_storage(storage_context)
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
        
        with open(json_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        documents = []
        total = len(articles)
        
        if progress_callback:
            progress_callback(f"Processing {total} documents...", 0, total)
        
        for i, article in enumerate(articles, 1):
            # Extract metadata
            metadata = {
                'title': article.get('title') or article.get('Title') or article.get('headline') or '',
                'url': article.get('url') or article.get('link') or '',
                'filename': article.get('filename') or article.get('file') or article.get('file_name') or '',
                'publication_date': article.get('publication_date') or article.get('date') or article.get('Date') or article.get('pubDate') or '',
                'dimension': article.get('Dimension') or article.get('dimension') or '',
                'tech': article.get('Tech') or article.get('tech') or '',
                'trl': str(article.get('TRL') or article.get('trl') or ''),
                'startup': article.get('Start-up') or article.get('Start_up') or article.get('start_up') or '',
                'indicator': article.get('Indicator') or article.get('indicator') or '',
            }
            
            # Build content with Indicator and raw content
            indicator = article.get('Indicator', '')
            raw_content = article.get('content', '')
            
            # Create full text
            full_text = f"Indicator:\n{indicator}\n\nRaw Content:\n{raw_content}"
            
            # Create Document
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
        sort_by_date: bool = None,
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
            if not filter_metadata and intent['filter_source']:
                filter_metadata = {}  # Will be handled by source filtering
        
        # Create query engine with custom prompt
        from llama_index.core.prompts import PromptTemplate
        
        # Build source mapping for citations (website names from index)
        source_name = self.current_index_name.rsplit('_', 1)[0] if self.current_index_name else "Source"
        
        qa_prompt_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the question. You MUST cite your sources using markdown hyperlink format: [Source Name](URL). "
            f"Each context document contains metadata with a 'url' field that you should use. "
            "For example: 'According to [" + source_name + "](https://example.com/article), the technology...' "
            "Always include source citations with hyperlinks when referencing information. "
            "Use the actual URL from the document's metadata.\n\n"
            "Question: {query_str}\n"
            "Answer: "
        )
        
        qa_prompt = PromptTemplate(qa_prompt_str)
        
        # If we need date sorting or filtering, retrieve more documents first
        retrieval_top_k = top_k * 3 if (sort_by_date or filter_metadata) else top_k
        
        # Create retriever
        retriever = self.index.as_retriever(similarity_top_k=retrieval_top_k)
        
        # Retrieve nodes
        nodes = retriever.retrieve(query_text)
        
        # Apply metadata filtering if specified
        if filter_metadata:
            filtered_nodes = []
            for node in nodes:
                match = True
                for key, value in filter_metadata.items():
                    node_value = node.node.metadata.get(key, '')
                    if isinstance(value, str):
                        if value.lower() not in str(node_value).lower():
                            match = False
                            break
                    elif node_value != value:
                        match = False
                        break
                if match:
                    filtered_nodes.append(node)
            nodes = filtered_nodes
        
        # Sort by date if requested
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
                key=lambda x: parse_date(x.node.metadata.get('date', '')),
                reverse=True  # Most recent first
            )
        
        # Take top K after filtering/sorting
        nodes = nodes[:top_k]
        
        # Create query engine and synthesize response
        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact",
        )
        
        # Update prompt
        query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})
        
        # Synthesize response from the filtered/sorted nodes
        from llama_index.core.schema import NodeWithScore
        from llama_index.core import get_response_synthesizer
        
        response_synthesizer = get_response_synthesizer(response_mode="compact")
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
                'text': node.node.text,
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
        self.index = load_index_from_storage(storage_context)
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
        if delete_from_s3 and self.enable_s3_sync:
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
        sort_by_date: bool = None
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
                    self.loaded_indexes[index_name] = load_index_from_storage(storage_context)
                except Exception as e:
                    continue
            
            # Query this index
            try:
                index = self.loaded_indexes[index_name]
                retriever = index.as_retriever(similarity_top_k=top_k * 2 if sort_by_date else top_k)
                nodes = retriever.retrieve(query)
                
                # Add source information to each node
                for node in nodes:
                    node.metadata['source_index'] = index_name
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
                key=lambda x: parse_date(x.metadata.get('date', '')),
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
            for node in top_results:
                source_index = node.metadata.get('source_index', 'Unknown')
                # Extract website name (remove date suffix)
                website_name = source_index.rsplit('_', 1)[0]
                if website_name not in source_citations:
                    source_citations[website_name] = source_index
            
            # Create custom prompt with source names
            from llama_index.core.prompts import PromptTemplate
            
            source_list = ", ".join([f"[{name}]" for name in source_citations.keys()])
            
            qa_prompt_str = (
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information and not prior knowledge, "
                "answer the question. You MUST cite your sources using markdown hyperlink format: [Source Name](URL). "
                f"The available sources are: {source_list}. "
                "Each context document contains metadata with a 'url' field that you should use for the hyperlink. "
                "For example: 'According to [canarymedia](https://example.com/article), the technology...' "
                "Use the appropriate source name and URL based on which document the information comes from. "
                "Always include source citations with hyperlinks when referencing information.\n\n"
                "Question: {query_str}\n"
                "Answer: "
            )
            
            qa_prompt = PromptTemplate(qa_prompt_str)
            
            # Create a query engine from the primary index (or first loaded)
            primary_index = self.loaded_indexes.get(
                index_names[0] if index_names else None
            ) or self.index
            
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
