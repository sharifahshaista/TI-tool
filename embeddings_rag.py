"""
LlamaIndex RAG with Persistent Vector Storage
Handles embedding generation, storage, and retrieval using LlamaIndex.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


class LlamaIndexRAG:
    """RAG system using LlamaIndex with persistent storage."""
    
    def __init__(
        self,
        persist_dir: str = "rag_storage",
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini"
    ):
        """
        Initialize RAG system with LlamaIndex.
        
        Args:
            persist_dir: Directory to persist vector index
            embedding_model: OpenAI embedding model name
            llm_model: OpenAI LLM model name
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure LlamaIndex settings
        Settings.embed_model = OpenAIEmbedding(
            model=embedding_model,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        Settings.llm = OpenAI(
            model=llm_model,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
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

    
    def build_index_from_json(
        self,
        json_path: str,
        force_rebuild: bool = False,
        progress_callback=None
    ) -> int:
        """
        Build or load vector index from JSON file.
        
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
        
        # Check if index already exists
        if index_dir.exists() and not force_rebuild:
            try:
                # Load existing index
                if progress_callback:
                    progress_callback("Loading existing index...", 0, 1)
                
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
                print(f"Failed to load existing index: {e}. Rebuilding...")
        
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
                'date': article.get('date') or article.get('Date') or article.get('pubDate') or '',
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
                'embedding_model': Settings.embed_model.model_name,
                'llm_model': Settings.llm.model,
            }, f, indent=2)
        
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
        Get list of available persisted indexes.
        
        Returns:
            List of index metadata dictionaries
        """
        indexes = []
        
        if not self.persist_dir.exists():
            return indexes
        
        for index_dir in self.persist_dir.iterdir():
            if index_dir.is_dir():
                metadata_file = index_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        metadata['index_name'] = index_dir.name
                        metadata['index_path'] = str(index_dir)
                        indexes.append(metadata)
                    except Exception as e:
                        print(f"Failed to load metadata for {index_dir.name}: {e}")
        
        return sorted(indexes, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def load_index(self, index_name: str) -> int:
        """
        Load a specific persisted index.
        
        Args:
            index_name: Name of the index directory
            
        Returns:
            Number of documents in the index
        """
        index_dir = self.persist_dir / index_name
        
        if not index_dir.exists():
            raise ValueError(f"Index '{index_name}' not found")
        
        storage_context = StorageContext.from_defaults(
            persist_dir=str(index_dir)
        )
        self.index = load_index_from_storage(storage_context)
        self.current_index_name = index_name
        
        # Count documents
        doc_count = len(storage_context.docstore.docs)
        
        return doc_count
    
    def delete_index(self, index_name: str):
        """Delete a persisted index."""
        index_dir = self.persist_dir / index_name
        
        if index_dir.exists():
            import shutil
            shutil.rmtree(index_dir)
            if self.current_index_name == index_name:
                self.index = None
                self.current_index_name = None
            if index_name in self.loaded_indexes:
                del self.loaded_indexes[index_name]
    
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
