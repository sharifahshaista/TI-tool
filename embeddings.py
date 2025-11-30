import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pickle

class CSVEmbeddingProcessor:
    """Simplified JSON processor without vector database - uses basic text search."""
    
    def __init__(self, csv_folder: str, collection_name: str = "carbon_articles"):
        self.csv_folder = Path(csv_folder)
        self.collection_name = collection_name
        self.documents = []
    
    def process_all(self, batch_size: int = 100):
        """Load all JSON files into memory."""
        self.documents = []
        json_files = list(Path(self.csv_folder).glob("*.json"))
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # JSON files contain arrays of documents
                if isinstance(data, list):
                    self.documents.extend(data)
                else:
                    self.documents.append(data)
        
        print(f"Loaded {len(self.documents)} documents from {len(json_files)} JSON files")
        return len(self.documents)
    
    def save_to_file(self, file_path: str):
        """Save documents to a pickle file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self.documents, f)
        print(f"Saved {len(self.documents)} documents to {file_path}")
    
    def load_from_file(self, file_path: str):
        """Load documents from a pickle file."""
        with open(file_path, 'rb') as f:
            self.documents = pickle.load(f)
        print(f"Loaded {len(self.documents)} documents from {file_path}")
        return len(self.documents)
    
    def query(self, query_text: str, n_results: int = 5, filters: Optional[Dict] = None):
        """Simple keyword-based search through documents."""
        query_lower = query_text.lower()
        results = []
        
        for doc in self.documents:
            # Simple text matching in key fields
            searchable_text = " ".join([
                str(doc.get('title', '')),
                str(doc.get('Indicator', '')),
                str(doc.get('categories', '')),
                str(doc.get('Tech', '')),
                str(doc.get('Dimension', ''))
            ]).lower()
            
            if any(word in searchable_text for word in query_lower.split()):
                results.append(doc)
        
        # Return top n_results
        results = results[:n_results]
        
        # Format to match expected ChromaDB output structure
        return {
            "ids": [[str(i) for i in range(len(results))]],
            "documents": [[str(doc) for doc in results]],
            "metadatas": [[{
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "publication_date": doc.get("publication_date", doc.get("date", "")),
                "categories": doc.get("categories", ""),
                "dimension": doc.get("Dimension", ""),
                "tech": doc.get("Tech", ""),
                "trl": doc.get("TRL", ""),
                "startup": doc.get("URL to start-ups", ""),
                "full_indicator": doc.get("Indicator", "")
            } for doc in results]]
        }
