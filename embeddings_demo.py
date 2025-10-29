import os
import json
import openai
from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core.schema import MetadataMode

# Load environment variables from .env file
load_dotenv()

# Load OPENAI_API_KEY from environment
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Please set OPENAI_API_KEY in your .env file")

# Load the summarized data
with open('summarised_content/thinkgeoenergy_20251028.json', 'r') as f:
    articles = json.load(f)

# Create documents with structured metadata
documents = []
for article in articles:
    # Create metadata dictionary
    metadata = {
        "filename": article["filename"],
        "url": article["url"],
        "title": article["title"],
        "date": article["date"],
        "dimension": article["Dimension"],
        "tech": article["Tech"],
        "trl": article["TRL"],
        "startup": article["Start-up"]
    }
    
    # Combine content with indicator
    full_content = f"Indicator:\n{article['Indicator']}\n\nRaw Content:\n{article['content']}"
    
    # Create document with metadata
    document = Document(
        text=full_content,
        metadata=metadata,
        excluded_llm_metadata_keys=["filename", "url"],  # Example of excluding certain metadata from LLM
        metadata_separator="\n",
        metadata_template="{key}: {value}",
        text_template="Metadata:\n{metadata_str}\n-----\nContent:\n{content}"
    )
    documents.append(document)

# Print example of what the LLM and embedding model see
if documents:
    print("\nThe LLM sees this for the first document:\n")
    print(documents[0].get_content(metadata_mode=MetadataMode.LLM))
    print("\nThe Embedding model sees this for the first document:\n")
    print(documents[0].get_content(metadata_mode=MetadataMode.EMBED))

# Example of how to get embeddings using OpenAI
from openai import OpenAI

client = OpenAI()  # This will automatically use your OPENAI_API_KEY

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Example of getting embeddings for the first document
if documents:
    embedding_text = documents[0].get_content(metadata_mode=MetadataMode.EMBED)
    print("\nGenerating embedding for first document...")
    embedding = get_embedding(embedding_text)
    print(f"Generated embedding of length: {len(embedding)}")
    # Print first few values of the embedding vector
    print("\nFirst few values of the embedding vector:")
    print(embedding[:5])