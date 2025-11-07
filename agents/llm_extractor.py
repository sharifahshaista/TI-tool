import os
import json
import time
import pandas as pd
from pathlib import Path
from typing import List, Optional, Callable
from pydantic import BaseModel, Field, ValidationError
import re

# Support for multiple LLM providers
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Pydantic model for LLM output
class ArticleExtraction(BaseModel):
    url: Optional[str] = None
    title: Optional[str] = None
    publication_date: Optional[str] = None
    main_content: str = ""
    categories: List[str] = Field(default_factory=list)

# Helper function to extract JSON from LLM response
def parse_llm_json(response_text: str):
    """Extract the first JSON object from the LLM response."""
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None

def get_openai_client(provider: str = "openai", base_url: str = None, api_key: str = None):
    """
    Get OpenAI client based on provider configuration.
    
    Args:
        provider: "openai", "azure", or "lm_studio"
        base_url: Base URL for API (for LM Studio or custom endpoints)
        api_key: API key (optional for LM Studio)
    
    Returns:
        OpenAI client instance
    """
    if OpenAI is None:
        raise ImportError("openai package not installed. Install with: pip install openai")
    
    provider = provider.lower()
    
    if provider == "lm_studio":
        # LM Studio uses OpenAI-compatible API
        return OpenAI(
            base_url=base_url or os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
            api_key=api_key or os.getenv("LM_STUDIO_API_KEY", "lm-studio")
        )
    elif provider == "azure":
        # Azure OpenAI
        from openai import AzureOpenAI
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION", "2023-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    else:
        # Standard OpenAI
        return OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

def llm_extract(client, model_name: str, source: str):
    """Use LLM to extract structured fields from markdown text."""
    prompt = f"""
You are an expert at extracting structured information from markdown documents.

Extract the following fields from the provided markdown content. Return ONLY what is explicitly present.

**OUTPUT FORMAT:**
Return a valid JSON object ONLY with these exact keys:
{{
"url": "string or null",
"title": "string or null",
"publication_date": "string or null",
"main_content": "string",
"categories": ["array of strings"]
}}

**INSTRUCTIONS:**
- url: Extract the source URL if mentioned
- title: Extract the article title
- publication_date: Extract the publication date in YYYY-MM-DD format if available
- main_content: Extract the main article content (remove navigation, ads, footers)
- categories: Extract relevant topic categories (e.g., ["Technology", "Renewable Energy"])

**SOURCE CONTENT:**
{source}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )

    result_text = response.choices[0].message.content.strip()

    # Extract JSON safely
    result_json = parse_llm_json(result_text)
    if result_json is None:
        print("JSON parsing failed, returning empty ArticleExtraction")
        result_json = {}

    # Validate and coerce output using Pydantic
    try:
        article = ArticleExtraction(**result_json)
    except ValidationError as e:
        print(f"Validation error: {e}")
        article = ArticleExtraction()

    # Token usage handling
    token_usage = {}
    if hasattr(response, "usage") and response.usage:
        token_usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        }

    return article, token_usage

async def process_folder_with_progress(
    folder_path: Path,
    output_dir: Path,
    client,  # OpenAI client instance (any provider)
    model_name: str,
    progress_callback: Optional[Callable] = None
):
    """
    Process all markdown files in a folder using LLM extraction.

    Args:
        folder_path: Path to folder containing markdown files
        output_dir: Path to output directory for CSV/JSON
        client: OpenAI client (Azure or Ollama)
        model_name: Model name to use
        progress_callback: Optional callback function(message, current, total)

    Returns:
        tuple: (DataFrame, stats_dict)
    """
    start_time = time.time()
    table = []

    # Get all markdown files
    md_files = list(folder_path.rglob('*.md'))
    total_files = len(md_files)

    if total_files == 0:
        return pd.DataFrame(), {
            'total_files': 0,
            'processed': 0,
            'skipped_error': 0,
            'duration_seconds': 0
        }

    processed = 0
    failed = 0

    for idx, file_path in enumerate(md_files, 1):
        file_name = file_path.name

        # Update progress
        if progress_callback:
            elapsed = time.time() - start_time
            if processed > 0:
                avg_time = elapsed / processed
                remaining_estimate = avg_time * (total_files - processed)
            else:
                remaining_estimate = 0

            progress_callback(
                f"Processing: {file_name}",
                idx,
                total_files
            )

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract using LLM
            result, token_usage = llm_extract(client, model_name, content)

            table.append({
                "file": file_name,
                "url": result.url,
                "title": result.title,
                "publication_date": result.publication_date,
                "content": result.main_content,
                "categories": ', '.join(result.categories) if result.categories else '',
                "prompt_tokens": token_usage.get("prompt_tokens"),
                "completion_tokens": token_usage.get("completion_tokens"),
                "total_tokens": token_usage.get("total_tokens"),
                "success": True
            })
            processed += 1

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            table.append({
                "file": file_name,
                "url": None,
                "title": None,
                "publication_date": None,
                "content": "",
                "categories": "",
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "success": False,
                "error": str(e)
            })
            failed += 1

    # Create DataFrame
    df = pd.DataFrame(table)

    # Save to output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = output_dir / f"llm_extraction_{int(time.time())}.csv"
    df.to_csv(csv_path, index=False)

    # Save JSON
    json_path = output_dir / f"llm_extraction_{int(time.time())}.json"
    df.to_json(json_path, orient='records', indent=2)

    end_time = time.time()
    duration = end_time - start_time

    stats = {
        'total_files': total_files,
        'processed': processed,
        'skipped_error': failed,
        'duration_seconds': duration,
        'output_csv': str(csv_path),
        'output_json': str(json_path)
    }

    return df, stats