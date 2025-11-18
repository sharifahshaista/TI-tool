import os
import json
import time
import pandas as pd
from pathlib import Path
from typing import List, Optional, Callable
from pydantic import BaseModel, Field, ValidationError
from datetime import datetime, timedelta
import re

# Import S3 storage
try:
    from aws_storage import get_storage
    HAS_S3_STORAGE = True
except ImportError:
    HAS_S3_STORAGE = False

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
"title": "string or null",
"publication_date": "string or null",
"main_content": "string",
"categories": ["array of strings"]
}}

**INSTRUCTIONS:**
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

def filter_extracted_data(df: pd.DataFrame, current_date: datetime = None) -> tuple:
    """
    Filter extracted data to remove rows where:
    1. Extracted content is empty
    2. Publication date is older than 2 years from current processing date
    
    Args:
        df: DataFrame with extracted data
        current_date: Reference date for filtering (defaults to now)
    
    Returns:
        tuple: (filtered_df, stats_dict)
    """
    if current_date is None:
        current_date = datetime.now()
    
    initial_count = len(df)
    
    # Track filtering reasons
    empty_content_count = 0
    old_date_count = 0
    
    # Create a mask for rows to keep
    keep_mask = pd.Series([True] * len(df), index=df.index)
    
    # Filter 1: Remove rows with empty main_content
    if 'main_content' in df.columns:
        empty_content_mask = (
            df['main_content'].isna() | 
            (df['main_content'].astype(str).str.strip() == '') |
            (df['main_content'].astype(str) == 'nan')
        )
        empty_content_count = empty_content_mask.sum()
        keep_mask &= ~empty_content_mask
    
    # Filter 2: Remove rows with publication_date older than 2 years
    if 'publication_date' in df.columns:
        def is_old_date(date_str):
            if pd.isna(date_str) or str(date_str).strip() == '' or str(date_str) == 'nan':
                return False  # Keep rows with no date (don't filter out)
            
            try:
                # Try to parse the date
                pub_date = pd.to_datetime(date_str, errors='coerce')
                if pd.isna(pub_date):
                    return False  # Keep if can't parse
                
                # Calculate age
                age_years = (current_date - pub_date).days / 365.25
                return age_years > 2
            except:
                return False  # Keep if any error
        
        old_date_mask = df['publication_date'].apply(is_old_date)
        old_date_count = old_date_mask.sum()
        keep_mask &= ~old_date_mask
    
    # Apply the filter
    filtered_df = df[keep_mask].copy()
    final_count = len(filtered_df)
    removed_count = initial_count - final_count
    
    stats = {
        'initial_rows': initial_count,
        'final_rows': final_count,
        'removed_total': removed_count,
        'removed_empty_content': empty_content_count,
        'removed_old_date': old_date_count,
        'filter_date_threshold': (current_date - timedelta(days=365.25*2)).strftime('%Y-%m-%d')
    }
    
    return filtered_df, stats


async def process_csv_with_progress(
    csv_path: Path,
    output_dir: Path,
    client,  # OpenAI client instance (any provider)
    model_name: str,
    text_column: str = "text_content",
    progress_callback: Optional[Callable] = None
):
    """
    Process CSV file with LLM extraction on text_content column.

    Args:
        csv_path: Path to CSV file with crawled data
        output_dir: Path to output directory for CSV/JSON
        client: OpenAI client (Azure, OpenAI, or LM Studio)
        model_name: Model name to use
        text_column: Name of column containing text to extract from (default: "text_content")
        progress_callback: Optional callback function(message, current, total)

    Returns:
        tuple: (DataFrame, stats_dict)
    """
    start_time = time.time()
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return pd.DataFrame(), {
            'total_rows': 0,
            'processed': 0,
            'skipped_error': 0,
            'duration_seconds': 0
        }
    
    # Verify text_column exists
    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found in CSV. Available columns: {df.columns.tolist()}")
        return pd.DataFrame(), {
            'total_rows': 0,
            'processed': 0,
            'skipped_error': 0,
            'duration_seconds': 0,
            'error': f"Column '{text_column}' not found"
        }
    
    total_rows = len(df)
    processed = 0
    failed = 0
    
    # Prepare new columns for extracted data
    extracted_data = []
    
    for idx, row in df.iterrows():
        row_num = idx + 1
        
        # Update progress
        if progress_callback:
            elapsed = time.time() - start_time
            if processed > 0:
                avg_time = elapsed / processed
                remaining_estimate = avg_time * (total_rows - processed)
            else:
                remaining_estimate = 0
            
            # Get URL for display (if available)
            display_url = row.get('url', f'Row {row_num}')
            if isinstance(display_url, str) and len(display_url) > 50:
                display_url = display_url[:50] + '...'
            
            progress_callback(
                f"Processing row {row_num}: {display_url}",
                row_num,
                total_rows
            )
        
        try:
            # Get text content from the specified column
            content = str(row[text_column])
            
            if not content or content.strip() == '' or content == 'nan':
                # Skip empty content
                extracted_data.append({
                    "title": None,
                    "publication_date": None,
                    "main_content": "",
                    "categories": ""
                })
                failed += 1
                continue
            
            # Extract using LLM
            result, token_usage = llm_extract(client, model_name, content)
            
            extracted_data.append({
                "title": result.title,
                "publication_date": result.publication_date,
                "main_content": result.main_content,
                "categories": ', '.join(result.categories) if result.categories else ''
            })
            processed += 1
            
        except Exception as e:
            print(f"Error processing row {row_num}: {e}")
            extracted_data.append({
                "title": None,
                "publication_date": None,
                "main_content": "",
                "categories": ""
            })
            failed += 1
    
    # Create DataFrame from extracted data
    extracted_df = pd.DataFrame(extracted_data)
    
    # Combine original CSV with extracted data
    result_df = pd.concat([df, extracted_df], axis=1)
    
    # Apply filtering to remove empty content and old dates
    if progress_callback:
        progress_callback(
            f"Filtering data (removing empty content and old dates)...",
            total_rows,
            total_rows
        )
    
    result_df_filtered, filter_stats = filter_extracted_data(result_df)
    
    # Select only required columns for output: text_content, url, and extracted fields
    output_columns = []
    
    # Add text_content if it exists
    if text_column in result_df_filtered.columns:
        output_columns.append(text_column)
    
    # Add url if it exists
    if 'url' in result_df_filtered.columns:
        output_columns.append('url')
    
    # Add all extracted fields
    extracted_fields = [
        'title',
        'publication_date',
        'main_content',
        'categories'
    ]
    
    for field in extracted_fields:
        if field in result_df_filtered.columns:
            output_columns.append(field)
    
    # Create output DataFrame with only selected columns
    output_df = result_df_filtered[output_columns].copy()
    
    # Save to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename from source CSV name and date
    source_name = csv_path.stem  # filename without extension
    
    # Remove any existing date suffix (e.g., "canarymedia_20251115" -> "canarymedia")
    # This handles filenames like "source_YYYYMMDD" from web crawler
    import re
    date_pattern = r'_\d{8}$'  # Matches "_YYYYMMDD" at the end
    source_name = re.sub(date_pattern, '', source_name)
    
    date_str = datetime.now().strftime('%Y%m%d')
    
    # Save CSV
    output_csv_path = output_dir / f"{source_name}_{date_str}.csv"
    output_df.to_csv(output_csv_path, index=False)
    
    # Save JSON
    output_json_path = output_dir / f"{source_name}_{date_str}.json"
    output_df.to_json(output_json_path, orient='records', indent=2)
    
    # Upload to S3 if available
    if HAS_S3_STORAGE:
        try:
            storage = get_storage()
            
            # Upload CSV to S3
            s3_csv_key = f"processed_data/{source_name}_{date_str}.csv"
            storage.upload_dataframe(output_df, s3_csv_key)
            print(f"✅ Uploaded CSV to S3: {s3_csv_key}")
            
            # Upload JSON to S3
            s3_json_key = f"processed_data/{source_name}_{date_str}.json"
            storage.upload_file(str(output_json_path), s3_json_key)
            print(f"✅ Uploaded JSON to S3: {s3_json_key}")
            
        except Exception as e:
            print(f"⚠️ Failed to upload to S3: {e}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    stats = {
        'total_rows': total_rows,
        'processed': processed,
        'skipped_error': failed,
        'filtered_rows': filter_stats['final_rows'],
        'removed_empty_content': filter_stats['removed_empty_content'],
        'removed_old_date': filter_stats['removed_old_date'],
        'filter_date_threshold': filter_stats['filter_date_threshold'],
        'duration_seconds': duration,
        'output_csv': str(output_csv_path),
        'output_json': str(output_json_path)
    }
    
    return output_df, stats


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
                "filename": file_name,
                "filepath": str(file_path),
                "url": result.url,
                "title": result.title,
                "publication_date": result.publication_date,
                "content": result.main_content,
                "categories": ', '.join(result.categories) if result.categories else ''
            })
            processed += 1

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            table.append({
                "filename": file_name,
                "filepath": str(file_path),
                "url": None,
                "title": None,
                "publication_date": None,
                "content": "",
                "categories": ""
            })
            failed += 1

    # Create DataFrame
    df = pd.DataFrame(table)

    # Save to output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename from source folder name and date
    source_name = folder_path.name
    date_str = datetime.now().strftime('%Y%m%d')
    
    # Save CSV
    csv_path = output_dir / f"{source_name}_{date_str}.csv"
    df.to_csv(csv_path, index=False)

    # Save JSON
    json_path = output_dir / f"{source_name}_{date_str}.json"
    df.to_json(json_path, orient='records', indent=2)
    
    # Upload to S3 if available
    if HAS_S3_STORAGE:
        try:
            storage = get_storage()
            
            # Upload CSV to S3
            s3_csv_key = f"processed_data/{source_name}_{date_str}.csv"
            storage.upload_dataframe(df, s3_csv_key)
            print(f"✅ Uploaded CSV to S3: {s3_csv_key}")
            
            # Upload JSON to S3
            s3_json_key = f"processed_data/{source_name}_{date_str}.json"
            with open(json_path, 'rb') as f:
                storage.upload_file(f, s3_json_key)
            print(f"✅ Uploaded JSON to S3: {s3_json_key}")
            
        except Exception as e:
            print(f"⚠️ Failed to upload to S3: {e}")

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