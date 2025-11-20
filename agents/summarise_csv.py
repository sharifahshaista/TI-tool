"""
Summarization Agent

This agent summarizes text content from CSV files using a tech-intelligence focused prompt.
Includes S3 storage integration for data persistence.
"""

from config.azure_model import model
from pydantic_ai import Agent
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import time

# Import S3 storage
try:
    from aws_storage import get_storage
    HAS_S3_STORAGE = True
except ImportError:
    HAS_S3_STORAGE = False

# Tech Intelligence Summarization Prompt
TECH_INTEL_PROMPT = """
You are an expert technology intelligence analyst specialising in extracting structured information from news articles. Your task is to analyse the provided news article
and extract specific information fields in a standardised format.

Extract the following fields from the article:

- Indicator: A concise 1-paragraph summary (3-5 sentences) strictly focusing on the key technological development, event, or trend described in the article. If there are any companies
mentioned in the article, make sure they are captured. Avoid generic openers like "The article hightlights..." or "The article discusses...". Only use English characters and standard numerals.

- Dimension: Primary category based on the article content (choose only one of them, using the abbreviation):
    - Tech (Technology)
    - Pol (Policy)
    - Econ (Economic)
    - Env&S (Environmental & Safety)
    - Legal&R (Legal & Regulatory)
    - Social&E (Social & Ethical)

- Tech: Specific technology domain or sector (e.g. AI, Blockchain, IoT, Grid, Renewable, Energy, Cybersecurity).

- TRL: Technology Readiness Level (1-9 scale):
    - 1-3: Basic Research and proof of concept
    - 4-5: Laboratory validation and demonstration
    - 6-7: Prototype testing in relevant/operational environment
    - 8-9: System proven and commercially deployed

- Start-up: If the news is about a start-up, include a link to the start-up's official webpage. If multiple sources, separate with semicolons. If not applicable, write "N/A".

Output Format:
Provide your response in the following exact format, one field per line:

INDICATOR: [your 1-paragraph summary here]
DIMENSION: [abbreviation only: Tech/Pol/Econ/Env&S/Legal&R/Social&E]
TECH: [specific technology domain]
TRL: [number 1-9]
START-UP: [URL or N/A]
"""


def create_summarization_agent(custom_model=None):
    """
    Create the summarization agent for tech-intelligence content
    
    Args:
        custom_model: Optional custom model to use (overrides default)
    """
    agent_model = custom_model if custom_model is not None else model
    
    summarization_agent = Agent(
        model=agent_model,
        output_type=str,
        system_prompt=TECH_INTEL_PROMPT
    )
    return summarization_agent


def parse_structured_output(output: str) -> dict:
    """
    Parse the structured output from the LLM into separate fields
    
    Args:
        output: Raw output from LLM
        
    Returns:
        Dictionary with keys: indicator, dimension, tech, trl, start_up
    """
    result = {
        'indicator': '',
        'dimension': '',
        'tech': '',
        'trl': '',
        'start_up': ''
    }
    
    try:
        lines = output.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.upper().startswith('INDICATOR:'):
                result['indicator'] = line.split(':', 1)[1].strip()
            elif line.upper().startswith('DIMENSION:'):
                result['dimension'] = line.split(':', 1)[1].strip()
            elif line.upper().startswith('TECH:'):
                result['tech'] = line.split(':', 1)[1].strip()
            elif line.upper().startswith('TRL:'):
                result['trl'] = line.split(':', 1)[1].strip()
            elif line.upper().startswith('START-UP:') or line.upper().startswith('STARTUP:'):
                result['start_up'] = line.split(':', 1)[1].strip()
    
    except Exception as e:
        logging.error(f"Error parsing structured output: {e}")
    
    return result


async def summarize_content(content: str, custom_model=None) -> dict:
    """
    Summarize a single piece of content using the tech-intel prompt
    
    Args:
        content: Raw text content to summarize
        custom_model: Optional custom model to use
        
    Returns:
        Dictionary with structured fields: indicator, dimension, tech, trl, start_up
    """
    agent = create_summarization_agent(custom_model)
    
    try:
        result = await agent.run(content)
        parsed = parse_structured_output(result.output)
        
        # Clear 'tech' field if dimension is not 'Tech'
        dimension = parsed.get('dimension', '').strip().lower()
        if dimension != 'tech':
            parsed['tech'] = ''
        
        return parsed
    except Exception as e:
        logging.error(f"Error summarizing content: {e}")
        return {
            'indicator': f"[Error: {str(e)}]",
            'dimension': '',
            'tech': '',
            'trl': '',
            'start_up': ''
        }


async def summarize_csv_file(
    csv_file_path: Path,
    content_column: str = "content",
    progress_callback=None,
    custom_model=None
) -> tuple[pd.DataFrame, float, dict]:
    """
    Process a CSV file and summarize the content column
    
    Args:
        csv_file_path: Path to the CSV file
        content_column: Name of the column containing content to summarize
        progress_callback: Optional callback function(current, total, elapsed, est_remaining)
        custom_model: Optional custom model to use for summarization
        
    Returns:
        Tuple of (processed_dataframe, duration_seconds, metadata)
    """
    start_time = time.time()
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        raise ValueError(f"Could not read CSV file: {e}")
    
    # Validate content column exists
    if content_column not in df.columns:
        raise ValueError(
            f"Column '{content_column}' not found in CSV. "
            f"Available columns: {', '.join(df.columns)}"
        )
    
    # Fill empty content with content snippet if available
    if 'content snippet' in df.columns:
        # Create a mask for rows where content is empty/null
        empty_content_mask = df[content_column].isna() | (df[content_column].astype(str).str.strip() == '') | (df[content_column].astype(str).str.lower() == 'nan')
        
        # Fill those rows with content snippet
        df.loc[empty_content_mask, content_column] = df.loc[empty_content_mask, 'content snippet']
        
        filled_count = empty_content_mask.sum()
        if filled_count > 0:
            logging.info(f"Filled {filled_count} empty content rows with content snippet")
    
    # Add structured columns for tech intelligence fields
    df['Indicator'] = None
    df['Dimension'] = None
    df['Tech'] = None
    df['TRL'] = None
    df['Start-up'] = None
    
    # Track processing stats
    total_rows = len(df)
    successful = 0
    failed = 0
    
    # Process each row
    for row_num, (idx, row) in enumerate(df.iterrows()):
        row_start_time = time.time()
        
        try:
            content = str(row[content_column])
            
            # Check if content is still empty after filling
            if not content or content.strip() == '' or content.lower() == 'nan':
                # Mark as empty - content snippet should have been filled earlier
                df.loc[idx, 'Indicator'] = "[Empty content - no summary generated]"  # type: ignore
                df.loc[idx, 'Dimension'] = ''  # type: ignore
                df.loc[idx, 'Tech'] = ''  # type: ignore
                df.loc[idx, 'Start-up'] = ''  # type: ignore
                failed += 1
                logging.warning(f"Row {row_num + 1}/{total_rows} has no content to process")
                continue
            
            # Summarize content and extract structured fields
            structured_data = await summarize_content(content, custom_model)
            
            # Populate the structured columns
            df.loc[idx, 'Indicator'] = structured_data.get('indicator', '')  # type: ignore
            df.loc[idx, 'Dimension'] = structured_data.get('dimension', '')  # type: ignore
            df.loc[idx, 'Tech'] = structured_data.get('tech', '')  # type: ignore
            df.loc[idx, 'TRL'] = structured_data.get('trl', '')  # type: ignore
            df.loc[idx, 'Start-up'] = structured_data.get('start_up', '')  # type: ignore
            
            successful += 1
            
            logging.info(f"Processed row {row_num + 1}/{total_rows}")
            
        except Exception as e:
            logging.error(f"Error processing row {idx}: {e}")
            df.loc[idx, 'Indicator'] = f"[Error: {str(e)}]"  # type: ignore
            df.loc[idx, 'Dimension'] = ''  # type: ignore
            df.loc[idx, 'Tech'] = ''  # type: ignore
            df.loc[idx, 'Start-up'] = ''  # type: ignore
            failed += 1
        
        # Calculate progress and time estimates
        current_row = row_num + 1
        elapsed_time = time.time() - start_time
        rows_remaining = total_rows - current_row
        
        # Calculate estimated time remaining
        if current_row > 0:
            avg_time_per_row = elapsed_time / current_row
            est_remaining = avg_time_per_row * rows_remaining
        else:
            est_remaining = 0
        
        # Call progress callback if provided
        if progress_callback:
            progress_callback(current_row, total_rows, elapsed_time, est_remaining)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Create metadata
    metadata = {
        'total_rows': total_rows,
        'successful': successful,
        'failed': failed,
        'duration_seconds': duration,
        'timestamp': datetime.now(),
        'source_file': csv_file_path.name,
        'content_column': content_column
    }
    
    return df, duration, metadata


def save_summarized_csv(
    df: pd.DataFrame,
    metadata: dict,
    output_dir: Path = Path("summarised_content")
) -> tuple[Path, Path, Path]:
    """
    Save the summarized data as both CSV and JSON, plus create a log file
    
    Args:
        df: Processed DataFrame with summaries
        metadata: Processing metadata
        output_dir: Directory to save files
        
    Returns:
        Tuple of (csv_path, json_path, log_path)
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp (YYYYMMDD format)
    timestamp_str = metadata['timestamp'].strftime('%Y%m%d')
    
    # Extract website name from original filename
    # Handle patterns like "thinkgeoenergy_processed_20251028_233006" or "thinkgeoenergy_20251028"
    original_name = Path(metadata['source_file']).stem
    
    # Extract just the website name (everything before first underscore or date pattern)
    import re
    # Remove common suffixes like _processed, _summarized, and timestamps
    website_name = re.split(r'_(?:processed|summarized|\d{8})', original_name)[0]
    
    csv_filename = f"{website_name}_{timestamp_str}.csv"
    json_filename = f"{website_name}_{timestamp_str}.json"
    log_filename = f"{website_name}_{timestamp_str}_log.txt"
    
    csv_path = output_dir / csv_filename
    json_path = output_dir / json_filename
    log_path = output_dir / log_filename
    
    # Format date columns to ISO date format (YYYY-MM-DD) without time
    df_copy = df.copy()
    
    # Define the required output columns in the correct order
    required_columns = [
        'filename', 'filepath', 'url', 'title', 'publication_date', 
        'content', 'categories', 'Indicator', 'Dimension', 'Tech', 'TRL', 'Start-up'
    ]
    
    # Select only the required columns that exist in the dataframe
    # Handle both old format (file) and new format (filename/filepath)
    available_columns = []
    for col in required_columns:
        if col in df_copy.columns:
            available_columns.append(col)
        elif col == 'filename' and 'file' in df_copy.columns:
            # For backward compatibility, map 'file' to 'filename'
            df_copy['filename'] = df_copy['file']
            available_columns.append('filename')
        elif col == 'filepath' and 'file' in df_copy.columns:
            # If only 'file' exists and no 'filepath', use empty string
            df_copy['filepath'] = ''
            available_columns.append('filepath')
    
    # Reorder dataframe to match required column order
    df_copy = df_copy[available_columns]
    
    for col in df_copy.columns:
        if col.lower() in ['pubdate', 'date', 'published', 'publish_date']:
            try:
                # Convert to datetime and extract just the date
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce').dt.strftime('%Y-%m-%d')
            except Exception as e:
                logging.warning(f"Could not format date column '{col}': {e}")
    
    # Save CSV
    df_copy.to_csv(csv_path, index=False, encoding='utf-8')
    
    # Save JSON
    df_copy.to_json(json_path, orient='records', indent=2)
    
    # Create log file
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("SUMMARIZATION LOG\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source File: {metadata['source_file']}\n")
        f.write(f"Date: {metadata['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Content Column: {metadata['content_column']}\n\n")
        f.write("-" * 60 + "\n")
        f.write("PROCESSING STATISTICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total Rows: {metadata['total_rows']}\n")
        f.write(f"Successfully Processed: {metadata['successful']}\n")
        f.write(f"Failed: {metadata['failed']}\n")
        f.write(f"Success Rate: {(metadata['successful'] / metadata['total_rows'] * 100):.2f}%\n\n")
        f.write("-" * 60 + "\n")
        f.write("DURATION\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total Duration: {metadata['duration_seconds']:.2f} seconds\n")
        f.write(f"Average per Row: {(metadata['duration_seconds'] / metadata['total_rows']):.2f} seconds\n\n")
        f.write("-" * 60 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("-" * 60 + "\n")
        f.write(f"Summarized CSV: {csv_filename}\n")
        f.write(f"Summarized JSON: {json_filename}\n")
        f.write(f"Log File: {log_filename}\n")
        f.write("\n" + "=" * 60 + "\n")
    
    logging.info(f"Saved summarized CSV to: {csv_path}")
    logging.info(f"Saved summarized JSON to: {json_path}")
    logging.info(f"Saved log to: {log_path}")
    
    # Upload to S3 if available (only CSV and JSON, not log or history files)
    if HAS_S3_STORAGE:
        try:
            from aws_storage import get_storage
            storage = get_storage()
            
            # Upload CSV to S3
            s3_csv_key = f"summarised_content/{csv_filename}"
            storage.upload_file(str(csv_path), s3_csv_key)
            logging.info(f"✓ CSV uploaded to S3: {s3_csv_key}")
            
            # Upload JSON to S3
            s3_json_key = f"summarised_content/{json_filename}"
            storage.upload_file(str(json_path), s3_json_key)
            logging.info(f"✓ JSON uploaded to S3: {s3_json_key}")
            
            # Note: Log files and history.json are kept local only
            
        except Exception as e:
            logging.warning(f"⚠️ Failed to upload to S3: {e}")
    
    return csv_path, json_path, log_path

