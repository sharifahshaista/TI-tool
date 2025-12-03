# TI-tool
This is a prototype tool to complement technology intelligence.

## Overview

TI Agent is a comprehensive Streamlit application that automates the entire workflow of technology intelligence gathering:

1. **Web Search** - AI-assisted research with clarification and SERP generation
2. **Web Crawler** - Website crawling with post-processing URL filtering
3. **LLM Extraction** - Extract structured metadata from crawled content using AI
4. **Summarization** - AI-powered tech-intelligence analysis and categorization
5. **Database** - Consolidated searchable database with advanced filtering
6. **RAG Chatbot** - Query your knowledge base with AI-powered citations
7. **LinkedIn Monitor** - Track LinkedIn posts (optional feature)

---

## 🎯 Quick Start (Local Development)

### Prerequisites

- Python 3.11+
- **One of the following LLM providers:**
  - Azure OpenAI API credentials (ideally), OR
  - OpenAI API key, OR
  - LM Studio running locally with a model loaded
- SearXNG instance (for web search)
- AWS S3 bucket (for persistent storage)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd TI-tool

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

### LLM Provider Configuration

This tool supports **three LLM providers**. Choose one and configure it in your `.env` file:

#### Option 1: Azure OpenAI (Recommended for enterprise)

```bash
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
OPENAI_API_VERSION=2023-12-01-preview
AZURE_OPENAI_MODEL_NAME=gpt-4
```

#### Option 2: OpenAI API (Easiest to get started)

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4
```

#### Option 3: LM Studio 
(Functional only if tool runs locally. Configure AWS S3 ```bash USE_S3_STORAGE=False``` too)

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Load a model (recommended: Llama 3, Mistral, or similar 7B+ model)
3. Start the local server (Server tab → Start Server)
4. Configure `.env`:

```bash
LLM_PROVIDER=lm_studio
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_API_KEY=lm-studio
```

**Note:** LM Studio uses whatever model you have loaded in the application.

### Required Environment Variables

Create `.env`:

```bash
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET_NAME=ti-tool-s3-storage
SEARXNG_URL=http://localhost:8080
```

Start SearXNG instance:
```docker run -d -p 32768:8080 searxng/searxng```


### Run the Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## Project Structure

```
TI-tool/
├── app.py                          # Main Streamlit application
├── aws_storage.py                  # S3 storage integration
├── embeddings_rag.py               # LlamaIndex RAG system
├── requirements.txt                # Python dependencies
├── packages.txt                    # System dependencies (for Streamlit Cloud)
├── .streamlit/
│   ├── config.toml                 # Streamlit configuration
│   └── secrets.toml.example        # Secrets template
│
├── README.md                       # This file
│
├── agents/                         # AI agents and processors
│   ├── clarification.py            # Research clarification agent
│   ├── serp.py                     # SERP query generation
│   ├── learn.py                    # Learning extraction agent
│   ├── llm_extractor.py            # Metadata extractor using LLM
│   ├── web_search.py               # Research agent
│   └── summarise_csv.py            # Tech-intelligence summarizer
│
├── config/                         # Configuration modules
│   ├── model_config.py             # LLM provider configuration
│   └── searxng_tools.py            # SearXNG integration
│
├── schemas/                        # Pydantic data models
│   └── datamodel.py                # Data schemas and validation
│
├── webcrawler/                     # Web crawler modules
│   ├── scraper.py                  # Core scraping logic
│   ├── url_utils.py                # URL utilities
│   ├── (...)                       # Others to handle robots.txt and URL tracking
│   └── content_extractor.py        # Content extraction
│
└── S3 Storage (crawled_data/, processed_data/, summarised_content/, rag_embeddings/)
    # All data stored in AWS S3 bucket for persistence
```

### Storage Architecture

This application uses **AWS S3** for persistent storage instead of local filesystem:

- **crawled_data/**: Raw crawled website data
- **processed_data/**: Filtered and processed URLs
- **summarised_content/**: AI-generated summaries and analysis
- **rag_embeddings/**: Vector embeddings for RAG chatbot
---

## Features

### 1. Web Search Pipeline

- **AI Clarification**: Refines research scope with targeted questions
- **SERP Generation**: Creates optimized search queries
- **Web Search**: Executes searches via SearXNG
- **Learning Extraction**: Extracts structured insights from results

### 2. Web Crawler

**Two-tab interface combining crawling and URL filtering:**

#### Tab 1: Crawl Websites
- Configure crawler based on number of pages, duration of delay and whether to overwrite previous history of crawling a website. Crawl logs are displayed in the terminal, **not** on the UI.

#### Tab 2: Filter URLs
- Remove unwanted URLs from crawled data (e.g., `/about`, `/author`, `/contact`) *(the noisy data)*
- Results are saved in a CSV file with column headers "date of crawl", "URL" and text_content"
- Saves filtered data to `crawled_data/` in S3

### 3. LLM Extraction
Use AI models to extract structured metadata from 'text_content' column in each row of crawled data CSV file:
   * Title
   * Publication Date
   * Main Content
   * Tags/Categories

**🔄 Interrupt Handling & Progress Preservation**
The LLM extraction process now supports resumable processing with automatic checkpointing:

- **Automatic Checkpointing**: Progress is saved every 10 processed items (configurable)
- **Interrupt Recovery**: If processing is interrupted (Ctrl+C, connection loss, system crash), progress is preserved
- **Resume Functionality**: Automatically resumes from the last checkpoint when restarted
- **Graceful Shutdown**: Ctrl+C triggers clean shutdown with progress saving

**Usage:**
```python
# Process with checkpointing enabled (default)
df, stats = await process_csv_with_progress(
    csv_path=csv_file,
    output_dir=output_dir,
    client=client,
    model_name="gpt-4.1-nano",
    checkpoint_interval=10,  # Save every 10 rows
    resume_from_checkpoint=True  # Resume from existing checkpoint
)
```
Results saved to `processed_data/` in S3.

### 4. Summarisation 
Upload or select processed CSV files from S3 to perform summarization and classification:
   * **Indicator**: A concise summary focusing on the key technological development, event, or trend described.
   * **Dimension**: Primary category from tech, policy, economic, environmental & safety, social & ethical, legal & regulatory.
   * **Tech**: Specific technology domain or sector.
   * **TRL**: Technology Readiness Level (1-9 scale).
   * **Start-up**: If the news is about a start-up, the URL to the start-up's official webpage is included.

Results saved to `summarised_content/` in S3.

### 5. Database

- **Unified View**: All summarized content in one searchable table
- **Full-Text Search**: Across all visible columns
- **Export Options**: CSV, Excel with filtered or complete data

### 6. RAG Chatbot

- **Multi-Index Support**: Query multiple data sources simultaneously
- **Persistent Storage**: Embeddings saved to disk (no rebuild needed)
- **Website Citations**: Responses cite sources by name (e.g., `[canarymedia]`)
- **Metadata Display**: Shows title, date, URL, tech fields

---

## Typical Workflow

```
1. **Web Crawler**
   ↓
   Crawl target websites
   ↓
   Filter out noisy, irrelevant URLs
   ↓
   Save crawled content in CSV named `{website}_{date of crawl}_filtered.{ext}`

2. **LLM Extraction**
   ↓
   Extract {Title, Publication Date, Main Content, Categories/Tags}
   ↓
   Save to processed files as `<website>_<date of crawl>_filtered_<date of extraction>.{ext}`

3. **Summarisation**
   ↓
   Generate summaries 
   Classify dimension, technology and TRL
   Identify URLs to start-up(s) if mentioned in articles
   ↓
   Save to summarised_content/ as `<website>_<date of summarisation>.{ext}`

4. Database
   ↓
   Filter by sources, date range and keywords
   ↓
   Export data as CSV or JSON formats

5. RAG Chatbot
   ↓
   Build vector index
   ↓
   Query with metadata filtering enabled alongside FAISS

6. Linkedin Home Feed Monitor
   ↓
   Adjust settings of 'scraper' based on number of days back and scroll pause duration. Otherwise, runtime is 1 hour.
   ↓
   Download results, or store in S3
```

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Framework** | Streamlit |
| **AI/LLM** | PydanticAI, OpenAI, Azure OpenAI, Anthropic, Groq |
| **RAG** | LlamaIndex, OpenAI Embeddings |
| **Web** | Playwright, BeautifulSoup4, Trafilatura, SearXNG |
| **Data** | Pandas, NumPy |
| **UI** | Streamlit-AgGrid |
| **Storage** | CSV, JSON, LlamaIndex Vector Store |

---

## File Naming Convention

All processed files follow the pattern: `{website}_{YYYYMMDD}.{ext}`

**Examples**:
- `canarymedia_20251029.csv`
- `thinkgeoenergy_20251029.json`
- `carboncapturemagazine_20251029_log.txt`

This makes it easy to:
- Identify the source website
- Track processing dates
- Manage multiple snapshots over time

---

## Configuration

### Model Selection

The app supports multiple LLM providers:

- **Azure OpenAI** (default): `gpt-4.1-nano`, `gpt-4o`, `gpt-4o-mini`
- **OpenAI**: Standard OpenAI models
- **LM Studio**: Local models

## Important Notes

### Crawling

- **Stay on page**: Navigating away interrupts the crawl, do other tasks while leaving the tab open.
- **Rate limiting**: Some sites may block aggressive crawling
- **Respect robots.txt**: Be a good web citizen!

### Processing

- **Stay on page**: Navigating interrupts summarization
- **API costs**: Summarization calls the LLM for each row
- **Token limits**: Large content may exceed model limits

### RAG

- **Persistent storage**: Embeddings are saved to S3 as pickle files in `rag_embeddings/`
- **No rebuild needed**: Load existing indexes instantly
- **Multiple sources**: Load the embeddings of the sources relevant to you from S3, then query across multiple indexes simultaneously

---
