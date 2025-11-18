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

Create `.streamlit/secrets.toml` (for Streamlit Cloud) or `.env` (for local):

```toml
[LLM_PROVIDER]
PROVIDER = "azure"

[AZURE_OPENAI]
AZURE_OPENAI_API_KEY = "your-azure-api-key"
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o-mini"
AZURE_OPENAI_API_VERSION = "2024-02-15-preview"

[AWS]
AWS_ACCESS_KEY_ID = "your-access-key-id"
AWS_SECRET_ACCESS_KEY = "your-secret-access-key"
AWS_DEFAULT_REGION = "us-east-1"
S3_BUCKET_NAME = "ti-tool-s3-storage"

[SEARXNG]
SEARXNG_URL = "http://your-searxng-instance:8080"
```

**For local development with `.env`:**
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

Start Crawl4AI instance:
```docker run -d \
-p 11235:11235 \
  --name crawl4ai \
  --shm-size=1g \
  unclecode/crawl4ai:latest```

```pip install crawl4ai
crawl4ai-setup
crawl4ai-doctor
```

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
└── S3 Storage (crawled_data/, processed_data/, summarised_content/, rag_storage/)
    # All data stored in AWS S3 bucket for persistence
```

### Storage Architecture

This application uses **AWS S3** for persistent storage instead of local filesystem:

- **crawled_data/**: Raw crawled website data
- **processed_data/**: Filtered and processed URLs
- **summarised_content/**: AI-generated summaries and analysis
- **rag_storage/**: Vector embeddings for RAG chatbot

**Benefits:**
- ✅ Works on Streamlit Cloud (stateless containers)
- ✅ Data persists across deployments
- ✅ Accessible from anywhere
- ✅ Automatic backups and versioning

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
- Preview filtered results before saving
- Saves filtered data to `processed_data/` in S3

### 3. LLM Extraction
Use AI models to intelligently extract structured metadata from markdown files:
   * Title
   * Publication Date
   * URL
   * Main Content
   * Author
   * Tags/Categories

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

**Example Usage: Watch the demo below**

https://github.com/user-attachments/assets/813787e0-2526-40b6-a148-da4eb46a82bc

---

## Typical Workflow

```
1. Web Crawler
   ↓
   Crawl target websites
   ↓
   Save markdown files

2. LLM Extraction
   ↓
   Extract structured metadata
   ↓
   Save to processed files

3. Summarization
   ↓
   Analyze with tech-intelligence
   ↓
   Save to summarised_content/

4. Database
   ↓
   Search, filter, export data

5. RAG Chatbot
   ↓
   Build vector index
   ↓
   Query with AI citations

6. Linkedin Home Feed Monitor
   ↓
   Adjust settings of 'scraper' based on number of days back and scroll pause duration.
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
| **UI** | Streamlit-AgGrid, Streamlit-Agraph |
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
- **Anthropic**: Claude models

Configure in the Summarization or RAG pages.
---

## Usage Examples

### Example 1: Crawl a News Site

1. Go to **Web Crawler**
2. Enter URL: `https://www.canarymedia.com`
3. Click **Auto-Detection**
4. Review recommended strategy (likely "Sitemap")
5. Set max pages: `500`
6. Click **Start Crawling**
7. Wait for completion (*may take hours*)

### Example 2: Extract Metadata with LLM

1. Go to **LLM Extraction**
2. Select folder: `canarymedia`
3. Configure extraction fields
4. Click **Start Extraction**
5. Files saved to extraction output folder

### Example 3: Generate Tech Intelligence

1. Go to **Summarization**
2. Upload the processed CSV
3. Select model: `pmo-gpt-4.1-nano`
4. Click **Start Summarization**
5. Review Dimension, Tech, TRL, Start-up fields
6. Save to summarised_content

### Example 4: Query Your Knowledge Base

1. Go to **RAG**
2. Select JSON file: `canarymedia_20251029.json`
3. Click **Build Index** (one-time)
4. Ask: "What are the latest solar panel innovations?"
5. Get cited answers: `[canarymedia]` references

---

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

- **Persistent storage**: Embeddings are saved to disk in rag_storage
- **No rebuild needed**: Load existing indexes instantly
- **Multiple sources**: Query across multiple indexes simultaneously

---

## Troubleshooting

### "503 Error" during crawl

**Cause**: Site's firewall blocked the crawler (too many 404s or requests)

**Solution**:
- Use **Sitemap** strategy
- Reduce crawl rate (add delays)
- Check `robots.txt` for restrictions

### "Progress display frozen"

**Cause**: Streamlit UI limitation during long-running tasks

**Solution**:
- Monitor in terminal: `watch -n 1 'ls -lh crawled_data/{folder} | tail -20'`
- Check output folder for new files
- The process IS running even if UI freezes

### "No JSON files found"

**Cause**: Summarization hasn't completed or files saved elsewhere

**Solution**:
- Check summarised_content folder
- Ensure summarization completed successfully
- Check processing history in Summarization tab

---

## Additional Resources

- **Logs**: Check research.log for detailed execution logs
- **Streamlit Docs**: https://docs.streamlit.io
- **LlamaIndex Docs**: https://docs.llamaindex.ai
- **Playwright Docs**: https://playwright.dev

---


