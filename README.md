# TI-tool
This is a prototype tool to complement technology intelligence.

## Overview

TI Agent is a comprehensive Streamlit application that automates the entire workflow of technology intelligence gathering:

1. **Web Search** - AI-assisted research with clarification and SERP generation
2. **Web Crawler** - Intelligent website crawling with multiple strategies
3. **Post-Processing** - Extract structured metadata from crawled content
4. **Summarization** - AI-powered tech-intelligence analysis and categorization
5. **Database** - Consolidated searchable database with advanced filtering
6. **RAG Chatbot** - Query your knowledge base with AI-powered citations

---

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key or Azure OpenAI credentials
- SearXNG instance (for web search)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd TI-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Required Environment Variables

```bash
OPENAI_API_KEY=your_openai_key # Used for embeddings
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_VERSION=2024-02-15-preview
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
TI-agent/
├── app.py                      # Main Streamlit application
├── embeddings_rag.py           # LlamaIndex RAG system with persistence
├── embeddings_demo.py          # Embedding utilities
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create from .env.example)
│
├── agents/                     # AI agents and processors
│   ├── clarification.py        # Research clarification agent
│   ├── serp.py                 # SERP query generation
│   ├── learn.py                # Learning extraction agent
│   ├── web_crawler.py          # Multi-strategy web crawler
│   ├── markdown_post_processor.py  # Metadata extraction
│   ├── summarise_csv.py        # Tech-intelligence summarizer
│   └── crawl_strategy_detector.py  # Auto-detect best crawl strategy
│
├── config/                     # Configuration modules
│   ├── model_config.py         # LLM provider configuration
│   └── searxng_tools.py        # SearXNG integration
│
├── schemas/                    # Pydantic data models
│   └── datamodel.py            # Data schemas and validation
│
├── data/                       # Research pipeline outputs
│   ├── *.json                  # Search results
│   └── *.md                    # Learning reports
│
├── crawled_data/               # Web crawler outputs
│   └── {website}/              # One folder per crawled site
│       ├── *.md                # Markdown content
│       ├── csv/                # CSV exports
│       └── json/               # JSON exports
│
├── processed_data/             # Post-processed data
│   └── {website}_{date}.csv/json
│
├── summarised_content/         # Tech-intelligence summaries
│   ├── {website}_{date}.csv    # Summarized articles
│   ├── {website}_{date}.json   # JSON format
│   ├── {website}_{date}_log.txt # Processing logs
│   └── history.json            # Processing history
│
└── rag_storage/                # Persistent vector embeddings
    └── {source}_{date}/        # One index per data source
        ├── docstore.json       # Document storage
        ├── index_store.json    # Index metadata
        ├── vector_store.json   # Vector embeddings
        └── metadata.json       # Custom metadata
```

---

## Features

### 1. Web Search Pipeline

- **AI Clarification**: Refines research scope with targeted questions
- **SERP Generation**: Creates optimized search queries
- **Web Search**: Executes searches via SearXNG
- **Learning Extraction**: Extracts structured insights from results

### 2. Web Crawler

- **Automatically determine** the optimal strategy to scrape websites from the user's input, chosen from six strategies. Users can manually select crawling strategy as well.

   * **Simple Discovery**  
     > Starts with the input URL, discovers all internal links on the page, and crawls each discovered link sequentially.

   * **Sitemap**  
     > Finds and parses XML sitemaps of the input URL. Prioritizes the largest numbered links containing terms such as *"post-sitemap"*, *"news"*, *"articles"*, or *"post"* (e.g., `post-sitemap17.xml`).

   * **Pagination**  
     > Automatically tests different URL-based and query-based pagination patterns (e.g., `/page/2`, `?page=2`, `/p2`, `/p/2`), then crawls links discovered on each page.

   * **Category**  
     > Focuses on a specific category or topic path, performing deep crawling with content filtering.  
     > Default category pages include `/articles/`, `/news/`, `/blog/`, and `/reports/`.  
     > The crawler stays within these category domains.

   * **Breadth-first Search (BFS) Deep Crawl**  
     > Crawls all pages at depth 1, then all pages at depth 2 and so on. Configurable by `max_depth`
     > Uses BFS traversal algorithm
     > Suitable if website structure is being studied.

   * **Depth-first Search (DFS) Deep Crawl**  
     > Crawls each pages from depth 1 to `max_depth`.
     > Uses DFS traversal algorithm
     > Suitable for content chains and focused explorations.

 - **Crawls** websites using [Crawl4AI](https://docs.crawl4ai.com/). Each website page is saved as markdown files. Crawled data also exportable as CSV and JSON.


### 3. Post-processing
1. Extracts the following metadata from each markdown file from crawled data folder and saves into CSV/JSON:
   * Title
   * Publication Date
   * URL
   * Main Content
   * Author
   * Tags/Categories
2. AI pattern detection is integrated to recognise regular expressions of the abovementioned metadata as well as to filter out noise in each markdown file.
   * *If disabled, the processing uses references from preset post-processing approaches for Canary Media, Carbon Capture Magazine, Cleantechnica, Cool Coalition and ThinkGeoEnergy.*

### 4. Summarisation 
Pass the 'Main Content' data from post-processing to perform summarisation and classification.
   * Indicator: A concise summary focusing on the key technological development, event, or trend described.
   * Dimension: Primary category from tech, policy, economic, environmental & safety, social & ethical, legal & regulatory.
   * Tech: Specific technology domain or sector.
   * TRL: Technology Readiness Level (1-9 scale).
   * Start-up: If the news is about a start-up, the URL to the start-up's official webpage is included.

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

https://github.com/user-attachments/assets/040f88ba-3a4d-4543-9ded-d297d0175d12

---

## Typical Workflow

```
1. Web Crawler
   ↓
   Crawl target websites
   ↓
   Save markdown files

2. Post-Processing
   ↓
   Extract structured metadata
   ↓
   Save to processed_data/

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

- **Azure OpenAI** (default): `pmo-gpt-4.1-nano`, `gpt-4o`, `gpt-4o-mini`
- **OpenAI**: Standard OpenAI models
- **LM Studio**: Local models
- **Anthropic**: Claude models
- **Groq**: Fast inference

Configure in the Summarization or RAG pages.

### Crawl Strategies

Choose the best strategy for your target:

| Strategy | Best For | Speed |
|----------|----------|-------|
| **Sitemap** | Sites with XML sitemaps | Fast ⚡ |
| **Pagination** | News sites with page numbers | Medium |
| **Category** | Sites with topic categories | Medium |
| **Deep BFS** | Comprehensive coverage | Slow |
| **Deep DFS** | Following content chains | Slow |

---

## Usage Examples

### Example 1: Crawl a News Site

1. Go to **Web Crawler**
2. Enter URL: `https://www.canarymedia.com`
3. Click **Auto-Detection**
4. Review recommended strategy (likely "Sitemap")
5. Set max pages: `500`
6. Click **Start Crawling**
7. Wait for completion (~2-5 minutes)

### Example 2: Process Crawled Content

1. Go to **Post-Processing** → **Quick Start**
2. Select preset: "Canary Media"
3. Select folder: `canarymedia`
4. Click **Start Quick Processing**
5. Files saved to `processed_data/canarymedia_20251029.csv`

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

- **Stay on page**: Navigating away interrupts the crawl
- **Rate limiting**: Some sites may block aggressive crawling
- **Respect robots.txt**: Be a good web citizen

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


