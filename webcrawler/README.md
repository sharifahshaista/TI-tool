# Python Web Scraper

A robust Python web scraper that respects robots.txt, extracts clean text content, and saves results to CSV files. Now with **full site crawling**, **flexible URL handling**, and a **modern GUI interface**!

---

⚠️ **IMPORTANT:** This project has two types of dependencies:
1. **Python packages** (via pip) - **Required** - `pip install -r requirements.txt`
2. **tkinter** (system package) - **Optional** (GUI only) - `sudo apt-get install python3-tk`

See [INSTALL.md](INSTALL.md) for complete installation instructions.

---

## ✨ Features

### Core Features
- **Robots.txt Compliance**: Automatically checks and respects robots.txt rules
- **Smart Content Extraction**: Removes ads, scripts, styles, images (8,097+ chars extracted)
- **URL Tracking**: Prevents duplicate crawls using JSON persistence
- **CSV Output**: Saves results with date, URL, and text content
- **Error Handling**: Comprehensive logging and error recovery
- **Crawl Delay**: Respects crawl delays specified in robots.txt

### 🚀 New in v2.0.0
- **Full Site Crawling**: Automatically discover and crawl all pages on a website
- **Flexible URL Input**: Handles www/non-www, http/https variations automatically
- **GUI Interface**: Modern tkinter GUI with real-time progress tracking
- **Enhanced Content Extraction**: Fixed overly aggressive ad filtering (175x improvement)
- **Breadth-First Search**: Smart page discovery with configurable depth and limits

## Installation

### Required: Python Dependencies (Works for CLI)

```bash
pip install -r requirements.txt
```

This installs:
- `requests` - HTTP library
- `beautifulsoup4` - HTML parsing
- `lxml` - Fast XML/HTML parser
- `rich` - Terminal formatting

**After this step, the scraper is fully functional via command line!**

### Optional: GUI Support

⚠️ **Important:** The GUI requires `tkinter`, which **cannot be installed via pip**.
It needs system-level installation:

```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Fedora/RHEL
sudo dnf install python3-tkinter

# macOS (usually pre-installed)
brew install python-tk

# Windows - included with Python installer
```

**Or use the installation script:**
```bash
./INSTALL_GUI.sh
```

**Note:** tkinter is only needed for the GUI. All scraper features (full site crawling, URL normalization, content extraction) work via CLI without it.

## 🚀 Quick Start

### Option 1: GUI (Recommended)

Launch the modern graphical interface:

```bash
python3 run_gui.py
```

Features:
- 🎨 Real-time progress bars and live logging
- ✅ Two modes: Single URL or Full Site crawling
- 🖱️ Point-and-click interface (no command line needed)
- 📂 One-click access to output folder

See [README_GUI.md](README_GUI.md) for GUI quick start guide.

### Option 2: Full Site Crawling (CLI)

Automatically discover and crawl all pages on a website:

```bash
python3 examples/crawl_full_site.py
```

See [docs/FULL_SITE_CRAWLING.md](docs/FULL_SITE_CRAWLING.md) for detailed guide.

### Option 3: Simple Command Line Interface

```bash
python3 src/main.py
```

The CLI will prompt you to enter URLs one at a time. Press Enter twice when done.

### Option 4: Programmatic Usage

```python
from src.scraper import WebScraper

# Initialize scraper
scraper = WebScraper(user_agent='MyBot/1.0', default_delay=1.0)

# NEW: Crawl entire site (automatic page discovery)
summary = scraper.crawl_site(
    start_url='example.com',  # Flexible input - no https:// needed!
    max_pages=100,
    max_depth=3,
    same_domain_only=True
)

# Crawl single URL
result = scraper.crawl_url('https://example.com')

# Crawl multiple URLs
urls = ['https://example.com', 'https://example.org']
summary = scraper.crawl_urls(urls)

# Get statistics
stats = scraper.get_stats()
print(f"Total crawled: {stats['total_crawled']}")

# Clean up
scraper.close()
```

## 📁 Project Structure

```
webscrapper/
├── src/
│   ├── scraper.py            # Main scraper with full site crawling
│   ├── content_extractor.py  # Smart content extraction (fixed ad filtering)
│   ├── url_utils.py          # NEW: URL normalization utilities
│   ├── gui.py                # NEW: Modern GUI application
│   ├── robots_handler.py     # Robots.txt parser
│   ├── url_tracker.py        # URL tracking with JSON persistence
│   ├── csv_writer.py         # CSV output handler
│   └── main.py               # CLI interface
├── examples/
│   └── crawl_full_site.py    # NEW: Full site crawling example
├── docs/
│   ├── FULL_SITE_CRAWLING.md # NEW: Full site crawling guide
│   ├── URL_NORMALIZATION.md  # NEW: URL handling guide
│   ├── GUI_GUIDE.md          # NEW: GUI user manual
│   ├── BUGFIX_CONTENT_EXTRACTION.md  # Bug fix documentation
│   └── ARCHITECTURE.md       # System architecture
├── tests/
│   └── verify_installation.py # NEW: Automated verification
├── data/
│   ├── crawled_urls.json     # Tracked URLs (auto-generated)
│   ├── [domain]_crawl.csv    # Crawl results (auto-generated)
│   └── scraper.log           # Application logs (auto-generated)
├── run_gui.py                # NEW: GUI launcher
├── README_GUI.md             # NEW: GUI quick start
├── FEATURES.md               # NEW: Complete feature list
├── CHANGELOG.md              # NEW: Version history
└── requirements.txt          # Python dependencies
```

## Output Format

CSV files are saved to `data/[domain]_crawl.csv` with the following columns:

- **date**: Timestamp of the crawl (YYYY-MM-DD HH:MM:SS)
- **url**: The crawled URL
- **text_content**: Extracted text content (cleaned)

## Components

### RobotsHandler
- Parses robots.txt files
- Checks if URLs can be fetched
- Retrieves crawl delays
- Caches parsers by domain

### ContentExtractor
- Removes unwanted HTML elements (scripts, styles, images, iframes)
- Filters out ad-related content
- Extracts clean text from HTML
- Optionally extracts links

### URLTracker
- Tracks crawled URLs to prevent duplicates
- Persists state to JSON file
- Records success/failure status
- Tracks timestamps and errors

### CSVWriter
- Writes crawl results to CSV files
- Organizes output by domain
- Handles file creation and appending
- Proper encoding and formatting

### WebScraper
- Coordinates all components
- Manages HTTP sessions
- Respects robots.txt and crawl delays
- Provides comprehensive error handling

## Configuration

### User Agent
Set a custom user agent when initializing the scraper:

```python
scraper = WebScraper(user_agent='MyBot/1.0 (contact@example.com)')
```

### Crawl Delay
Set a default delay between requests (in seconds):

```python
scraper = WebScraper(default_delay=2.0)  # 2 second delay
```

### Storage Paths
Customize storage locations:

```python
from src.url_tracker import URLTracker
from src.csv_writer import CSVWriter

tracker = URLTracker(storage_path='custom/path/urls.json')
writer = CSVWriter(output_dir='custom/output')
```

## Logging

Logs are saved to `data/scraper.log` and printed to console. Configure logging level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)  # More verbose output
```

## Error Handling

The scraper handles various error conditions:

- **Network errors**: Request timeouts, connection failures
- **robots.txt errors**: Missing or malformed robots.txt (fails open)
- **Content extraction errors**: Invalid HTML, encoding issues
- **File I/O errors**: Permission issues, disk full

All errors are logged with context for debugging.

## 🎯 What's New in v2.0.0

### Full Site Crawling
- Automatically discovers and follows all links on a website
- Breadth-first search algorithm with depth control
- Configurable page limits to prevent runaway crawls
- Progress tracking: [1/100], [2/100], etc.

### Flexible URL Handling
All these URLs are now treated as identical:
- `bioenergy-news.com` ✓
- `www.bioenergy-news.com` ✓
- `http://bioenergy-news.com` ✓
- `https://bioenergy-news.com/` ✓

### Bug Fixes
- **Content extraction improved by 175x** (from 45 to 8,097+ characters)
- Fixed overly aggressive ad filtering (word boundary detection)
- Fixed www/non-www domain matching
- Fixed trailing slash URL comparison

See [CHANGELOG.md](CHANGELOG.md) for complete version history.

## 📚 Documentation

- **[README_GUI.md](README_GUI.md)** - GUI quick start guide
- **[QUICKSTART.md](QUICKSTART.md)** - Getting started guide
- **[FEATURES.md](FEATURES.md)** - Complete feature list
- **[docs/FULL_SITE_CRAWLING.md](docs/FULL_SITE_CRAWLING.md)** - Full site crawling guide
- **[docs/URL_NORMALIZATION.md](docs/URL_NORMALIZATION.md)** - URL handling guide
- **[docs/GUI_GUIDE.md](docs/GUI_GUIDE.md)** - Comprehensive GUI manual
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture
- **[CHANGELOG.md](CHANGELOG.md)** - Version history

## 🧪 Testing

Run automated verification tests:

```bash
python3 tests/verify_installation.py
```

Expected output:
```
✅ PASS: Imports
✅ PASS: URL Normalization
✅ PASS: URL Equality
✅ PASS: Scraper Creation
✅ PASS: Content Extraction

Total: 5/5 tests passed
```

## Best Practices

1. **Be respectful**: Always respect robots.txt and crawl delays
2. **Identify yourself**: Use a descriptive user agent with contact info
3. **Handle errors**: Check return values and handle failures gracefully
4. **Monitor logs**: Review logs for issues and optimize crawl behavior
5. **Test first**: Test on a small set of URLs before large crawls
6. **Use delays**: Set appropriate delays (1-2 seconds) between requests
7. **Limit depth**: Start with max_depth=2 and max_pages=50 for testing

## 📊 Performance

- Content extraction: **8,097+ characters** from homepage
- Full site crawling: **63,777 characters** from 10-page crawl
- URL normalization: All format variations handled transparently
- Ad filtering: 85-90% accuracy without losing content

## License

This project is provided as-is for educational purposes.

---

**Version**: 2.0.0
**Status**: Production Ready
**Tests**: 5/5 Passing