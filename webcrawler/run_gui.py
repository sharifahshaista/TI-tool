#!/usr/bin/env python3
"""
Web Scraper GUI Launcher

Simple launcher script for the web scraper GUI.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Check if tkinter is available
try:
    import tkinter
except ImportError:
    print("=" * 60)
    print("❌ GUI Not Available - tkinter is not installed")
    print("=" * 60)
    print()
    print("Tkinter is Python's GUI library but requires system installation.")
    print()
    print("To install tkinter:")
    print("  Ubuntu/Debian: sudo apt-get install python3-tk")
    print("  Fedora/RHEL:   sudo dnf install python3-tkinter")
    print("  macOS:         brew install python-tk")
    print("  Windows:       Included with Python installer")
    print()
    print("Or run the installation script:")
    print("  ./INSTALL_GUI.sh")
    print()
    print("=" * 60)
    print("Alternative: Use CLI Instead (No Installation Needed)")
    print("=" * 60)
    print()
    print("All features work via command line:")
    print()
    print("  1. Interactive CLI:")
    print("     python3 src/main.py")
    print()
    print("  2. Full site crawling:")
    print("     python3 examples/crawl_full_site.py")
    print()
    print("  3. Quick Python script:")
    print("     python3 -c \"from src.scraper import WebScraper; \\")
    print("                 s = WebScraper(); \\")
    print("                 s.crawl_site('example.com', 10, 2)\"")
    print()
    sys.exit(1)

from src.gui import main

if __name__ == '__main__':
    main()