#!/usr/bin/env python3
"""
Web Scraper GUI

A modern graphical interface for the web scraper with real-time
progress tracking and results display.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper import WebScraper
from src.url_utils import ensure_scheme, is_valid_url


class WebScraperGUI:
    """Main GUI application for web scraper."""

    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Web Scraper - Extract Content from Websites")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        # State variables
        self.scraper = None
        self.is_crawling = False
        self.crawl_thread = None

        # Setup logging to capture in GUI
        self.setup_logging()

        # Create GUI components
        self.create_widgets()

        # Center window
        self.center_window()

    def setup_logging(self):
        """Setup logging to display in GUI."""
        self.log_handler = GUILogHandler(self)
        self.log_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                      datefmt='%H:%M:%S')
        self.log_handler.setFormatter(formatter)

        # Add handler to root logger
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)

    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)

        # === Title ===
        title_label = ttk.Label(main_frame, text="🕷️ Web Scraper",
                                font=('Helvetica', 18, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 10))

        # === Input Section ===
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)

        # Crawl mode
        ttk.Label(input_frame, text="Mode:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.mode_var = tk.StringVar(value="single")
        mode_frame = ttk.Frame(input_frame)
        mode_frame.grid(row=0, column=1, sticky=tk.W)
        ttk.Radiobutton(mode_frame, text="Single URLs", variable=self.mode_var,
                        value="single", command=self.on_mode_change).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(mode_frame, text="Full Site", variable=self.mode_var,
                        value="full", command=self.on_mode_change).pack(side=tk.LEFT)

        # URL input with helper text
        url_label_frame = ttk.Frame(input_frame)
        url_label_frame.grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Label(url_label_frame, text="URL:").pack(side=tk.LEFT)

        self.url_entry = ttk.Entry(input_frame, width=50)
        self.url_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        self.url_entry.insert(0, "bioenergy-news.com")

        # Mode helper text
        self.mode_help = ttk.Label(input_frame, text="Single URLs: Crawls one page only. Full Site: Follows links (use for multi-page crawls)",
                                   font=('Helvetica', 8), foreground='gray')
        self.mode_help.grid(row=0, column=2, sticky=tk.W, padx=(10, 0))

        # Max pages (for full site mode)
        ttk.Label(input_frame, text="Max Pages:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.max_pages_var = tk.StringVar(value="50")
        self.max_pages_entry = ttk.Entry(input_frame, textvariable=self.max_pages_var, width=10)
        self.max_pages_entry.grid(row=2, column=1, sticky=tk.W, pady=5, padx=(10, 0))

        # Max depth (for full site mode)
        ttk.Label(input_frame, text="Max Depth:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.max_depth_var = tk.StringVar(value="2")
        self.max_depth_entry = ttk.Entry(input_frame, textvariable=self.max_depth_var, width=10)
        self.max_depth_entry.grid(row=3, column=1, sticky=tk.W, pady=5, padx=(10, 0))

        # Crawl delay
        delay_label_frame = ttk.Frame(input_frame)
        delay_label_frame.grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Label(delay_label_frame, text="Delay (sec):").pack(side=tk.LEFT)

        self.delay_var = tk.StringVar(value="1.0")
        delay_entry_frame = ttk.Frame(input_frame)
        delay_entry_frame.grid(row=4, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        ttk.Entry(delay_entry_frame, textvariable=self.delay_var, width=10).pack(side=tk.LEFT)
        ttk.Label(delay_entry_frame, text=" (min delay, robots.txt respected)",
                 font=('Helvetica', 8), foreground='gray').pack(side=tk.LEFT, padx=(5, 0))

        # Clear tracking option
        self.clear_tracking_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(input_frame, text="Clear previous crawl tracking for this site (re-crawl all pages)",
                       variable=self.clear_tracking_var).grid(row=5, column=1, sticky=tk.W, pady=5, padx=(10, 0))

        # === Control Buttons ===
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, pady=(0, 10))

        self.start_button = ttk.Button(button_frame, text="▶ Start Crawl",
                                       command=self.start_crawl, style='Accent.TButton')
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(button_frame, text="⏹ Stop",
                                      command=self.stop_crawl, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="📂 Open Output Folder",
                   command=self.open_output_folder).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="🗑️ Clear Log",
                   command=self.clear_log).pack(side=tk.LEFT, padx=5)

        # === Output Section ===
        output_frame = ttk.LabelFrame(main_frame, text="Output & Progress", padding="10")
        output_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(1, weight=1)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(output_frame, variable=self.progress_var,
                                            maximum=100, mode='determinate')
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        # Status label
        self.status_var = tk.StringVar(value="Ready to crawl")
        self.status_label = ttk.Label(output_frame, textvariable=self.status_var,
                                      foreground='blue')
        self.status_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 5))

        # Log output
        log_frame = ttk.Frame(output_frame)
        log_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD,
                                                   font=('Courier', 9))
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # === Status Bar ===
        status_bar = ttk.Frame(main_frame, relief=tk.SUNKEN)
        status_bar.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        self.stats_var = tk.StringVar(value="Pages: 0 | Successful: 0 | Failed: 0 | Warnings: 0")
        ttk.Label(status_bar, textvariable=self.stats_var).pack(side=tk.LEFT, padx=5)

        # Initial mode setup
        self.on_mode_change()

    def center_window(self):
        """Center the window on screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def on_mode_change(self):
        """Handle mode change."""
        mode = self.mode_var.get()
        if mode == "single":
            self.max_pages_entry.config(state=tk.DISABLED)
            self.max_depth_entry.config(state=tk.DISABLED)
        else:
            self.max_pages_entry.config(state=tk.NORMAL)
            self.max_depth_entry.config(state=tk.NORMAL)

    def log_message(self, message):
        """Add message to log display."""
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)

    def clear_log(self):
        """Clear the log display."""
        self.log_text.delete(1.0, tk.END)
        self.log_message("Log cleared.")

    def update_status(self, message, color='blue'):
        """Update status message."""
        self.status_var.set(message)
        self.status_label.config(foreground=color)

    def update_stats(self, total, successful, failed, warnings=0):
        """Update statistics display."""
        self.stats_var.set(f"Pages: {total} | Successful: {successful} | Failed: {failed} | Warnings: {warnings}")

    def update_progress(self, current, total):
        """Update progress bar."""
        if total > 0:
            percentage = (current / total) * 100
            self.progress_var.set(percentage)

    def start_crawl(self):
        """Start the crawling process."""
        # Validate input
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter a URL")
            return

        # Add scheme if missing
        url = ensure_scheme(url)

        if not is_valid_url(url):
            messagebox.showerror("Error", "Invalid URL format")
            return

        # Get parameters
        try:
            delay = float(self.delay_var.get())
            if delay < 0:
                raise ValueError("Delay must be positive")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid delay value: {e}")
            return

        mode = self.mode_var.get()

        if mode == "full":
            try:
                max_pages = int(self.max_pages_var.get())
                max_depth = int(self.max_depth_var.get())
                if max_pages < 1 or max_depth < 0:
                    raise ValueError("Invalid range")
            except ValueError:
                messagebox.showerror("Error", "Max pages must be >= 1, max depth >= 0")
                return
        else:
            max_pages = 1
            max_depth = 0

        # Check robots.txt delay BEFORE starting crawl
        from src.robots_handler import RobotsHandler
        robots_handler = RobotsHandler(user_agent='WebScraperGUI/1.0')
        robots_delay = robots_handler.get_crawl_delay(url)

        # If robots.txt specifies a delay > 30 seconds, prompt user
        if robots_delay and robots_delay > 30:
            minutes = robots_delay / 60
            hours = robots_delay / 3600

            time_str = f"{robots_delay} seconds"
            if minutes >= 1:
                time_str = f"{minutes:.1f} minutes"
            if hours >= 1:
                time_str = f"{hours:.1f} hours"

            est_time = max_pages * robots_delay / 3600
            est_time_30s = max_pages * 30 / 3600

            response = messagebox.askyesno(
                "High Crawl Delay Detected",
                f"⚠️ WARNING: Extremely High Crawl Delay\n\n"
                f"Site: {url}\n"
                f"robots.txt delay: {time_str}\n\n"
                f"Estimated crawl time:\n"
                f"  • With site delay: ~{est_time:.1f} hours for {max_pages} pages\n"
                f"  • With 30s cap: ~{est_time_30s:.1f} hours for {max_pages} pages\n\n"
                f"Would you like to cap the delay at 30 seconds?\n\n"
                f"⚠️ Note: Capping may not fully respect robots.txt rules.\n\n"
                f"YES = Use 30 second cap (faster)\n"
                f"NO = Use site's delay (slower but compliant)",
                icon='warning'
            )

            if response:  # User clicked YES - cap at 30s
                delay = 30.0
                self.delay_var.set(str(delay))  # Update GUI to reflect capped delay
                self.log_message(f"⚠️ User chose to cap crawl delay at 30 seconds")
                self.log_message(f"   (Site requested {robots_delay}s = {time_str})")
            else:  # User clicked NO - use full delay
                delay = robots_delay
                self.delay_var.set(str(delay))  # Update GUI to reflect full delay
                self.log_message(f"⚠️ User chose to respect site's full delay: {time_str}")
                self.log_message(f"   Crawl will be very slow. Use Stop button to cancel.")


        # Update UI state
        self.is_crawling = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.url_entry.config(state=tk.DISABLED)
        self.progress_var.set(0)

        # Clear previous log
        self.log_text.delete(1.0, tk.END)

        # Update status
        self.update_status(f"Starting {'full site' if mode == 'full' else 'single URL'} crawl...", 'blue')

        # Log delay information
        self.log_message("=" * 60)
        self.log_message(f"Crawl Configuration:")
        self.log_message(f"  Minimum delay: {delay} seconds")
        self.log_message(f"  Note: robots.txt delays will be respected if higher")
        self.log_message("=" * 60)
        self.log_message("")

        # Start crawl in background thread
        self.crawl_thread = threading.Thread(
            target=self._run_crawl,
            args=(url, mode, delay, max_pages, max_depth),
            daemon=True
        )
        self.crawl_thread.start()

    def _run_crawl(self, url, mode, delay, max_pages, max_depth):
        """Run the crawl in a background thread."""
        try:
            # Clear tracking if requested
            if self.clear_tracking_var.get():
                self._clear_tracking_for_domain(url)

            # Create scraper
            self.scraper = WebScraper(
                user_agent='WebScraperGUI/1.0',
                default_delay=delay
            )

            # Run crawl
            if mode == "full":
                self.log_message(f"Starting full site crawl of {url}")
                self.log_message(f"Max pages: {max_pages}, Max depth: {max_depth}\n")

                summary = self.scraper.crawl_site(
                    start_url=url,
                    max_pages=max_pages,
                    max_depth=max_depth,
                    same_domain_only=True,
                    should_stop=lambda: not self.is_crawling
                )
            else:
                self.log_message(f"Crawling single URL: {url}\n")
                # Extract links even in single mode to show what's available
                result = self.scraper.crawl_url(url, extract_links=True, should_stop=lambda: not self.is_crawling)

                # Log links found
                if result.get('links'):
                    self.log_message(f"Found {len(result['links'])} links on this page")
                    self.log_message("(Switch to 'Full Site' mode to follow these links)\n")

                summary = {
                    'total': 1,
                    'successful': 1 if result['success'] else 0,
                    'failed': 0 if result['success'] else 1,
                    'warnings': 1 if not result['success'] and result.get('warning', False) else 0,
                    'results': [result],
                    'total_links_found': len(result.get('links', []))
                }

            # Update UI with results
            self.root.after(0, self._crawl_complete, summary)

        except Exception as e:
            logging.error(f"Crawl error: {e}", exc_info=True)
            self.root.after(0, self._crawl_error, str(e))

        finally:
            if self.scraper:
                self.scraper.close()

    def _crawl_complete(self, summary):
        """Handle crawl completion."""
        self.is_crawling = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.url_entry.config(state=tk.NORMAL)
        self.progress_var.set(100)

        # Update stats
        warnings_count = summary.get('warnings', 0)
        self.update_stats(summary['total'], summary['successful'], summary['failed'], warnings_count)

        # Update status
        if summary['successful'] > 0:
            self.update_status(
                f"✅ Crawl complete! {summary['successful']}/{summary['total']} pages successful",
                'green'
            )
            self.log_message(f"\n{'='*60}")
            self.log_message(f"✅ CRAWL COMPLETE!")
            self.log_message(f"{'='*60}")
            self.log_message(f"Total pages: {summary['total']}")
            self.log_message(f"Successful: {summary['successful']}")
            self.log_message(f"Failed: {summary['failed']}")
            self.log_message(f"Warnings: {warnings_count}")

            # Show output files
            if self.scraper:
                stats = self.scraper.get_stats()
                if stats['output_files']:
                    self.log_message(f"\nOutput files:")
                    for file in stats['output_files']:
                        self.log_message(f"  📄 {file}")

            messagebox.showinfo("Success",
                                f"Crawl complete!\n\n"
                                f"Pages crawled: {summary['successful']}/{summary['total']}\n"
                                f"Check data/ folder for CSV files.")
        else:
            self.update_status(f"❌ Crawl failed", 'red')
            messagebox.showwarning("Warning", "No pages were successfully crawled.")

    def _crawl_error(self, error_msg):
        """Handle crawl error."""
        self.is_crawling = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.url_entry.config(state=tk.NORMAL)

        self.update_status(f"❌ Error: {error_msg}", 'red')
        messagebox.showerror("Crawl Error", f"An error occurred:\n\n{error_msg}")

    def stop_crawl(self):
        """Stop the crawling process."""
        if self.is_crawling:
            self.update_status("Stopping crawl...", 'orange')
            self.is_crawling = False
            # Crawl loop will check this flag and exit gracefully after current page

    def _clear_tracking_for_domain(self, url):
        """Clear crawl tracking for the domain of the given URL."""
        from urllib.parse import urlparse
        import json

        try:
            domain = urlparse(url).netloc
            tracking_file = Path(__file__).parent.parent / 'data' / 'crawled_urls.json'

            if tracking_file.exists():
                with open(tracking_file, 'r') as f:
                    data = json.load(f)

                before = len(data)
                # Remove all URLs from this domain
                data = {k: v for k, v in data.items() if domain not in k}
                after = len(data)

                with open(tracking_file, 'w') as f:
                    json.dump(data, f, indent=2)

                removed = before - after
                if removed > 0:
                    self.log_message(f"✓ Cleared {removed} previously crawled URLs from {domain}")
                else:
                    self.log_message(f"✓ No previous crawl history found for {domain}")
            else:
                self.log_message("✓ No tracking file found (fresh start)")

        except Exception as e:
            self.log_message(f"⚠ Warning: Could not clear tracking: {e}")

    def open_output_folder(self):
        """Open the output folder in file manager."""
        data_dir = Path(__file__).parent.parent / 'data'
        data_dir.mkdir(exist_ok=True)

        import platform
        if platform.system() == 'Windows':
            os.startfile(data_dir)
        elif platform.system() == 'Darwin':  # macOS
            os.system(f'open "{data_dir}"')
        else:  # Linux
            os.system(f'xdg-open "{data_dir}"')


class GUILogHandler(logging.Handler):
    """Custom log handler that sends logs to GUI."""

    def __init__(self, gui):
        super().__init__()
        self.gui = gui

    def emit(self, record):
        """Emit a log record to the GUI."""
        try:
            msg = self.format(record)
            # Schedule GUI update in main thread
            self.gui.root.after(0, self.gui.log_message, msg)

            # Update progress if this is a crawl progress message
            if '[' in msg and '/' in msg and ']' in msg:
                try:
                    # Extract progress like "[5/100]"
                    start = msg.index('[') + 1
                    end = msg.index(']')
                    progress = msg[start:end]
                    current, total = map(int, progress.split('/'))
                    self.gui.root.after(0, self.gui.update_progress, current, total)
                except (ValueError, IndexError):
                    pass

        except Exception:
            pass


def main():
    """Launch the GUI application."""
    root = tk.Tk()

    # Try to set a nice theme
    try:
        style = ttk.Style()
        # Use 'clam' theme which looks good on most platforms
        style.theme_use('clam')
    except:
        pass

    app = WebScraperGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()