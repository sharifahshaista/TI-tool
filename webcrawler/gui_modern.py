#!/usr/bin/env python3
"""
Modern Web Scraper GUI with Dark Theme

A beautiful, modern graphical interface for the web scraper with:
- Dark theme with modern color palette
- Custom styled components
- Smooth animations and hover effects
- Enhanced typography
- Better visual hierarchy
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


# Modern Color Palette (Dark Theme)
COLORS = {
    'bg_dark': '#1e1e2e',           # Dark purple-gray background
    'bg_surface': '#2d2d44',         # Lighter surface panels
    'bg_card': '#363650',            # Card backgrounds
    'primary': '#6c63ff',            # Modern purple
    'primary_hover': '#5a52d5',      # Darker purple for hover
    'accent': '#00d4ff',             # Cyan accent
    'accent_dim': '#00a8cc',         # Dimmer cyan
    'text_primary': '#e0e0e0',       # Light gray text
    'text_secondary': '#a0a0b0',     # Dimmer text
    'text_disabled': '#606070',      # Disabled text
    'success': '#00e676',            # Bright green
    'warning': '#ffc107',            # Amber/orange
    'error': '#ff5252',              # Red
    'border': '#404055',             # Subtle borders
    'border_focus': '#6c63ff',       # Focused borders
}


class ModernWebScraperGUI:
    """Modern GUI application for web scraper with dark theme."""

    def __init__(self, root):
        """Initialize the modern GUI."""
        self.root = root
        self.root.title("Web Scraper - Modern Interface")
        self.root.geometry("1000x750")
        self.root.minsize(900, 650)

        # Configure root background
        self.root.configure(bg=COLORS['bg_dark'])

        # State variables
        self.scraper = None
        self.is_crawling = False
        self.crawl_thread = None

        # Setup modern theme
        self.setup_modern_theme()

        # Setup logging to capture in GUI
        self.setup_logging()

        # Create GUI components
        self.create_widgets()

        # Center window
        self.center_window()

    def setup_modern_theme(self):
        """Configure modern ttk theme with dark colors."""
        style = ttk.Style()

        # Use 'clam' as base (most customizable)
        style.theme_use('clam')

        # Configure general styles
        style.configure('.',
            background=COLORS['bg_dark'],
            foreground=COLORS['text_primary'],
            bordercolor=COLORS['border'],
            darkcolor=COLORS['bg_surface'],
            lightcolor=COLORS['bg_surface'],
            troughcolor=COLORS['bg_surface'],
            focuscolor=COLORS['primary'],
            selectbackground=COLORS['primary'],
            selectforeground='white',
            fieldbackground=COLORS['bg_surface'],
            font=('Segoe UI', 10)
        )

        # Frame styles
        style.configure('TFrame', background=COLORS['bg_dark'])
        style.configure('Card.TFrame',
            background=COLORS['bg_card'],
            relief='flat',
            borderwidth=1
        )

        # LabelFrame (card style)
        style.configure('TLabelframe',
            background=COLORS['bg_surface'],
            bordercolor=COLORS['border'],
            relief='flat',
            borderwidth=2
        )
        style.configure('TLabelframe.Label',
            background=COLORS['bg_surface'],
            foreground=COLORS['accent'],
            font=('Segoe UI', 11, 'bold')
        )

        # Label styles
        style.configure('TLabel',
            background=COLORS['bg_dark'],
            foreground=COLORS['text_primary'],
            font=('Segoe UI', 10)
        )
        style.configure('Title.TLabel',
            background=COLORS['bg_dark'],
            foreground=COLORS['primary'],
            font=('Segoe UI', 26, 'bold')
        )
        style.configure('Heading.TLabel',
            background=COLORS['bg_surface'],
            foreground=COLORS['accent'],
            font=('Segoe UI', 12, 'bold')
        )
        style.configure('Status.TLabel',
            background=COLORS['bg_surface'],
            foreground=COLORS['accent'],
            font=('Segoe UI', 10, 'bold')
        )
        style.configure('Secondary.TLabel',
            background=COLORS['bg_surface'],
            foreground=COLORS['text_secondary'],
            font=('Segoe UI', 9)
        )

        # Button styles - Primary
        style.configure('Primary.TButton',
            background=COLORS['primary'],
            foreground='white',
            bordercolor=COLORS['primary'],
            focuscolor='none',
            borderwidth=0,
            relief='flat',
            font=('Segoe UI', 10, 'bold'),
            padding=(20, 10)
        )
        style.map('Primary.TButton',
            background=[('active', COLORS['primary_hover']), ('pressed', COLORS['primary_hover'])],
            bordercolor=[('focus', COLORS['primary_hover'])]
        )

        # Button styles - Accent
        style.configure('Accent.TButton',
            background=COLORS['accent'],
            foreground=COLORS['bg_dark'],
            bordercolor=COLORS['accent'],
            focuscolor='none',
            borderwidth=0,
            relief='flat',
            font=('Segoe UI', 10, 'bold'),
            padding=(20, 10)
        )
        style.map('Accent.TButton',
            background=[('active', COLORS['accent_dim']), ('pressed', COLORS['accent_dim'])],
        )

        # Button styles - Secondary
        style.configure('Secondary.TButton',
            background=COLORS['bg_surface'],
            foreground=COLORS['text_primary'],
            bordercolor=COLORS['border'],
            focuscolor='none',
            borderwidth=1,
            relief='flat',
            font=('Segoe UI', 10),
            padding=(15, 8)
        )
        style.map('Secondary.TButton',
            background=[('active', COLORS['bg_card']), ('pressed', COLORS['bg_card'])],
            bordercolor=[('focus', COLORS['accent'])]
        )

        # Button styles - Danger (for stop)
        style.configure('Danger.TButton',
            background=COLORS['error'],
            foreground='white',
            bordercolor=COLORS['error'],
            focuscolor='none',
            borderwidth=0,
            relief='flat',
            font=('Segoe UI', 10, 'bold'),
            padding=(20, 10)
        )
        style.map('Danger.TButton',
            background=[('active', '#e04040'), ('pressed', '#e04040')]
        )

        # Entry styles
        style.configure('TEntry',
            fieldbackground=COLORS['bg_surface'],
            background=COLORS['bg_surface'],
            foreground=COLORS['text_primary'],
            bordercolor=COLORS['border'],
            insertcolor=COLORS['text_primary'],
            selectbackground=COLORS['primary'],
            selectforeground='white',
            relief='flat',
            borderwidth=2,
            padding=10
        )
        style.map('TEntry',
            bordercolor=[('focus', COLORS['border_focus'])]
        )

        # Radiobutton styles
        style.configure('TRadiobutton',
            background=COLORS['bg_surface'],
            foreground=COLORS['text_primary'],
            indicatorcolor=COLORS['bg_surface'],
            selectcolor=COLORS['primary'],
            borderwidth=0,
            font=('Segoe UI', 10)
        )
        style.map('TRadiobutton',
            background=[('active', COLORS['bg_surface'])],
            foreground=[('active', COLORS['accent'])]
        )

        # Progressbar styles
        style.configure('TProgressbar',
            background=COLORS['primary'],
            troughcolor=COLORS['bg_surface'],
            bordercolor=COLORS['border'],
            lightcolor=COLORS['primary'],
            darkcolor=COLORS['primary'],
            thickness=20
        )

    def setup_logging(self):
        """Setup logging to display in GUI."""
        self.log_handler = ModernGUILogHandler(self)
        self.log_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                      datefmt='%H:%M:%S')
        self.log_handler.setFormatter(formatter)

        # Add handler to root logger
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)

    def create_widgets(self):
        """Create all GUI widgets with modern styling."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="16")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)

        # === Modern Title Header ===
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))

        title_label = ttk.Label(header_frame,
            text="🕷️ Web Scraper",
            style='Title.TLabel'
        )
        title_label.pack(side=tk.LEFT)

        subtitle_label = ttk.Label(header_frame,
            text="Modern Interface",
            font=('Segoe UI', 11),
            foreground=COLORS['text_secondary']
        )
        subtitle_label.pack(side=tk.LEFT, padx=(10, 0))

        # === Input Card ===
        input_frame = ttk.LabelFrame(main_frame, text="⚙️ Configuration", padding="16")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 16))
        input_frame.columnconfigure(1, weight=1)

        row = 0

        # Crawl mode with modern radio buttons
        mode_label = ttk.Label(input_frame, text="Crawl Mode")
        mode_label.grid(row=row, column=0, sticky=tk.W, pady=(0, 8))

        self.mode_var = tk.StringVar(value="single")
        mode_frame = ttk.Frame(input_frame)
        mode_frame.grid(row=row, column=1, sticky=tk.W, pady=(0, 8))

        single_radio = ttk.Radiobutton(mode_frame,
            text="📄 Single URLs",
            variable=self.mode_var,
            value="single",
            command=self.on_mode_change
        )
        single_radio.pack(side=tk.LEFT, padx=(0, 20))

        full_radio = ttk.Radiobutton(mode_frame,
            text="🌐 Full Site Crawl",
            variable=self.mode_var,
            value="full",
            command=self.on_mode_change
        )
        full_radio.pack(side=tk.LEFT)

        row += 1

        # URL input with icon
        ttk.Label(input_frame, text="🔗 Target URL").grid(
            row=row, column=0, sticky=tk.W, pady=(8, 8)
        )
        self.url_entry = ttk.Entry(input_frame, width=60, font=('Segoe UI', 10))
        self.url_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=(8, 8), padx=(16, 0))
        self.url_entry.insert(0, "bioenergy-news.com")

        row += 1

        # Two-column layout for numeric inputs
        params_frame = ttk.Frame(input_frame)
        params_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(8, 0))
        params_frame.columnconfigure(1, weight=1)
        params_frame.columnconfigure(3, weight=1)

        # Max pages
        ttk.Label(params_frame, text="📊 Max Pages").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 8)
        )
        self.max_pages_var = tk.StringVar(value="50")
        self.max_pages_entry = ttk.Entry(params_frame,
            textvariable=self.max_pages_var,
            width=10,
            font=('Segoe UI', 10)
        )
        self.max_pages_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 32))

        # Max depth
        ttk.Label(params_frame, text="🔢 Max Depth").grid(
            row=0, column=2, sticky=tk.W, padx=(0, 8)
        )
        self.max_depth_var = tk.StringVar(value="2")
        self.max_depth_entry = ttk.Entry(params_frame,
            textvariable=self.max_depth_var,
            width=10,
            font=('Segoe UI', 10)
        )
        self.max_depth_entry.grid(row=0, column=3, sticky=tk.W)

        row += 1

        # Crawl delay
        delay_frame = ttk.Frame(input_frame)
        delay_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(16, 0))

        ttk.Label(delay_frame, text="⏱️ Request Delay").pack(side=tk.LEFT)

        self.delay_var = tk.StringVar(value="1.0")
        delay_entry = ttk.Entry(delay_frame,
            textvariable=self.delay_var,
            width=10,
            font=('Segoe UI', 10)
        )
        delay_entry.pack(side=tk.LEFT, padx=(16, 8))

        ttk.Label(delay_frame, text="seconds",
            foreground=COLORS['text_secondary'],
            font=('Segoe UI', 9)
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(delay_frame,
            text="(robots.txt rules will be respected)",
            style='Secondary.TLabel'
        ).pack(side=tk.LEFT)

        # === Control Panel ===
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, pady=(0, 16))

        self.start_button = ttk.Button(control_frame,
            text="▶️ Start Crawl",
            command=self.start_crawl,
            style='Accent.TButton',
            width=15
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 8))

        self.stop_button = ttk.Button(control_frame,
            text="⏹️ Stop",
            command=self.stop_crawl,
            state=tk.DISABLED,
            style='Danger.TButton',
            width=12
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(control_frame,
            text="📂 Output Folder",
            command=self.open_output_folder,
            style='Secondary.TButton'
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(control_frame,
            text="🗑️ Clear Log",
            command=self.clear_log,
            style='Secondary.TButton'
        ).pack(side=tk.LEFT)

        # === Output & Progress Card ===
        output_frame = ttk.LabelFrame(main_frame, text="📊 Progress & Output", padding="16")
        output_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(2, weight=1)

        # Status with icon
        status_container = ttk.Frame(output_frame)
        status_container.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        self.status_icon = ttk.Label(status_container,
            text="●",
            foreground=COLORS['accent'],
            font=('Segoe UI', 14)
        )
        self.status_icon.pack(side=tk.LEFT, padx=(0, 8))

        self.status_var = tk.StringVar(value="Ready to crawl")
        self.status_label = ttk.Label(status_container,
            textvariable=self.status_var,
            style='Status.TLabel'
        )
        self.status_label.pack(side=tk.LEFT)

        # Modern progress bar with percentage
        progress_container = ttk.Frame(output_frame)
        progress_container.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 16))
        progress_container.columnconfigure(0, weight=1)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_container,
            variable=self.progress_var,
            maximum=100,
            mode='determinate',
            length=400
        )
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 12))

        self.progress_label = ttk.Label(progress_container,
            text="0%",
            font=('Segoe UI', 10, 'bold'),
            foreground=COLORS['accent'],
            width=5
        )
        self.progress_label.grid(row=0, column=1)

        # Log output with modern styling
        log_container = ttk.Frame(output_frame)
        log_container.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_container.columnconfigure(0, weight=1)
        log_container.rowconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_container,
            height=15,
            wrap=tk.WORD,
            font=('Consolas', 9),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_primary'],
            insertbackground=COLORS['accent'],
            selectbackground=COLORS['primary'],
            selectforeground='white',
            relief='flat',
            borderwidth=0,
            padx=12,
            pady=12
        )
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure log text tags for colored output
        self.log_text.tag_config('INFO', foreground=COLORS['text_primary'])
        self.log_text.tag_config('SUCCESS', foreground=COLORS['success'])
        self.log_text.tag_config('WARNING', foreground=COLORS['warning'])
        self.log_text.tag_config('ERROR', foreground=COLORS['error'])
        self.log_text.tag_config('HEADER', foreground=COLORS['accent'], font=('Consolas', 9, 'bold'))

        # === Modern Status Bar ===
        status_bar_frame = ttk.Frame(main_frame)
        status_bar_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(16, 0))
        status_bar_frame.configure(style='Card.TFrame')

        # Statistics with icons
        stats_container = ttk.Frame(status_bar_frame)
        stats_container.pack(fill=tk.X, padx=12, pady=8)

        self.stats_labels = {}
        stats_items = [
            ('total', '📄', 'Pages', '0'),
            ('success', '✅', 'Success', '0'),
            ('failed', '❌', 'Failed', '0'),
            ('warnings', '⚠️', 'Warnings', '0')
        ]

        for key, icon, label, value in stats_items:
            frame = ttk.Frame(stats_container)
            frame.pack(side=tk.LEFT, padx=(0, 24))

            ttk.Label(frame,
                text=icon,
                font=('Segoe UI', 11)
            ).pack(side=tk.LEFT, padx=(0, 4))

            ttk.Label(frame,
                text=label + ':',
                foreground=COLORS['text_secondary'],
                font=('Segoe UI', 9)
            ).pack(side=tk.LEFT, padx=(0, 4))

            stat_label = ttk.Label(frame,
                text=value,
                foreground=COLORS['accent'],
                font=('Segoe UI', 10, 'bold')
            )
            stat_label.pack(side=tk.LEFT)
            self.stats_labels[key] = stat_label

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
        """Handle mode change with visual feedback."""
        mode = self.mode_var.get()
        if mode == "single":
            self.max_pages_entry.config(state=tk.DISABLED)
            self.max_depth_entry.config(state=tk.DISABLED)
        else:
            self.max_pages_entry.config(state=tk.NORMAL)
            self.max_depth_entry.config(state=tk.NORMAL)

    def log_message(self, message, level='INFO'):
        """Add message to log display with color coding."""
        # Determine tag based on level or content
        tag = 'INFO'
        if 'ERROR' in level or 'error' in message.lower():
            tag = 'ERROR'
        elif 'WARNING' in level or 'warning' in message.lower():
            tag = 'WARNING'
        elif '✅' in message or 'success' in message.lower() or 'complete' in message.lower():
            tag = 'SUCCESS'
        elif '=' * 10 in message or message.startswith('Crawl'):
            tag = 'HEADER'

        self.log_text.insert(tk.END, message + '\n', tag)
        self.log_text.see(tk.END)

    def clear_log(self):
        """Clear the log display."""
        self.log_text.delete(1.0, tk.END)
        self.log_message("Log cleared.", 'INFO')

    def update_status(self, message, status_type='info'):
        """Update status message with colored indicator."""
        self.status_var.set(message)

        # Update status icon color
        color_map = {
            'info': COLORS['accent'],
            'success': COLORS['success'],
            'warning': COLORS['warning'],
            'error': COLORS['error'],
            'processing': COLORS['primary']
        }
        self.status_icon.config(foreground=color_map.get(status_type, COLORS['accent']))
        self.status_label.config(foreground=color_map.get(status_type, COLORS['accent']))

    def update_stats(self, total, successful, failed, warnings=0):
        """Update statistics display with modern styling."""
        self.stats_labels['total'].config(text=str(total))
        self.stats_labels['success'].config(text=str(successful))
        self.stats_labels['failed'].config(text=str(failed))
        self.stats_labels['warnings'].config(text=str(warnings))

    def update_progress(self, current, total):
        """Update progress bar with percentage."""
        if total > 0:
            percentage = (current / total) * 100
            self.progress_var.set(percentage)
            self.progress_label.config(text=f"{percentage:.0f}%")

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
        robots_handler = RobotsHandler(user_agent='WebScraperGUI/2.0')
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
                self.log_message(f"⚠️ User chose to cap crawl delay at 30 seconds", 'WARNING')
                self.log_message(f"   (Site requested {robots_delay}s = {time_str})", 'WARNING')
            else:  # User clicked NO - use full delay
                delay = robots_delay
                self.delay_var.set(str(delay))  # Update GUI to reflect full delay
                self.log_message(f"⚠️ User chose to respect site's full delay: {time_str}", 'WARNING')
                self.log_message(f"   Crawl will be very slow. Use Stop button to cancel.", 'WARNING')


        # Update UI state
        self.is_crawling = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.url_entry.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.progress_label.config(text="0%")

        # Clear previous log
        self.log_text.delete(1.0, tk.END)

        # Update status
        self.update_status(
            f"Starting {'full site' if mode == 'full' else 'single URL'} crawl...",
            'processing'
        )

        # Log configuration
        self.log_message("=" * 60, 'HEADER')
        self.log_message("Crawl Configuration:", 'HEADER')
        self.log_message(f"  Mode: {'Full Site Crawl' if mode == 'full' else 'Single URL'}")
        self.log_message(f"  Target: {url}")
        self.log_message(f"  Minimum delay: {delay} seconds")
        if mode == 'full':
            self.log_message(f"  Max pages: {max_pages}")
            self.log_message(f"  Max depth: {max_depth}")
        self.log_message("  Note: robots.txt delays will be respected if higher")
        self.log_message("=" * 60, 'HEADER')
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
            # Create scraper
            self.scraper = WebScraper(
                user_agent='WebScraperGUI/2.0',
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
                result = self.scraper.crawl_url(url, should_stop=lambda: not self.is_crawling)
                summary = {
                    'total': 1,
                    'successful': 1 if result['success'] else 0,
                    'failed': 0 if result['success'] else 1,
                    'warnings': 1 if not result['success'] and result.get('warning', False) else 0,
                    'results': [result]
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
        self.progress_label.config(text="100%")

        # Update stats
        warnings_count = summary.get('warnings', 0)
        self.update_stats(summary['total'], summary['successful'], summary['failed'], warnings_count)

        # Update status
        if summary['successful'] > 0:
            self.update_status(
                f"Crawl complete! {summary['successful']}/{summary['total']} pages successful",
                'success'
            )
            self.log_message(f"\n{'='*60}", 'HEADER')
            self.log_message(f"✅ CRAWL COMPLETE!", 'SUCCESS')
            self.log_message(f"{'='*60}", 'HEADER')
            self.log_message(f"Total pages: {summary['total']}", 'SUCCESS')
            self.log_message(f"Successful: {summary['successful']}", 'SUCCESS')
            self.log_message(f"Failed: {summary['failed']}", 'WARNING' if summary['failed'] > 0 else 'INFO')
            self.log_message(f"Warnings: {warnings_count}", 'WARNING' if warnings_count > 0 else 'INFO')

            # Show output files
            if self.scraper:
                stats = self.scraper.get_stats()
                if stats['output_files']:
                    self.log_message(f"\nOutput files:", 'HEADER')
                    for file in stats['output_files']:
                        self.log_message(f"  📄 {file}", 'SUCCESS')

            messagebox.showinfo("Success",
                                f"Crawl complete!\n\n"
                                f"Pages crawled: {summary['successful']}/{summary['total']}\n"
                                f"Check data/ folder for CSV files.")
        else:
            self.update_status(f"Crawl failed - no pages retrieved", 'error')
            messagebox.showwarning("Warning", "No pages were successfully crawled.")

    def _crawl_error(self, error_msg):
        """Handle crawl error."""
        self.is_crawling = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.url_entry.config(state=tk.NORMAL)

        self.update_status(f"Error: {error_msg}", 'error')
        self.log_message(f"\n❌ ERROR: {error_msg}", 'ERROR')
        messagebox.showerror("Crawl Error", f"An error occurred:\n\n{error_msg}")

    def stop_crawl(self):
        """Stop the crawling process."""
        if self.is_crawling:
            self.update_status("Stopping crawl...", 'warning')
            self.is_crawling = False
            # Crawl loop will check this flag and exit gracefully after current page

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


class ModernGUILogHandler(logging.Handler):
    """Custom log handler for modern GUI with color support."""

    def __init__(self, gui):
        super().__init__()
        self.gui = gui

    def emit(self, record):
        """Emit a log record to the GUI with appropriate styling."""
        try:
            msg = self.format(record)
            level = record.levelname

            # Schedule GUI update in main thread
            self.gui.root.after(0, self.gui.log_message, msg, level)

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
    """Launch the modern GUI application."""
    root = tk.Tk()

    # Set icon if available (optional)
    try:
        # You can add an icon here if you have one
        # root.iconbitmap('icon.ico')
        pass
    except:
        pass

    app = ModernWebScraperGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()