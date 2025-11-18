#!/usr/bin/env python3
"""
Modern UI Components for Web Scraper GUI

Provides reusable, styled components following modern design principles:
- Card-based layouts with subtle shadows
- Enhanced buttons with states (primary, secondary, icon)
- Status badges with color coding
- Modern input fields with validation feedback
- Enhanced log display with syntax highlighting
- Tooltips for better UX
"""

import tkinter as tk
from tkinter import ttk, font
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, Any, List
import re


class ModernCard(ttk.Frame):
    """
    Modern card component with subtle elevation and rounded appearance.

    Features:
    - Subtle background differentiation
    - Simulated rounded corners (4-6px)
    - Consistent internal padding (12-16px)
    - Optional header section
    """

    def __init__(self, parent, title: Optional[str] = None, padding: int = 16, **kwargs):
        """
        Initialize a modern card component.

        Args:
            parent: Parent widget
            title: Optional card title
            padding: Internal padding in pixels (default: 16)
            **kwargs: Additional frame arguments
        """
        super().__init__(parent, **kwargs)

        # Configure card styling
        self.configure(relief=tk.RIDGE, borderwidth=2, padding=padding)

        # Create header if title provided
        if title:
            self._create_header(title)

        # Content frame
        self.content_frame = ttk.Frame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True)

    def _create_header(self, title: str):
        """Create card header with title."""
        header_frame = ttk.Frame(self)
        header_frame.pack(fill=tk.X, pady=(0, 12))

        title_label = ttk.Label(
            header_frame,
            text=title,
            font=('Helvetica', 12, 'bold')
        )
        title_label.pack(side=tk.LEFT)

        # Separator line
        separator = ttk.Separator(self, orient=tk.HORIZONTAL)
        separator.pack(fill=tk.X, pady=(0, 8))


class ModernButton(ttk.Button):
    """
    Enhanced button with multiple styles and states.

    Styles:
    - primary: Filled button with bold color
    - secondary: Outlined button
    - icon: Icon-only button

    States:
    - Normal: Interactive
    - Disabled: Grayed out
    - Hover: Visual feedback (cursor change)
    """

    STYLES = {
        'primary': {'background': '#0066CC', 'foreground': 'white'},
        'secondary': {'background': 'white', 'foreground': '#0066CC'},
        'icon': {'background': 'transparent'}
    }

    def __init__(self, parent, text: str = "", icon: str = "",
                 style: str = 'primary', tooltip: Optional[str] = None,
                 command: Optional[Callable] = None, **kwargs):
        """
        Initialize an enhanced button.

        Args:
            parent: Parent widget
            text: Button text
            icon: Unicode/emoji icon
            style: Button style ('primary', 'secondary', 'icon')
            tooltip: Tooltip text
            command: Button command callback
            **kwargs: Additional button arguments
        """
        button_text = f"{icon} {text}".strip() if icon else text

        super().__init__(parent, text=button_text, command=command, **kwargs)

        # Apply style
        self._apply_style(style)

        # Add tooltip if provided
        if tooltip:
            ToolTip(self, tooltip)

        # Cursor feedback
        self.bind('<Enter>', lambda e: self.config(cursor='hand2'))
        self.bind('<Leave>', lambda e: self.config(cursor=''))

    def _apply_style(self, style: str):
        """Apply button style."""
        if style in self.STYLES:
            # Note: ttk buttons have limited styling,
            # but we can use cursor and relief for feedback
            if style == 'primary':
                self.config(style='Accent.TButton')
            elif style == 'icon':
                self.config(width=3)


class StatusBadge(tk.Label):
    """
    Colored badge for displaying status information.

    Status types:
    - running: Blue badge
    - success: Green badge
    - error: Red badge
    - warning: Orange badge
    - info: Gray badge
    """

    STATUS_COLORS = {
        'running': {'bg': '#0066CC', 'fg': 'white'},
        'success': {'bg': '#28A745', 'fg': 'white'},
        'error': {'bg': '#DC3545', 'fg': 'white'},
        'warning': {'bg': '#FFC107', 'fg': 'black'},
        'info': {'bg': '#6C757D', 'fg': 'white'},
        'idle': {'bg': '#E9ECEF', 'fg': 'black'}
    }

    STATUS_ICONS = {
        'running': '▶',
        'success': '✓',
        'error': '✗',
        'warning': '⚠',
        'info': 'ℹ',
        'idle': '○'
    }

    def __init__(self, parent, status: str = 'idle', text: str = "", **kwargs):
        """
        Initialize a status badge.

        Args:
            parent: Parent widget
            status: Status type ('running', 'success', 'error', etc.)
            text: Badge text (auto-capitalizes status if empty)
            **kwargs: Additional label arguments
        """
        super().__init__(parent, **kwargs)

        self.status = status
        self.set_status(status, text)

        # Style badge
        self.config(
            font=('Helvetica', 9, 'bold'),
            relief=tk.RAISED,
            borderwidth=1,
            padx=8,
            pady=4
        )

    def set_status(self, status: str, text: str = ""):
        """
        Update badge status.

        Args:
            status: New status type
            text: New badge text
        """
        self.status = status

        if status in self.STATUS_COLORS:
            colors = self.STATUS_COLORS[status]
            self.config(bg=colors['bg'], fg=colors['fg'])

        icon = self.STATUS_ICONS.get(status, '')
        display_text = text or status.capitalize()
        self.config(text=f"{icon} {display_text}")


class CounterBadge(tk.Label):
    """
    Numeric counter badge with color coding.

    Used for displaying statistics like page counts, success/failure counts.
    """

    def __init__(self, parent, label: str, count: int = 0,
                 color: str = '#0066CC', **kwargs):
        """
        Initialize a counter badge.

        Args:
            parent: Parent widget
            label: Counter label text
            count: Initial count value
            color: Badge color
            **kwargs: Additional label arguments
        """
        super().__init__(parent, **kwargs)

        self.label_text = label
        self.count = count
        self.color = color

        self._update_display()

        # Style badge
        self.config(
            font=('Helvetica', 10, 'bold'),
            bg=color,
            fg='white',
            relief=tk.RAISED,
            borderwidth=1,
            padx=12,
            pady=6
        )

    def set_count(self, count: int):
        """Update counter value."""
        self.count = count
        self._update_display()

    def increment(self, amount: int = 1):
        """Increment counter."""
        self.count += amount
        self._update_display()

    def reset(self):
        """Reset counter to zero."""
        self.count = 0
        self._update_display()

    def _update_display(self):
        """Update display text."""
        self.config(text=f"{self.label_text}: {self.count}")


class ModernEntry(ttk.Frame):
    """
    Enhanced entry field with focus indicators and validation.

    Features:
    - Placeholder text
    - Visual validation feedback
    - Clear button
    - Focus indicators
    """

    def __init__(self, parent, placeholder: str = "",
                 validator: Optional[Callable[[str], bool]] = None,
                 show_clear: bool = True, **kwargs):
        """
        Initialize a modern entry field.

        Args:
            parent: Parent widget
            placeholder: Placeholder text
            validator: Validation function (returns bool)
            show_clear: Show clear button
            **kwargs: Additional entry arguments
        """
        super().__init__(parent)

        self.placeholder = placeholder
        self.validator = validator
        self.has_focus = False
        self.is_valid = True

        # Entry widget
        self.entry = ttk.Entry(self, **kwargs)
        self.entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Clear button
        if show_clear:
            self.clear_btn = ModernButton(
                self,
                icon="✕",
                style='icon',
                command=self.clear,
                tooltip="Clear"
            )
            self.clear_btn.pack(side=tk.RIGHT, padx=(2, 0))

        # Setup behaviors
        self._setup_placeholder()
        self._setup_validation()
        self._setup_focus()

    def _setup_placeholder(self):
        """Setup placeholder behavior."""
        if self.placeholder:
            self.entry.insert(0, self.placeholder)
            self.entry.config(foreground='gray')

            def on_focus_in(event):
                if self.entry.get() == self.placeholder:
                    self.entry.delete(0, tk.END)
                    self.entry.config(foreground='black')

            def on_focus_out(event):
                if not self.entry.get():
                    self.entry.insert(0, self.placeholder)
                    self.entry.config(foreground='gray')

            self.entry.bind('<FocusIn>', on_focus_in)
            self.entry.bind('<FocusOut>', on_focus_out)

    def _setup_validation(self):
        """Setup validation feedback."""
        if self.validator:
            def validate(event=None):
                text = self.get()
                if text and text != self.placeholder:
                    self.is_valid = self.validator(text)
                    # Visual feedback via border color would need custom styling
                    if self.is_valid:
                        self.entry.config(foreground='black')
                    else:
                        self.entry.config(foreground='red')

            self.entry.bind('<KeyRelease>', validate)

    def _setup_focus(self):
        """Setup focus indicators."""
        def on_focus_in(event):
            self.has_focus = True
            self.config(relief=tk.SOLID, borderwidth=2)

        def on_focus_out(event):
            self.has_focus = False
            self.config(relief=tk.FLAT, borderwidth=0)

        self.entry.bind('<FocusIn>', on_focus_in)
        self.entry.bind('<FocusOut>', on_focus_out)

    def get(self) -> str:
        """Get entry value (excluding placeholder)."""
        text = self.entry.get()
        return "" if text == self.placeholder else text

    def set(self, value: str):
        """Set entry value."""
        self.entry.delete(0, tk.END)
        self.entry.insert(0, value)
        self.entry.config(foreground='black')

    def clear(self):
        """Clear entry value."""
        self.entry.delete(0, tk.END)
        if self.placeholder:
            self.entry.insert(0, self.placeholder)
            self.entry.config(foreground='gray')


class EnhancedLogDisplay(ttk.Frame):
    """
    Enhanced log display with syntax highlighting and features.

    Features:
    - Syntax highlighting for log levels
    - Alternating row colors
    - Auto-scroll with indicator
    - Timestamp formatting with relative times
    - Search/filter capabilities
    """

    LOG_COLORS = {
        'INFO': '#0066CC',
        'WARNING': '#FFC107',
        'ERROR': '#DC3545',
        'SUCCESS': '#28A745',
        'DEBUG': '#6C757D'
    }

    def __init__(self, parent, height: int = 15, **kwargs):
        """
        Initialize enhanced log display.

        Args:
            parent: Parent widget
            height: Display height in lines
            **kwargs: Additional text widget arguments
        """
        super().__init__(parent)

        # Create text widget with scrollbar
        self.text = tk.Text(
            self,
            height=height,
            wrap=tk.WORD,
            font=('Courier', 9),
            bg='#F8F9FA',
            relief=tk.FLAT,
            **kwargs
        )

        scrollbar = ttk.Scrollbar(self, command=self.text.yview)
        self.text.config(yscrollcommand=scrollbar.set)

        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure tags for syntax highlighting
        self._setup_tags()

        # Track auto-scroll
        self.auto_scroll = True
        self.last_scroll_pos = 0.0

        # Scroll indicator
        self._setup_scroll_indicator()

        # Line counter
        self.line_count = 0

    def _setup_tags(self):
        """Setup text tags for syntax highlighting."""
        # Log level tags
        for level, color in self.LOG_COLORS.items():
            self.text.tag_config(level, foreground=color, font=('Courier', 9, 'bold'))

        # Timestamp tag
        self.text.tag_config('timestamp', foreground='#6C757D')

        # Alternating row background
        self.text.tag_config('even', background='#FFFFFF')
        self.text.tag_config('odd', background='#F8F9FA')

        # URL highlighting
        self.text.tag_config('url', foreground='#0066CC', underline=True)

    def _setup_scroll_indicator(self):
        """Setup scroll-to-bottom indicator."""
        self.scroll_indicator = tk.Label(
            self,
            text="↓ New messages",
            bg='#FFC107',
            fg='black',
            font=('Helvetica', 9, 'bold'),
            cursor='hand2',
            padx=8,
            pady=4
        )

        # Show indicator when not at bottom
        def on_scroll(*args):
            self.text.yview(*args)
            pos = self.text.yview()[1]
            if pos < 0.99 and self.auto_scroll:
                self.scroll_indicator.place(relx=0.5, rely=1.0, anchor=tk.S, y=-10)
            else:
                self.scroll_indicator.place_forget()

        self.text.config(yscrollcommand=lambda *args: (
            self.text['yscrollcommand'](*args) if callable(self.text['yscrollcommand']) else None,
            on_scroll(*args)
        ))

        # Click indicator to scroll to bottom
        self.scroll_indicator.bind('<Button-1>', lambda e: self.scroll_to_bottom())

    def add_message(self, message: str, level: str = 'INFO', timestamp: Optional[datetime] = None):
        """
        Add a log message with syntax highlighting.

        Args:
            message: Log message text
            level: Log level (INFO, WARNING, ERROR, etc.)
            timestamp: Message timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Format timestamp
        time_str = timestamp.strftime('%H:%M:%S')

        # Check if we should alternate row color
        row_tag = 'even' if self.line_count % 2 == 0 else 'odd'

        # Insert timestamp
        self.text.insert(tk.END, f"[{time_str}] ", ('timestamp', row_tag))

        # Insert level
        self.text.insert(tk.END, f"{level}: ", (level, row_tag))

        # Highlight URLs in message
        parts = re.split(r'(https?://[^\s]+)', message)
        for part in parts:
            if part.startswith(('http://', 'https://')):
                self.text.insert(tk.END, part, ('url', row_tag))
            else:
                self.text.insert(tk.END, part, (row_tag,))

        self.text.insert(tk.END, '\n')

        # Auto-scroll if enabled
        if self.auto_scroll:
            self.scroll_to_bottom()

        self.line_count += 1

    def add_raw_message(self, message: str):
        """Add raw message without formatting."""
        self.text.insert(tk.END, message + '\n')
        if self.auto_scroll:
            self.scroll_to_bottom()
        self.line_count += 1

    def scroll_to_bottom(self):
        """Scroll to bottom of log."""
        self.text.see(tk.END)
        self.auto_scroll = True
        self.scroll_indicator.place_forget()

    def clear(self):
        """Clear all log messages."""
        self.text.delete(1.0, tk.END)
        self.line_count = 0

    def get_text(self) -> str:
        """Get all log text."""
        return self.text.get(1.0, tk.END)


class ToolTip:
    """
    Tooltip widget that appears on hover.

    Provides informative text when hovering over widgets.
    """

    def __init__(self, widget, text: str, delay: int = 500):
        """
        Initialize tooltip.

        Args:
            widget: Widget to attach tooltip to
            text: Tooltip text
            delay: Delay before showing tooltip (ms)
        """
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip_window = None
        self.timer_id = None

        # Bind events
        widget.bind('<Enter>', self._on_enter)
        widget.bind('<Leave>', self._on_leave)
        widget.bind('<Button>', self._on_leave)

    def _on_enter(self, event):
        """Handle mouse enter."""
        self._schedule_show()

    def _on_leave(self, event):
        """Handle mouse leave."""
        self._cancel_show()
        self._hide()

    def _schedule_show(self):
        """Schedule tooltip display."""
        self._cancel_show()
        self.timer_id = self.widget.after(self.delay, self._show)

    def _cancel_show(self):
        """Cancel scheduled tooltip."""
        if self.timer_id:
            self.widget.after_cancel(self.timer_id)
            self.timer_id = None

    def _show(self):
        """Display tooltip."""
        if self.tooltip_window:
            return

        # Get widget position
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        # Create tooltip window
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        # Create label
        label = tk.Label(
            self.tooltip_window,
            text=self.text,
            background='#FFFFE0',
            foreground='black',
            relief=tk.SOLID,
            borderwidth=1,
            font=('Helvetica', 9),
            padx=8,
            pady=4
        )
        label.pack()

    def _hide(self):
        """Hide tooltip."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class ProgressIndicator(ttk.Frame):
    """
    Enhanced progress indicator with percentage and status.

    Extends basic progress bar with:
    - Percentage display
    - Status text
    - Color coding based on state
    - Animated spinner for indeterminate progress
    """

    def __init__(self, parent, mode: str = 'determinate', **kwargs):
        """
        Initialize progress indicator.

        Args:
            parent: Parent widget
            mode: Progress mode ('determinate' or 'indeterminate')
            **kwargs: Additional progressbar arguments
        """
        super().__init__(parent)

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progressbar = ttk.Progressbar(
            self,
            variable=self.progress_var,
            maximum=100,
            mode=mode,
            **kwargs
        )
        self.progressbar.pack(fill=tk.X, expand=True)

        # Info frame (percentage + status)
        info_frame = ttk.Frame(self)
        info_frame.pack(fill=tk.X, pady=(4, 0))

        # Percentage label
        self.percentage_var = tk.StringVar(value="0%")
        self.percentage_label = ttk.Label(
            info_frame,
            textvariable=self.percentage_var,
            font=('Helvetica', 9, 'bold')
        )
        self.percentage_label.pack(side=tk.LEFT)

        # Status label
        self.status_var = tk.StringVar(value="")
        self.status_label = ttk.Label(
            info_frame,
            textvariable=self.status_var,
            font=('Helvetica', 9),
            foreground='gray'
        )
        self.status_label.pack(side=tk.RIGHT)

    def set_progress(self, value: float, status: str = ""):
        """
        Update progress value.

        Args:
            value: Progress value (0-100)
            status: Status text
        """
        self.progress_var.set(value)
        self.percentage_var.set(f"{int(value)}%")

        if status:
            self.status_var.set(status)

    def set_status(self, status: str):
        """Update status text."""
        self.status_var.set(status)

    def reset(self):
        """Reset progress to zero."""
        self.progress_var.set(0)
        self.percentage_var.set("0%")
        self.status_var.set("")


# Utility functions for component styling

def apply_modern_theme(root: tk.Tk):
    """
    Apply modern theme styling to the application.

    Args:
        root: Root window
    """
    style = ttk.Style(root)

    # Try to use modern theme
    try:
        style.theme_use('clam')
    except:
        pass

    # Configure styles
    style.configure('TFrame', background='white')
    style.configure('TLabel', background='white')
    style.configure('TButton', padding=8)

    # Accent button style
    style.configure(
        'Accent.TButton',
        font=('Helvetica', 10, 'bold')
    )

    # Card style
    style.configure(
        'Card.TFrame',
        relief=tk.RIDGE,
        borderwidth=2
    )


def create_stats_panel(parent, stats: Dict[str, int]) -> ttk.Frame:
    """
    Create a statistics panel with counter badges.

    Args:
        parent: Parent widget
        stats: Dictionary of stat_name: count pairs

    Returns:
        Frame containing stats panel
    """
    panel = ttk.Frame(parent)

    colors = {
        'total': '#0066CC',
        'successful': '#28A745',
        'failed': '#DC3545',
        'warnings': '#FFC107',
        'pages': '#6C757D'
    }

    for i, (label, count) in enumerate(stats.items()):
        color = colors.get(label.lower(), '#6C757D')
        badge = CounterBadge(panel, label.capitalize(), count, color)
        badge.pack(side=tk.LEFT, padx=5)

    return panel


# Example usage and testing
if __name__ == '__main__':
    # Create test window
    root = tk.Tk()
    root.title("Modern UI Components Demo")
    root.geometry("800x600")

    apply_modern_theme(root)

    # Main container
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Card example
    card = ModernCard(main_frame, title="Sample Card")
    card.pack(fill=tk.X, pady=10)

    # Buttons
    btn_frame = ttk.Frame(card.content_frame)
    btn_frame.pack(fill=tk.X, pady=10)

    ModernButton(btn_frame, "Primary Button", icon="▶",
                 tooltip="Start action").pack(side=tk.LEFT, padx=5)
    ModernButton(btn_frame, "Secondary", style='secondary',
                 tooltip="Cancel action").pack(side=tk.LEFT, padx=5)
    ModernButton(btn_frame, icon="🗑️", style='icon',
                 tooltip="Delete").pack(side=tk.LEFT, padx=5)

    # Status badges
    badge_frame = ttk.Frame(card.content_frame)
    badge_frame.pack(fill=tk.X, pady=10)

    StatusBadge(badge_frame, 'running').pack(side=tk.LEFT, padx=5)
    StatusBadge(badge_frame, 'success').pack(side=tk.LEFT, padx=5)
    StatusBadge(badge_frame, 'error').pack(side=tk.LEFT, padx=5)
    StatusBadge(badge_frame, 'warning').pack(side=tk.LEFT, padx=5)

    # Counter badges
    stats_panel = create_stats_panel(card.content_frame, {
        'Pages': 42,
        'Successful': 38,
        'Failed': 4,
        'Warnings': 2
    })
    stats_panel.pack(fill=tk.X, pady=10)

    # Modern entry
    entry = ModernEntry(
        card.content_frame,
        placeholder="Enter URL here...",
        validator=lambda x: x.startswith('http')
    )
    entry.pack(fill=tk.X, pady=10)

    # Enhanced log
    log_card = ModernCard(main_frame, title="Log Output")
    log_card.pack(fill=tk.BOTH, expand=True, pady=10)

    log = EnhancedLogDisplay(log_card.content_frame)
    log.pack(fill=tk.BOTH, expand=True)

    # Add sample log messages
    log.add_message("Starting crawl...", "INFO")
    log.add_message("Processing https://example.com", "INFO")
    log.add_message("Page crawled successfully", "SUCCESS")
    log.add_message("Rate limit detected", "WARNING")
    log.add_message("Failed to connect to https://broken.com", "ERROR")

    # Progress indicator
    progress = ProgressIndicator(main_frame)
    progress.pack(fill=tk.X, pady=10)
    progress.set_progress(65, "Processing page 13 of 20...")

    root.mainloop()