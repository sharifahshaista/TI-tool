"""
Theme System for Web Scraper GUI
Provides dark/light mode switching with persistent preferences
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Literal, Optional
from dataclasses import dataclass, asdict


@dataclass
class ColorScheme:
    """Color scheme definition for a theme"""
    bg: str  # Background
    fg: str  # Foreground/text
    surface: str  # Surface elements (cards, panels)
    primary: str  # Primary accent
    accent: str  # Secondary accent
    border: str  # Border color

    # Semantic colors (status indicators)
    success: str
    warning: str
    error: str
    info: str

    # Additional UI elements
    button_bg: str
    button_fg: str
    button_hover: str
    input_bg: str
    input_fg: str
    scrollbar: str
    scrollbar_hover: str


class ThemeManager:
    """Manages theme switching and persistence for the application"""

    # Dark theme color scheme
    DARK_THEME = ColorScheme(
        bg='#1e1e2e',
        fg='#e0e0e0',
        surface='#2d2d44',
        primary='#6c63ff',
        accent='#00d4ff',
        border='#404058',
        success='#50fa7b',
        warning='#ffb86c',
        error='#ff5555',
        info='#8be9fd',
        button_bg='#6c63ff',
        button_fg='#ffffff',
        button_hover='#5548d9',
        input_bg='#2d2d44',
        input_fg='#e0e0e0',
        scrollbar='#404058',
        scrollbar_hover='#6c63ff'
    )

    # Light theme color scheme
    LIGHT_THEME = ColorScheme(
        bg='#f5f5f5',
        fg='#1a1a1a',
        surface='#ffffff',
        primary='#5548d9',
        accent='#0099cc',
        border='#e0e0e0',
        success='#28a745',
        warning='#ffc107',
        error='#dc3545',
        info='#17a2b8',
        button_bg='#5548d9',
        button_fg='#ffffff',
        button_hover='#4237c8',
        input_bg='#ffffff',
        input_fg='#1a1a1a',
        scrollbar='#d0d0d0',
        scrollbar_hover='#5548d9'
    )

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize theme manager

        Args:
            config_dir: Directory to store theme preferences (default: ~/.webscraper)
        """
        self.config_dir = config_dir or Path.home() / '.webscraper'
        self.config_file = self.config_dir / 'theme_config.json'
        self.current_theme: Literal['dark', 'light'] = 'dark'
        self.theme_callbacks: list = []

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Load saved theme preference
        self._load_theme_preference()

    def _load_theme_preference(self) -> None:
        """Load theme preference from config file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.current_theme = config.get('theme', 'dark')
        except Exception as e:
            print(f"Error loading theme preference: {e}")
            self.current_theme = 'dark'

    def _save_theme_preference(self) -> None:
        """Save current theme preference to config file"""
        try:
            config = {'theme': self.current_theme}
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving theme preference: {e}")

    def get_current_theme(self) -> ColorScheme:
        """Get current theme color scheme"""
        return self.DARK_THEME if self.current_theme == 'dark' else self.LIGHT_THEME

    def get_theme_name(self) -> str:
        """Get current theme name"""
        return self.current_theme

    def is_dark_mode(self) -> bool:
        """Check if dark mode is active"""
        return self.current_theme == 'dark'

    def apply_dark_theme(self) -> None:
        """Apply dark theme"""
        if self.current_theme != 'dark':
            self.current_theme = 'dark'
            self._save_theme_preference()
            self._notify_theme_change()

    def apply_light_theme(self) -> None:
        """Apply light theme"""
        if self.current_theme != 'light':
            self.current_theme = 'light'
            self._save_theme_preference()
            self._notify_theme_change()

    def toggle_theme(self) -> str:
        """
        Toggle between dark and light themes

        Returns:
            New theme name
        """
        if self.current_theme == 'dark':
            self.apply_light_theme()
        else:
            self.apply_dark_theme()
        return self.current_theme

    def register_callback(self, callback) -> None:
        """
        Register a callback to be called when theme changes

        Args:
            callback: Function to call on theme change (receives ColorScheme as argument)
        """
        if callback not in self.theme_callbacks:
            self.theme_callbacks.append(callback)

    def unregister_callback(self, callback) -> None:
        """Remove a theme change callback"""
        if callback in self.theme_callbacks:
            self.theme_callbacks.remove(callback)

    def _notify_theme_change(self) -> None:
        """Notify all registered callbacks of theme change"""
        current_colors = self.get_current_theme()
        for callback in self.theme_callbacks:
            try:
                callback(current_colors)
            except Exception as e:
                print(f"Error in theme callback: {e}")

    def get_widget_styles(self) -> Dict[str, Any]:
        """
        Get ttk style configuration for current theme

        Returns:
            Dictionary of style configurations for ttk widgets
        """
        colors = self.get_current_theme()

        return {
            'TFrame': {
                'background': colors.bg
            },
            'TLabel': {
                'background': colors.bg,
                'foreground': colors.fg
            },
            'TButton': {
                'background': colors.button_bg,
                'foreground': colors.button_fg,
                'bordercolor': colors.border,
                'focuscolor': colors.primary,
                'lightcolor': colors.button_hover,
                'darkcolor': colors.button_bg
            },
            'TEntry': {
                'fieldbackground': colors.input_bg,
                'foreground': colors.input_fg,
                'bordercolor': colors.border,
                'insertcolor': colors.fg
            },
            'TCombobox': {
                'fieldbackground': colors.input_bg,
                'foreground': colors.input_fg,
                'background': colors.input_bg,
                'bordercolor': colors.border,
                'arrowcolor': colors.fg
            },
            'Treeview': {
                'background': colors.surface,
                'foreground': colors.fg,
                'fieldbackground': colors.surface,
                'bordercolor': colors.border
            },
            'Treeview.Heading': {
                'background': colors.primary,
                'foreground': colors.button_fg,
                'bordercolor': colors.border
            },
            'TScrollbar': {
                'background': colors.scrollbar,
                'troughcolor': colors.bg,
                'bordercolor': colors.border,
                'arrowcolor': colors.fg
            },
            'TNotebook': {
                'background': colors.bg,
                'bordercolor': colors.border
            },
            'TNotebook.Tab': {
                'background': colors.surface,
                'foreground': colors.fg,
                'bordercolor': colors.border
            },
            'Horizontal.TProgressbar': {
                'background': colors.primary,
                'troughcolor': colors.surface,
                'bordercolor': colors.border
            }
        }

    def get_text_widget_config(self) -> Dict[str, str]:
        """
        Get configuration for Text widgets (non-ttk)

        Returns:
            Dictionary of configuration options
        """
        colors = self.get_current_theme()

        return {
            'bg': colors.surface,
            'fg': colors.fg,
            'insertbackground': colors.fg,
            'selectbackground': colors.primary,
            'selectforeground': colors.button_fg,
            'highlightbackground': colors.border,
            'highlightcolor': colors.primary,
            'highlightthickness': 1
        }

    def get_canvas_config(self) -> Dict[str, str]:
        """
        Get configuration for Canvas widgets

        Returns:
            Dictionary of configuration options
        """
        colors = self.get_current_theme()

        return {
            'bg': colors.bg,
            'highlightbackground': colors.border,
            'highlightcolor': colors.primary,
            'highlightthickness': 1
        }

    def get_listbox_config(self) -> Dict[str, str]:
        """
        Get configuration for Listbox widgets

        Returns:
            Dictionary of configuration options
        """
        colors = self.get_current_theme()

        return {
            'bg': colors.surface,
            'fg': colors.fg,
            'selectbackground': colors.primary,
            'selectforeground': colors.button_fg,
            'highlightbackground': colors.border,
            'highlightcolor': colors.primary,
            'highlightthickness': 1
        }

    def get_status_color(self, status: Literal['success', 'warning', 'error', 'info']) -> str:
        """
        Get semantic color for status indicator

        Args:
            status: Status type

        Returns:
            Hex color code
        """
        colors = self.get_current_theme()
        return getattr(colors, status)

    def export_theme_spec(self) -> Dict[str, Any]:
        """
        Export complete theme specification

        Returns:
            Dictionary containing both theme definitions
        """
        return {
            'dark': asdict(self.DARK_THEME),
            'light': asdict(self.LIGHT_THEME),
            'current': self.current_theme
        }


# Singleton instance for global access
_theme_manager_instance: Optional[ThemeManager] = None


def get_theme_manager() -> ThemeManager:
    """Get or create the global theme manager instance"""
    global _theme_manager_instance
    if _theme_manager_instance is None:
        _theme_manager_instance = ThemeManager()
    return _theme_manager_instance


def apply_theme_to_widget(widget, theme_manager: Optional[ThemeManager] = None) -> None:
    """
    Apply current theme to a widget and all its children

    Args:
        widget: Root widget to apply theme to
        theme_manager: ThemeManager instance (uses global if not provided)
    """
    if theme_manager is None:
        theme_manager = get_theme_manager()

    colors = theme_manager.get_current_theme()

    def apply_recursive(w):
        try:
            widget_class = w.winfo_class()

            # Apply theme based on widget type
            if widget_class in ('Frame', 'Toplevel', 'TFrame'):
                w.configure(background=colors.bg)
            elif widget_class in ('Label', 'TLabel'):
                w.configure(background=colors.bg, foreground=colors.fg)
            elif widget_class in ('Button', 'TButton'):
                w.configure(background=colors.button_bg, foreground=colors.button_fg)
            elif widget_class in ('Entry', 'TEntry'):
                w.configure(background=colors.input_bg, foreground=colors.input_fg)
            elif widget_class == 'Text':
                config = theme_manager.get_text_widget_config()
                w.configure(**config)
            elif widget_class == 'Canvas':
                config = theme_manager.get_canvas_config()
                w.configure(**config)
            elif widget_class == 'Listbox':
                config = theme_manager.get_listbox_config()
                w.configure(**config)

            # Recursively apply to children
            for child in w.winfo_children():
                apply_recursive(child)

        except Exception as e:
            # Skip widgets that don't support theming
            pass

    apply_recursive(widget)


if __name__ == '__main__':
    # Example usage
    manager = ThemeManager()

    print(f"Current theme: {manager.get_theme_name()}")
    print(f"Is dark mode: {manager.is_dark_mode()}")

    # Toggle theme
    print(f"\nToggling to: {manager.toggle_theme()}")

    # Get widget styles
    styles = manager.get_widget_styles()
    print(f"\nWidget styles: {list(styles.keys())}")

    # Export theme spec
    spec = manager.export_theme_spec()
    print(f"\nTheme spec exported with keys: {list(spec.keys())}")