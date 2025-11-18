"""
Terminal UI Module using Rich Library

Provides an enhanced terminal interface with colored output,
progress bars, and interactive elements.
"""

import sys
import logging
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich import box
from .scraper import WebScraper

console = Console()


def setup_logging():
    """Set up logging with rich handler."""
    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


def display_banner():
    """Display welcome banner."""
    banner = """
    ╔══════════════════════════════════════════╗
    ║      🕷️  Python Web Scraper UI 🕷️       ║
    ║                                          ║
    ║  Respects robots.txt & crawl delays     ║
    ╚══════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")


def prompt_for_urls():
    """
    Prompt user for URLs to crawl with enhanced UI.

    Returns:
        list: List of validated URLs
    """
    console.print("\n[bold cyan]Enter URLs to scrape[/bold cyan]")
    console.print("Type URLs one per line. Press Enter on empty line when done.\n")

    urls = []
    url_num = 1

    while True:
        try:
            url = Prompt.ask(
                f"[yellow]URL {url_num}[/yellow]",
                default=""
            ).strip()

            if not url:
                if url_num == 1:
                    console.print("[red]At least one URL is required![/red]")
                    continue
                break

            # Basic validation
            if not url.startswith(('http://', 'https://')):
                console.print("  [yellow]⚠[/yellow] URL should start with http:// or https://")
                if not Confirm.ask("  Add anyway?", default=False):
                    continue

            urls.append(url)
            console.print(f"  [green]✓[/green] Added: {url}")
            url_num += 1

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted by user[/yellow]")
            if urls and Confirm.ask("\nUse entered URLs?", default=True):
                break
            sys.exit(0)

    return urls


def validate_urls(urls, scraper):
    """
    Validate URLs and check robots.txt permissions.

    Args:
        urls (list): List of URLs to validate
        scraper: WebScraper instance

    Returns:
        list: Validated URLs that are allowed to crawl
    """
    console.print("\n[bold cyan]Validating URLs...[/bold cyan]\n")

    table = Table(box=box.ROUNDED)
    table.add_column("URL", style="cyan", no_wrap=False)
    table.add_column("Status", justify="center")
    table.add_column("Crawl Delay", justify="right")

    validated = []

    for url in urls:
        can_fetch = scraper.robots_handler.can_fetch(url)
        delay = scraper.robots_handler.get_crawl_delay(url)

        if can_fetch:
            status = Text("✓ Allowed", style="green")
            validated.append(url)
        else:
            status = Text("✗ Blocked", style="red")

        delay_text = f"{delay}s" if delay else "1.0s (default)"
        table.add_row(url, status, delay_text)

    console.print(table)

    if len(validated) < len(urls):
        blocked_count = len(urls) - len(validated)
        console.print(f"\n[yellow]⚠ {blocked_count} URL(s) blocked by robots.txt[/yellow]")

    return validated


def crawl_with_progress(urls, scraper):
    """
    Crawl URLs with progress bar display.

    Args:
        urls (list): List of URLs to crawl
        scraper: WebScraper instance

    Returns:
        dict: Crawl summary
    """
    console.print(f"\n[bold cyan]Starting crawl of {len(urls)} URL(s)...[/bold cyan]\n")

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:

        task = progress.add_task("[cyan]Crawling...", total=len(urls))

        for idx, url in enumerate(urls, 1):
            progress.update(task, description=f"[cyan]Crawling {idx}/{len(urls)}: {url[:50]}...")

            result = scraper.crawl_url(url)
            results.append(result)

            progress.advance(task)

    # Calculate summary
    successful = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])

    summary = {
        'total': len(urls),
        'successful': successful,
        'failed': failed,
        'results': results
    }

    return summary


def display_results(summary):
    """
    Display crawl results in a formatted table.

    Args:
        summary (dict): Crawl summary
    """
    console.print("\n")

    # Summary panel
    summary_text = f"""
    [bold]Total URLs:[/bold] {summary['total']}
    [bold green]Successful:[/bold green] {summary['successful']}
    [bold red]Failed:[/bold red] {summary['failed']}
    """

    console.print(Panel(summary_text, title="[bold cyan]Crawl Results[/bold cyan]", border_style="cyan"))

    # Detailed results table
    if summary['results']:
        console.print("\n[bold cyan]Detailed Results:[/bold cyan]\n")

        table = Table(box=box.ROUNDED, show_lines=True)
        table.add_column("#", justify="right", style="dim")
        table.add_column("URL", style="cyan", no_wrap=False)
        table.add_column("Status", justify="center")
        table.add_column("Notes")

        for idx, result in enumerate(summary['results'], 1):
            url = result['url']

            if result['success']:
                status = Text("✓ Success", style="green bold")
                notes = "Content extracted and saved"
            else:
                status = Text("✗ Failed", style="red bold")
                notes = result.get('error', 'Unknown error')

            table.add_row(str(idx), url, status, notes)

        console.print(table)


def display_stats(scraper):
    """
    Display scraper statistics.

    Args:
        scraper: WebScraper instance
    """
    stats = scraper.get_stats()

    console.print("\n[bold cyan]Scraper Statistics:[/bold cyan]\n")

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total URLs crawled", str(stats['total_crawled']))
    table.add_row("Successful", f"[green]{stats['successful']}[/green]")
    table.add_row("Failed", f"[red]{stats['failed']}[/red]")

    if stats['output_files']:
        table.add_row("Output files", str(len(stats['output_files'])))

    console.print(table)

    if stats['output_files']:
        console.print("\n[bold cyan]Output Files:[/bold cyan]")
        for filepath in stats['output_files']:
            console.print(f"  [green]→[/green] {filepath}")

    console.print()


def main():
    """Main terminal UI entry point."""
    try:
        # Setup
        setup_logging()
        display_banner()

        # Initialize scraper
        scraper = WebScraper(user_agent='PythonWebScraper/1.0', default_delay=1.0)

        try:
            # Get URLs from user
            urls = prompt_for_urls()

            if not urls:
                console.print("\n[yellow]No URLs provided. Exiting.[/yellow]")
                return

            # Validate URLs
            validated_urls = validate_urls(urls, scraper)

            if not validated_urls:
                console.print("\n[red]No valid URLs to crawl. Exiting.[/red]")
                return

            if len(validated_urls) < len(urls):
                if not Confirm.ask("\nContinue with allowed URLs?", default=True):
                    console.print("[yellow]Cancelled by user[/yellow]")
                    return

            # Crawl URLs
            summary = crawl_with_progress(validated_urls, scraper)

            # Display results
            display_results(summary)
            display_stats(scraper)

            console.print("\n[bold green]✓ Scraping complete![/bold green]\n")

        finally:
            scraper.close()

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user. Exiting...[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        logging.error(f"Error in main: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()