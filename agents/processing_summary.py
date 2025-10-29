"""
Processing Summary Generator
Displays clear completion summaries for both quick start and custom post-processing
"""

def display_completion_banner(
    processor_type: str,
    stats: dict,
    preset_name: str = None,
    input_folder: str = None,
    output_folder: str = None
):
    """Display a clear completion banner with processing statistics."""
    
    # Prepare the banner
    print("\n" + "="*80)
    print(f"{processor_type} POST-PROCESSING COMPLETE")
    print("="*80)
    
    if preset_name:
        print(f"Preset: {preset_name}")
    if input_folder:
        print(f"Input: {input_folder}")
    if output_folder:
        print(f"Output: {output_folder}")
    print("-"*80)
    
    # Display stats
    if 'total_files' in stats:
        success_rate = (stats['processed'] / stats['total_files'] * 100) if stats['total_files'] > 0 else 0
        print(f"Total files processed: {stats['total_files']}")
        print(f"Successfully processed: {stats['processed']} ({success_rate:.1f}%)")
        print(f"Skipped (old): {stats.get('skipped_old', 0)}")
        print(f"Skipped (error): {stats.get('skipped_error', 0)}")
        
        if stats.get('date_cutoff'):
            print(f"\nDate filter: Articles after {stats['date_cutoff'].strftime('%Y-%m-%d')}")
    
    # Display file types processed
    print("\nOutput formats:")
    print("  - CSV")
    print("  - JSON")
    print("  - Statistics summary")
    
    print("\nFiles saved:")
    if stats.get('output_files'):
        for file_type, path in stats['output_files'].items():
            print(f"  - {file_type}: {path}")
    
    # Display additional stats if available
    if stats.get('pattern_detection'):
        print("\nPattern Detection:")
        print(f"  Method: {'AI-powered' if stats['pattern_detection'].get('ai_powered', False) else 'Fallback patterns'}")
        if stats['pattern_detection'].get('fields_detected'):
            print("  Fields detected:")
            for field, success in stats['pattern_detection']['fields_detected'].items():
                print(f"    - {field}: {'✓' if success else '✗'}")
    
    print("="*80)
    print("✨ Processing complete! Data is ready for use.")
    print("="*80)


def display_error_banner(error_msg: str, processor_type: str = "MARKDOWN"):
    """Display a clear error banner."""
    print("\n" + "="*80)
    print(f"{processor_type} PROCESSING ERROR")
    print("="*80)
    print(f"Error: {error_msg}")
    print("-"*80)
    print("Please check:")
    print("  1. Input folder exists and contains markdown files")
    print("  2. You have write permissions for the output directory")
    print("  3. Files are valid markdown format")
    print("="*80)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    minutes = int(seconds / 60)
    remaining_seconds = seconds % 60
    return f"{minutes}m {remaining_seconds:.1f}s"