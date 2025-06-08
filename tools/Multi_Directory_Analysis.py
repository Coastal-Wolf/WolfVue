#!/usr/bin/env python3
"""
Wildlife Directory Analyzer
Analyzes multiple directories with a YOLO model to determine species concentrations
and ranks directories by highest counts of each species.

Based on WolfVue by Nathan Bluto
Data from The Gray Wolf Research Project
Facilitated by Dr. Ausband
"""

import os
import sys
import yaml
import cv2
import time
import platform
from pathlib import Path
from datetime import timedelta
from ultralytics import YOLO
import shutil
from collections import defaultdict

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ============= CONFIGURATION (EASILY ADJUSTABLE) =============
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

# Define paths relative to the script directory
CONFIG_FILE = SCRIPT_DIR / "WlfCamData.yaml"       # YAML file
DEFAULT_MODEL_PATH = SCRIPT_DIR / "weights" / "WolfVue_Beta1" / "best.pt"  # YOLO model

# Analysis parameters (adjust as needed)
CONFIDENCE_THRESHOLD = 0.40  # Minimum confidence for detections in videos
IMAGE_CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence for detections in images

# UI Settings
PROGRESS_BAR_WIDTH = 50  # Width of the console progress bar
UPDATE_FREQUENCY = 10  # Update the progress bar every N frames
MAX_PATH_DISPLAY_LENGTH = 60  # Max length to display for paths

# ============= END CONFIGURATION =============

# Check if Windows
IS_WINDOWS = platform.system() == 'Windows'

# ASCII Art for the analyzer
ANALYZER_ASCII_ART = r"""
██╗    ██╗██╗██╗     ██████╗ ██╗     ██╗███████╗███████╗
██║    ██║██║██║     ██╔══██╗██║     ██║██╔════╝██╔════╝
██║ █╗ ██║██║██║     ██║  ██║██║     ██║█████╗  █████╗  
██║███╗██║██║██║     ██║  ██║██║     ██║██╔══╝  ██╔══╝  
╚███╔███╔╝██║███████╗██████╔╝███████╗██║██║     ███████╗
 ╚══╝╚══╝ ╚═╝╚══════╝╚═════╝ ╚══════╝╚═╝╚═╝     ╚══════╝
                                                        
 █████╗ ███╗   ██╗ █████╗ ██╗  ██╗   ██╗███████╗███████╗██████╗ 
██╔══██╗████╗  ██║██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝██╔══██╗
███████║██╔██╗ ██║███████║██║   ╚████╔╝   ███╔╝ █████╗  ██████╔╝
██╔══██║██║╚██╗██║██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  ██╔══██╗
██║  ██║██║ ╚████║██║  ██║███████╗██║   ███████╗███████╗██║  ██║
╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝╚═╝  ╚═╝
"""

# Title with fancy border
TITLE_DISPLAY = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    WILDLIFE DIRECTORY ANALYZER                              ║
║                                                                              ║
║              Analyze Multiple Directories for Species Concentration          ║
║                  Based on WolfVue by Nathan Bluto                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# Box drawing characters for borders
BOX_CHARS = {
    'h_line': '═',
    'v_line': '║',
    'tl_corner': '╔',
    'tr_corner': '╗',
    'bl_corner': '╚',
    'br_corner': '╝',
    'lt_junction': '╠',
    'rt_junction': '╣',
    'tt_junction': '╦',
    'bt_junction': '╩',
    'cross': '╬'
}

# Terminal colors for pretty output
class Colors:
    # Base colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Text styles
    BOLD = '\033[1m'
    FAINT = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Reset
    END = '\033[0m'
    
    # Special combinations for UI
    HEADER = BOLD + BRIGHT_BLUE
    SUBHEADER = BOLD + BRIGHT_CYAN
    SUCCESS = BRIGHT_GREEN
    WARNING = BRIGHT_YELLOW
    ERROR = BRIGHT_RED
    INFO = BRIGHT_CYAN
    HIGHLIGHT = BOLD + BRIGHT_WHITE
    SUBTLE = BRIGHT_BLACK

# Get terminal width
def get_terminal_width():
    """Get the width of the terminal."""
    try:
        width = shutil.get_terminal_size().columns
    except (AttributeError, ValueError, OSError):
        width = 80  # Default width
    return width

# Initialize colors for Windows if needed
def init_colors():
    """Initialize colors for Windows terminal if needed."""
    if IS_WINDOWS:
        try:
            import colorama
            colorama.init()
        except ImportError:
            # If colorama is not available, disable colors
            for name in dir(Colors):
                if not name.startswith('__') and isinstance(getattr(Colors, name), str):
                    setattr(Colors, name, '')

def center_text_block(text, width=None):
    """Center a block of text as a whole, preserving relative spacing."""
    if width is None:
        width = get_terminal_width()
    
    lines = text.rstrip().split('\n')
    # Find the maximum line length
    max_length = max(len(line) for line in lines)
    # Calculate left padding for the entire block
    left_padding = max(0, (width - max_length) // 2)
    
    # Apply padding to each line
    padded_lines = [' ' * left_padding + line for line in lines]
    return '\n'.join(padded_lines)

def center_text(text, width=None):
    """Center a single line of text."""
    if width is None:
        width = get_terminal_width()
    
    # Remove color codes for length calculation
    clean_text = text
    for name in dir(Colors):
        if not name.startswith('__') and isinstance(getattr(Colors, name), str):
            clean_text = clean_text.replace(getattr(Colors, name), '')
    clean_text = clean_text.replace(Colors.END, '')
    
    spaces = max(0, (width - len(clean_text)) // 2)
    return ' ' * spaces + text

def truncate_path(path, max_length=MAX_PATH_DISPLAY_LENGTH):
    """Truncate a path for display purposes."""
    path = str(path)  # Convert Path object to string if needed
    if len(path) <= max_length:
        return path
    
    parts = Path(path).parts
    result = str(Path(*parts[-2:]))  # Start with just the last two parts
    
    # Add more parts from the end until we reach max length
    i = 3
    while i <= len(parts) and len(str(Path(*parts[-i:]))) <= max_length:
        result = str(Path(*parts[-i:]))
        i += 1
    
    # If we couldn't fit even with just the last parts, just truncate
    if len(result) > max_length:
        return "..." + path[-(max_length-3):]
    
    # Add ... at the beginning to indicate truncation
    return "..." + os.path.sep + result

def draw_box(content, width=80, title=None, footer=None, style='single'):
    """Draw a box around content."""
    if style == 'double':
        chars = BOX_CHARS
    else:  # single
        chars = {
            'h_line': '─',
            'v_line': '│',
            'tl_corner': '┌',
            'tr_corner': '┐',
            'bl_corner': '└',
            'br_corner': '┘',
            'lt_junction': '├',
            'rt_junction': '┤',
            'tt_junction': '┬',
            'bt_junction': '┴',
            'cross': '┼'
        }
    
    # Split content into lines
    lines = content.strip().split('\n')
    
    # Calculate width if not specified
    if width is None:
        width = max(len(line) for line in lines) + 4  # padding
    
    # Ensure width is enough for title and footer
    if title:
        width = max(width, len(title) + 4)
    if footer:
        width = max(width, len(footer) + 4)
    
    # Draw the box
    result = []
    
    # Top border with optional title
    if title:
        title_space = width - 4
        title_text = f" {title} "
        padding = title_space - len(title_text)
        left_pad = padding // 2
        right_pad = padding - left_pad
        top_border = (
            f"{chars['tl_corner']}{chars['h_line'] * left_pad}"
            f"{title_text}"
            f"{chars['h_line'] * right_pad}{chars['tr_corner']}"
        )
    else:
        top_border = f"{chars['tl_corner']}{chars['h_line'] * (width - 2)}{chars['tr_corner']}"
    
    result.append(top_border)
    
    # Content lines
    for line in lines:
        line_space = width - 4
        line_length = len(line.strip())
        padding = line_space - line_length
        left_pad = padding // 2
        right_pad = padding - left_pad
        result.append(f"{chars['v_line']} {' ' * left_pad}{line.strip()}{' ' * right_pad} {chars['v_line']}")
    
    # Bottom border with optional footer
    if footer:
        footer_space = width - 4
        footer_text = f" {footer} "
        padding = footer_space - len(footer_text)
        left_pad = padding // 2
        right_pad = padding - left_pad
        bottom_border = (
            f"{chars['bl_corner']}{chars['h_line'] * left_pad}"
            f"{footer_text}"
            f"{chars['h_line'] * right_pad}{chars['br_corner']}"
        )
    else:
        bottom_border = f"{chars['bl_corner']}{chars['h_line'] * (width - 2)}{chars['br_corner']}"
    
    result.append(bottom_border)
    
    return '\n'.join(result)

def print_fancy_header(text, width=None):
    """Print a fancy header with gradient-style decoration."""
    if width is None:
        width = get_terminal_width()
    
    print(f"\n{Colors.HEADER}{BOX_CHARS['h_line'] * width}{Colors.END}")
    print(f"{Colors.HEADER}{BOX_CHARS['v_line']}{Colors.END}{Colors.BOLD}{Colors.BRIGHT_WHITE}{text.center(width-2)}{Colors.END}{Colors.HEADER}{BOX_CHARS['v_line']}{Colors.END}")
    print(f"{Colors.HEADER}{BOX_CHARS['h_line'] * width}{Colors.END}")

def print_subheader(text):
    """Print a formatted subheader."""
    print(f"\n{Colors.SUBHEADER}{text}{Colors.END}")
    print(f"{Colors.BRIGHT_CYAN}{BOX_CHARS['h_line'] * len(text)}{Colors.END}")

def print_success(text):
    """Print a success message."""
    print(f"{Colors.SUCCESS}✓ {text}{Colors.END}")

def print_warning(text):
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.END}")

def print_error(text):
    """Print an error message."""
    print(f"{Colors.ERROR}✗ {text}{Colors.END}")

def print_info(text):
    """Print an info message."""
    print(f"{Colors.INFO}ℹ {text}{Colors.END}")

def print_result(text):
    """Print a result message."""
    print(f"{Colors.HIGHLIGHT}{text}{Colors.END}")

def clear_current_line():
    """Clear the current line in the terminal."""
    sys.stdout.write("\r" + " " * 100)
    sys.stdout.write("\r")
    sys.stdout.flush()

def create_progress_bar(progress, total, width=PROGRESS_BAR_WIDTH):
    """Create a text-based progress bar."""
    percent = int(progress * 100 / total)
    filled_length = int(width * progress // total)
    
    # Create gradient-style progress bar
    if filled_length > 0:
        bar = '█' * filled_length + '░' * (width - filled_length)
    else:
        bar = '░' * width
    
    return f"[{bar}] {percent}%"

def format_time(seconds):
    """Format seconds into a readable time string."""
    return str(timedelta(seconds=int(seconds)))

def load_config(config_file):
    """Load and parse the YAML configuration file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print_error(f"Error loading configuration file: {e}")
        sys.exit(1)

def load_yolo_model(model_path):
    """Load the YOLO model."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print_error(f"Error loading YOLO model: {e}")
        sys.exit(1)

def get_all_files_in_directory(directory):
    """Get all image and video files in a directory."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    all_files = []
    
    # Find all files
    for ext in video_extensions + image_extensions:
        # Check both lowercase and uppercase
        all_files.extend(list(Path(directory).glob(f'*{ext}')))
        all_files.extend(list(Path(directory).glob(f'*{ext.upper()}')))
    
    # Remove duplicates and separate by type
    all_files = list(set(all_files))
    
    video_files = []
    image_files = []
    
    for file_path in all_files:
        ext = file_path.suffix.lower()
        if ext in video_extensions:
            video_files.append(file_path)
        elif ext in image_extensions:
            image_files.append(file_path)
    
    return video_files, image_files

def analyze_file_with_yolo(file_path, model, class_names):
    """Analyze a single file (image or video) and return species counts."""
    species_counts = defaultdict(int)
    file_ext = file_path.suffix.lower()
    
    # Determine if it's an image or video
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    is_image = file_ext in image_extensions
    
    try:
        if is_image:
            # Process image
            image = cv2.imread(str(file_path))
            if image is None:
                return species_counts
            
            results = model(image)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())
                    
                    if conf >= IMAGE_CONFIDENCE_THRESHOLD:
                        species_name = class_names[cls_id]
                        species_counts[species_name] += 1
        else:
            # Process video
            video = cv2.VideoCapture(str(file_path))
            if not video.isOpened():
                return species_counts
            
            frame_count = 0
            while True:
                success, frame = video.read()
                if not success:
                    break
                
                frame_count += 1
                # Process every 30th frame to speed up analysis
                if frame_count % 30 == 0:
                    results = model(frame)
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            conf = box.conf[0].item()
                            cls_id = int(box.cls[0].item())
                            
                            if conf >= CONFIDENCE_THRESHOLD:
                                species_name = class_names[cls_id]
                                species_counts[species_name] += 1
            
            video.release()
    
    except Exception as e:
        print_warning(f"Error processing {file_path.name}: {e}")
    
    return species_counts

def analyze_directory(directory_path, model, class_names):
    """Analyze all files in a directory and return species counts."""
    print_subheader(f"Analyzing directory: {truncate_path(directory_path)}")
    
    video_files, image_files = get_all_files_in_directory(directory_path)
    total_files = len(video_files) + len(image_files)
    
    if total_files == 0:
        print_warning(f"No image or video files found in {directory_path}")
        return {}
    
    print_info(f"Found {len(video_files)} videos and {len(image_files)} images ({total_files} total)")
    
    # Initialize species counts
    species_counts = defaultdict(int)
    processed_files = 0
    
    # Process all files
    all_files = video_files + image_files
    
    if TQDM_AVAILABLE:
        # Use tqdm for progress bar
        for file_path in tqdm(all_files, desc="Processing files", unit="file"):
            file_counts = analyze_file_with_yolo(file_path, model, class_names)
            for species, count in file_counts.items():
                species_counts[species] += count
    else:
        # Manual progress bar
        for file_path in all_files:
            file_counts = analyze_file_with_yolo(file_path, model, class_names)
            for species, count in file_counts.items():
                species_counts[species] += count
            
            processed_files += 1
            
            # Update progress
            if processed_files % 5 == 0 or processed_files == total_files:
                progress_bar = create_progress_bar(processed_files, total_files)
                clear_current_line()
                sys.stdout.write(f"\rProcessing files: {progress_bar} ({processed_files}/{total_files})")
                sys.stdout.flush()
        
        print()  # New line after progress bar
    
    # Convert defaultdict to regular dict and filter out zero counts
    species_counts = {species: count for species, count in species_counts.items() if count > 0}
    
    print_success(f"Analysis complete! Found {sum(species_counts.values())} total detections across {len(species_counts)} species")
    
    return species_counts

def get_directories_from_user():
    """Get directory paths from user input."""
    directories = []
    
    print_fancy_header("DIRECTORY INPUT")
    print_info("Enter the directories you want to analyze. Press Enter with empty input when done.")
    print_info("You can drag and drop folders into the terminal or type paths manually.")
    
    while True:
        dir_input = input(f"\n{Colors.BOLD}Enter directory path (or press Enter to finish): {Colors.END}").strip()
        
        if not dir_input:
            break
        
        # Clean the path (remove quotes if present)
        dir_input = dir_input.strip('"\'')
        
        # Check if directory exists
        if os.path.isdir(dir_input):
            directories.append(Path(dir_input))
            print_success(f"Added: {truncate_path(dir_input)}")
        else:
            print_error(f"Directory not found: {dir_input}")
            continue
    
    return directories

def create_species_ranking_report(directory_results, class_names):
    """Create a ranking report showing directories with highest counts for each species."""
    print_fancy_header("SPECIES CONCENTRATION RANKING")
    
    # Get all unique species found across all directories
    all_species = set()
    for counts in directory_results.values():
        all_species.update(counts.keys())
    
    # Sort species alphabetically for consistent output
    all_species = sorted(all_species)
    
    # Create ranking for each species
    for species in all_species:
        print_subheader(f"🎯 Highest {species} Concentrations")
        
        # Get directories that have this species and sort by count
        species_dirs = []
        for dir_path, counts in directory_results.items():
            if species in counts:
                species_dirs.append((dir_path, counts[species]))
        
        # Sort by count (descending)
        species_dirs.sort(key=lambda x: x[1], reverse=True)
        
        if species_dirs:
            for i, (dir_path, count) in enumerate(species_dirs[:5], 1):  # Show top 5
                dir_name = dir_path.name
                if len(dir_name) > 40:
                    dir_name = "..." + dir_name[-37:]
                
                if i == 1:
                    print(f"  {Colors.SUCCESS}🏆 #{i}: {dir_name:<40} - {count:>6} detections{Colors.END}")
                elif i == 2:
                    print(f"  {Colors.WARNING}🥈 #{i}: {dir_name:<40} - {count:>6} detections{Colors.END}")
                elif i == 3:
                    print(f"  {Colors.YELLOW}🥉 #{i}: {dir_name:<40} - {count:>6} detections{Colors.END}")
                else:
                    print(f"  {Colors.INFO}   #{i}: {dir_name:<40} - {count:>6} detections{Colors.END}")
            
            if len(species_dirs) > 5:
                print(f"  {Colors.SUBTLE}   ... and {len(species_dirs) - 5} more directories{Colors.END}")
        else:
            print(f"  {Colors.SUBTLE}No detections found{Colors.END}")
        
        print()

def create_directory_summary_report(directory_results):
    """Create a summary report showing all species counts per directory."""
    print_fancy_header("DIRECTORY ANALYSIS SUMMARY")
    
    for dir_path, species_counts in directory_results.items():
        dir_name = dir_path.name
        print_subheader(f"📁 {dir_name}")
        
        if species_counts:
            # Sort species by count (descending)
            sorted_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)
            
            total_detections = sum(species_counts.values())
            print_info(f"Total detections: {total_detections}")
            
            for species, count in sorted_species:
                percentage = (count / total_detections) * 100
                print(f"  {Colors.HIGHLIGHT}{species:<15}{Colors.END}: {count:>6} ({percentage:>5.1f}%)")
        else:
            print_warning("No wildlife detected")
        
        print()

def save_detailed_report(directory_results, report_path):
    """Save a detailed report to a text file."""
    with open(report_path, 'w') as f:
        f.write("Wildlife Directory Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write("Generated by Wildlife Directory Analyzer\n")
        f.write(f"Analysis completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Directory summaries
        f.write("DIRECTORY SUMMARIES\n")
        f.write("-" * 20 + "\n\n")
        
        for dir_path, species_counts in directory_results.items():
            f.write(f"Directory: {dir_path}\n")
            if species_counts:
                total_detections = sum(species_counts.values())
                f.write(f"Total detections: {total_detections}\n")
                
                # Sort by count
                sorted_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)
                for species, count in sorted_species:
                    percentage = (count / total_detections) * 100
                    f.write(f"  {species}: {count} ({percentage:.1f}%)\n")
            else:
                f.write("  No wildlife detected\n")
            f.write("\n")
        
        # Species rankings
        f.write("\nSPECIES CONCENTRATION RANKINGS\n")
        f.write("-" * 30 + "\n\n")
        
        # Get all unique species
        all_species = set()
        for counts in directory_results.values():
            all_species.update(counts.keys())
        
        for species in sorted(all_species):
            f.write(f"{species} Rankings:\n")
            
            # Get and sort directories for this species
            species_dirs = []
            for dir_path, counts in directory_results.items():
                if species in counts:
                    species_dirs.append((dir_path, counts[species]))
            
            species_dirs.sort(key=lambda x: x[1], reverse=True)
            
            for i, (dir_path, count) in enumerate(species_dirs, 1):
                f.write(f"  {i}. {dir_path.name}: {count} detections\n")
            
            if not species_dirs:
                f.write("  No detections found\n")
            f.write("\n")

def display_splash_screen():
    """Display a splash screen with analyzer ASCII art and app title."""
    # Clear screen
    os.system('cls' if IS_WINDOWS else 'clear')
    
    # Get terminal width
    width = get_terminal_width()
    
    # Print analyzer ASCII art (as a single block to preserve formatting)
    print(center_text_block(ANALYZER_ASCII_ART))
    
    # Print centered title
    print(center_text_block(TITLE_DISPLAY))
    
    # Add a small delay for effect
    time.sleep(1)

def clean_path(path):
    """Clean a path by removing quotes and extra spaces."""
    if not path:
        return path
    
    # Remove leading/trailing whitespace
    path = path.strip()
    
    # Remove quotes if present
    if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
        path = path[1:-1]
    
    return path

def main():
    """Main function to run the analyzer."""
    # Initialize colors
    init_colors()
    
    # Display fancy splash screen
    display_splash_screen()
    
    # Get configuration and model paths
    config_path = input(f"{Colors.BOLD}Enter the path to the YAML configuration file (or press Enter to use default): {Colors.END}").strip()
    config_path = clean_path(config_path)
    if not config_path:
        config_path = CONFIG_FILE
    
    model_path = input(f"{Colors.BOLD}Enter the YOLO model path (or press Enter to use default): {Colors.END}").strip()
    model_path = clean_path(model_path)
    if not model_path:
        model_path = DEFAULT_MODEL_PATH
    
    # Load configuration and model
    print_subheader("Loading Configuration and Model")
    config = load_config(config_path)
    class_names = config.get('names', {})
    print_success(f"Loaded {len(class_names)} species classifications")
    
    model = load_yolo_model(model_path)
    print_success(f"Model loaded successfully")
    
    # Get directories to analyze
    directories = get_directories_from_user()
    
    if not directories:
        print_error("No directories provided. Exiting.")
        sys.exit(1)
    
    print_success(f"Will analyze {len(directories)} directories")
    
    # Analyze each directory
    directory_results = {}
    total_start_time = time.time()
    
    for i, directory in enumerate(directories, 1):
        print_fancy_header(f"ANALYZING DIRECTORY {i} OF {len(directories)}")
        start_time = time.time()
        
        species_counts = analyze_directory(directory, model, class_names)
        directory_results[directory] = species_counts
        
        analysis_time = time.time() - start_time
        print_success(f"Directory analysis completed in {format_time(analysis_time)}")
        
        # Show quick summary
        if species_counts:
            total_detections = sum(species_counts.values())
            top_species = max(species_counts.items(), key=lambda x: x[1])
            print_info(f"Quick summary: {total_detections} total detections, top species: {top_species[0]} ({top_species[1]})")
        
        print(f"{Colors.SUBTLE}{BOX_CHARS['h_line'] * get_terminal_width()}{Colors.END}")
    
    total_time = time.time() - total_start_time
    
    # Generate reports
    print_fancy_header("ANALYSIS COMPLETE - GENERATING REPORTS")
    print_success(f"All directories analyzed in {format_time(total_time)}")
    
    # Create and display reports
    create_directory_summary_report(directory_results)
    create_species_ranking_report(directory_results, class_names)
    
    # Save detailed report
    report_path = Path.cwd() / f"wildlife_analysis_report_{int(time.time())}.txt"
    save_detailed_report(directory_results, report_path)
    print_success(f"Detailed report saved to: {report_path}")
    
    # Final summary
    total_dirs_with_wildlife = sum(1 for counts in directory_results.values() if counts)
    total_detections = sum(sum(counts.values()) for counts in directory_results.values())
    
    final_summary = [
        f"Directories analyzed: {len(directories)}",
        f"Directories with wildlife: {total_dirs_with_wildlife}",
        f"Total detections: {total_detections:,}",
        f"Analysis time: {format_time(total_time)}"
    ]
    
    # Create and center the final summary box
    summary_box = draw_box('\n'.join(final_summary), title="Analysis Complete!", style='double')
    print(center_text_block(summary_box))

if __name__ == "__main__":
    main()