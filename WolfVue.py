#!/usr/bin/env python3
"""
WolfVue: Wildlife Video Classifier
Processes trail camera videos and images using a YOLO model and sorts them into folders
based on detected species according to a predefined taxonomy.

Created by Nathan Bluto
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
from ultralytics import YOLO  # Using Ultralytics YOLOv8 implementation
import shutil  # For getting terminal size

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
VIDEO_PATH = SCRIPT_DIR / "input_videos"  # Input videos
OUTPUT_PATH = SCRIPT_DIR / "output_videos"  # Output folder
CONFIG_FILE = SCRIPT_DIR / "weights" / "Misc_Models" / "WolfVue_Beta_BroadV2" / "WolfVue_BetaV2.yaml"  # YAML file
DEFAULT_MODEL_PATH = SCRIPT_DIR / "weights" / "Misc_Models" / "WolfVue_Beta_BroadV2" / "aug_low_best.pt"  # YOLO model

# VIDEO Algorithm parameters (adjust as needed)
CONFIDENCE_THRESHOLD = 0.40  # Minimum confidence for detections
DOMINANT_SPECIES_THRESHOLD = 0.9  # Minimum % for dominant species (0.7 = 70%)
MAX_SPECIES_TRANSITIONS = 5  # Maximum allowed transitions between species
CONSECUTIVE_EMPTY_FRAMES = 15  # Frames without detection to break a sequence

# IMAGE Algorithm parameters (adjust as needed for photos)
IMAGE_CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence for detections in images
IMAGE_MIN_DETECTIONS = 1  # Minimum number of detections required to classify (set to 1 for any detection)
IMAGE_MULTI_SPECIES_THRESHOLD = 0.60  # If multiple species detected, confidence difference needed to pick winner

# NEW: Image confidence-based unsorted parameters (easily adjustable)
IMAGE_UNSORTED_MIN_CONFIDENCE = 0.35  # Minimum confidence to trigger unsorted classification
IMAGE_UNSORTED_MAX_CONFIDENCE = 0.65  # Maximum confidence to trigger unsorted classification
# NOTE: If confidence is between these values, image goes to "unsorted" folder

# UI Settings
PROGRESS_BAR_WIDTH = 50  # Width of the console progress bar
UPDATE_FREQUENCY = 10  # Update the progress bar every N frames
MAX_PATH_DISPLAY_LENGTH = 60  # Max length to display for paths

# Predator-Prey Classification (for conflict detection)
PREDATORS = ["Cougar", "Lynx", "Wolf", "Coyote", "Fox", "Bear"]
PREY = ["WhiteTail", "MuleDeer", "Elk", "Moose"]

# Simplified taxonomy structure (no redundant subfolders) - NOW LOADED FROM YAML
TAXONOMY = {
    "Ungulates": {
        "WhiteTail": ["WhiteTail"],
        "MuleDeer": ["MuleDeer"],
        "Elk": ["Elk"],
        "Moose": ["Moose"]
    },
    "Predators": {
        "Cougar": ["Cougar"],
        "Lynx": ["Lynx"],
        "Wolf": ["Wolf"],
        "Coyote": ["Coyote"],
        "Fox": ["Fox"],
        "Bear": ["Bear"]
    }
}
# ============= END CONFIGURATION =============

# Check if Windows
IS_WINDOWS = platform.system() == 'Windows'

# ASCII Art to display at startup - preserved exactly as provided
WOLF_ASCII_ART = r"""
                                                       █████████████████████████████
                                                █████████████████████████████████████████
                                          █████████████████████████████████████████████████████
                                       ████████████████████████████████████████████████████████████
                                   ███████████████████████████████████████████████████████████████████
                                ██████████████████████████████████████████████████████   ████████████████
                             ██████████████████████████████████████████████████████        █████████████████
                           █████████████████████████████████     ████████████████      █     █████████████████
                        ███████████████████████████████████       ██████          ████████     ██████████████████
                       ████████████████████████████████████       █████   █████    ████████      ██████████████████
                     █████████████     █████████████████████     ████   ███████    ████████  ███    █████████████████
                   ██████████████            ████████       █████    ███████████    ███████ ██████     ███████████████
                 ███████████████        █████         ██████    ███████████████    ██████████████████  █████████████████
               ████████████████        █████████      ████████   ████████████  ███████████████████████  ██████████████████
              ████████████████       █████████████████ █████████  ████████   ██████████████████████████  ██████████████████
             ████████████████         ████████████████   █████████  ███████████████████████████████████   ███████████████████
           █████████████████          ███████████████████   ████████ ██████████████████████████████████   ████ ███████████████
          █████████████████            ███████████████████   ████████  ████████████████████████   █████         ███████████████
         █████████████████              ██████████ ██████████ █████████  ███████████████████████   ████  ██████  ███████████████
        █████████████████                 ████████████████████████████████████████████████████     ██   ████████ ████████████████
       ██████████████████                    ████████████████████████████████████████████████    ███   ██████████ ████████████████
      ██████████████████                ██████   ██████████████████████████████████████████     ███  █████████████ ████████████████
      █████████████████           █████████████████████████████████████████████████████████     ███ ████████████████████████████████
     ███████████████████████    ██████████████████████████████████████████████████████████████ ████████████████████████   ██████████
    ████████████████████████████████████████████████████████████████████████████████████████████████████████████████    ██  █████████
   ████████████████████████████████████████████████████████████████████████████████████████████████████████████████  ██████  ██  █████
   ███████████     ███████████████████████████████████████████████████████████████████████████████      ███████████  ██████       ████
  █████████    █      ████████████████████████████████████████████████████████████████████████   █   █    ██████████         ██████████
  ████████  █████████  ██████████████████████████████████████████        █████████████████   █████       ████████████    ██████████████
 █████████  ██████████ ██████████████████████████████████████    ███████         █████   ██████  █████  █ ███████████████████   ████████
 █████████  ██████████ █████████████████████████████████       ████████████████    █████████  █   ███████ █████████████         ████████
 █████████      ████  ███████████████████████████████     █████████████      ███████████  ██  ██████████ ██████████████ ███████  ████████
 █████████████  █    ██████████████████████████████    █████████████  █    ██████████████     ███████    ██████████████████████  ████████
████████ ███████████████████████████████████        ███████████████  ████████████████████ █████████ ███ ████████████████        █████████
████████         █████████████████████        ███  ████████████████ █████████████████████████████ ████ █████████████████  ███████████████
███████████████ ███████████             ████      █  █████████████  ██████████████████████████  █████ █████████████████████     █████████
██████████████   █████████   ██    ███    █ ██████  ██████████████ ███████████████████████  ███          ████████████████    ███  ███████
██████████████  █████████   ██              ██████████  ███████████████████████████████████████████████  ████████████████ ███████ ███████
███████     █   ██████████  ██  ██████████████████████ ███████████████████████████████████████████████   ████████████████ ███████ ███████
███████ ███ ███  ██████████   ███████      ████████████  ███████████████████████████████████████████   ███████████████████       ████████
███████  ██  ██  ███████████   ███████████     █████  ███████████████████████████████████████████     ████████████████████ ██ ███████████
 ██████         ██████████████    █████████████████ ████████        ██████████████████████████     ██████████████████████   █████████████
 ██████  ██████  ████████████████   ██████████████ █████████  ██████    ███████████████████    ██████████████████████████ ██████████████
 ████████████    ███████████████████       ██    ███████████████  ████       █████████████  ████████████████████████████       █  ██████
  ██   ████   █████████████████████████       ████████████████████    ██████ ████████  ██ ████████████████████████████████████   ███████
  ███      ███████████████████████████████  ██████████████████████████        ████  ███  ███████████████████████████    ██ ████████████
  ███████         █████████████████████   ███████████████████████████████  █████  ████  ███████████████████████████   ███      ████████
   ██████████████ ███████████████████   ██     ████████████████    ██  ███  ██  ██████  ██████████████████████████  █████ ███   ██████
   ██████████████████ ██████████████          ██████████████████████████████   ████     ██████████████████████████  ██████████ ███████
    █████████████████  █████████████    ███  ███████████    ███     ██  █████  ████  ██ ██████████████████████████  █████████  ██████
     ██████████████   ██████████████  ███   ███  ████        ███ ██  █   █████    █  ███████████████████████████████   █████  ██████
      ████████████  ████████████████████         ██     ███    █████ ███████   ███    ████████████████████████████████       ███████
       █████████   ██████  █████████████     █  █    ██████            ███████  █████ █████████████████████████████████████████████
       ████████  ███        ████████████  ███      ███████   ██  ██████████████  █████████████████████████████████████████████████
        ██████     ███   ████████████████████     ███████   ███████████████████████████████████████████████████    ██████████████
         ████████████  ██████ ██████████████     ████████████████████████████████████████████████████████  ███  ██   ███████████
          █████████   ████     ████████████     █████████████████████████████████████████████████████████    █ █████     ██████
           ███████        █████████████████    █████████████████████████████████████████████████████████████    ██████ ███████
             █████████████      ███████████  ████████████████████████████████████████████████████████████  ████   ███████████
              ███████████  █████ █████  ███████████████████████████████████████████████████████████████  ██ █████   ███████
                ████████  ██████  ███   ██████████████████████████████████████████████████████████████  ██████ ███ ███████
                 ███████  █████  ███  ███     ████████████████████████████████████████████████████   █  ██████  ████████
                   ██████   █   ███  ██   ██  █████████████████████████████████████████████████    ████   ██   ███████
                     █████████████  ███     ██████    ██████████████████████████████████████████  ███████    ████████
                       ████████    ███  ██████████  ███   ██████████████████████████████  █   ███  ████████████████
                         ███████  ███  ███████████  █████  █████████████████████    ███  ████  ███   ███████████
                           ████████   ███████████   ████   ███     ████     ███ ████  ████  █   ███  █████████
                              ███████████████████  ██  █████  ████  █  ████ ██  ██   ███  ███    ███████████
                                ████████████████  ████  ███         ██      ██    ████ █      ███████████
                                    ███████████   █████  ███  █████████████  ██   ██  ████████████████
                                       ███████████████    ██    █   ██  ███  ████   ██████████████
                                           ███████████████████████████████████████████████████
                                                █████████████████████████████████████████
                                                        █████████████████████████
"""

# Title with fancy border
TITLE_DISPLAY = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                 WOLFVUE: WILDLIFE VIDEO CLASSIFIER                           ║
║                                                                              ║
║                  Created by Nathan Bluto                                     ║
║                  Data from The Gray Wolf Research Project                    ║
║                  Facilitated by Dr. Ausband                                  ║
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
        return "..." + path[-(max_length - 3):]

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
    print(
        f"{Colors.HEADER}{BOX_CHARS['v_line']}{Colors.END}{Colors.BOLD}{Colors.BRIGHT_WHITE}{text.center(width - 2)}{Colors.END}{Colors.HEADER}{BOX_CHARS['v_line']}{Colors.END}")
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


def extract_taxonomy_from_config(config):
    """Extract taxonomy structure from the YAML config file."""
    # Try to get taxonomy from config, fall back to default if not found
    if 'taxonomy' in config:
        print_success("Using taxonomy from YAML file")
        return config['taxonomy']
    else:
        print_warning("No taxonomy found in YAML file, using default taxonomy")
        return TAXONOMY


def create_folder_structure(base_path, taxonomy):
    """Create the folder structure based on the taxonomy."""
    print_subheader(f"Creating folder structure in {truncate_path(base_path)}")

    # Create base directories
    os.makedirs(os.path.join(base_path, "Sorted"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "Unsorted"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "No_Animal"), exist_ok=True)  # No animal detections

    # Create taxonomy-based directories (simplified, no redundant subfolders)
    for category, subcategories in taxonomy.items():
        category_path = os.path.join(base_path, "Sorted", category)
        os.makedirs(category_path, exist_ok=True)

        for species, _ in subcategories.items():
            species_path = os.path.join(category_path, species)
            os.makedirs(species_path, exist_ok=True)

    print_success("Folder structure created successfully")


def count_video_frames(video_path):
    """Count frames in a video file."""
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            return 0

        # Get frame count using the property
        count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        return count
    except Exception as e:
        print_warning(f"Could not count frames in {os.path.basename(video_path)}: {e}")
        return 0


def pre_scan_files(video_files, image_files):
    """Count total frames and estimate processing time."""
    print_subheader("Pre-scanning files to estimate processing time...")

    total_frames = 0
    total_videos = len(video_files)
    total_images = len(image_files)
    total_files = total_videos + total_images

    if TQDM_AVAILABLE:
        # Count video frames
        for i, video_file in enumerate(tqdm(video_files, desc="Scanning videos", unit="video")):
            frames = count_video_frames(str(video_file))
            total_frames += frames

        # Images are 1 frame each
        total_frames += total_images

    else:
        # Count video frames
        for i, video_file in enumerate(video_files):
            frames = count_video_frames(str(video_file))
            total_frames += frames

            # Update progress
            progress = (i + 1) / total_files if total_files > 0 else 0
            progress_bar = create_progress_bar(i + 1, total_files)
            clear_current_line()
            sys.stdout.write(f"\rScanning files: {progress_bar} ({i + 1}/{total_files})")
            sys.stdout.flush()

        # Images are 1 frame each
        total_frames += total_images

        # Update final progress
        if total_files > 0:
            progress_bar = create_progress_bar(total_files, total_files)
            clear_current_line()
            sys.stdout.write(f"\rScanning files: {progress_bar} ({total_files}/{total_files})")
            sys.stdout.flush()

    print("\n")  # New line after progress bar

    # Estimate time (rough estimate: ~0.1 seconds per frame)
    estimate_time_seconds = total_frames * 0.1

    # Create a nice box for the summary
    summary = [
        f"Total videos: {total_videos}",
        f"Total images: {total_images}",
        f"Total frames: {total_frames:,}",
        f"Estimated processing time: {format_time(estimate_time_seconds)}"
    ]

    boxed_summary = draw_box('\n'.join(summary), title="Pre-Scan Results", style='double')
    # Center the entire box, not each line
    print(center_text(boxed_summary))

    return total_frames


def process_image_with_yolo(image_path, model, class_names):
    """Process image with YOLO model and return detection data in same format as video frames."""
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print_error(f"Error: Could not open image {os.path.basename(image_path)}")
            return None

        print_info(f"Processing image: {Colors.BOLD}{os.path.basename(image_path)}{Colors.END}")

        start_time = time.time()

        # Run YOLO detection on the image
        results = model(image)

        # Extract detections into a structured format
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())

                # Use image-specific confidence threshold
                if conf >= IMAGE_CONFIDENCE_THRESHOLD:
                    detections.append({
                        'class_id': cls_id,
                        'class_name': class_names[cls_id],
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })

        # Create frame data (single frame for images)
        frame_data = [{
            'frame_idx': 0,
            'timestamp': 0.0,
            'detections': detections
        }]

        processing_time = time.time() - start_time
        print_success(f"Completed in {format_time(processing_time)}")

        return frame_data

    except Exception as e:
        print_error(f"Error processing image {os.path.basename(image_path)}: {e}")
        return None


def process_video_with_yolo(video_path, model, class_names, total_processed_frames=0, total_frames=1):
    """Process video with YOLO model and collect frame-by-frame detection data."""
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print_error(f"Error: Could not open video {os.path.basename(video_path)}")
        return None

    video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    print_info(
        f"Processing video: {Colors.BOLD}{os.path.basename(video_path)}{Colors.END} {Colors.INFO}({video_frames:,} frames @ {fps:.1f} fps){Colors.END}")

    frame_data = []
    frame_idx = 0
    start_time = time.time()
    last_update_time = start_time

    while True:
        success, frame = video.read()
        if not success:
            break

        # Run YOLO detection on this frame
        results = model(frame)

        # Extract detections into a structured format
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())

                # Use video confidence threshold for videos
                if conf >= CONFIDENCE_THRESHOLD:
                    detections.append({
                        'class_id': cls_id,
                        'class_name': class_names[cls_id],
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })

        # Store frame data
        frame_data.append({
            'frame_idx': frame_idx,
            'timestamp': frame_idx / fps,
            'detections': detections
        })

        frame_idx += 1
        total_processed_frames += 1

        # Update progress bar (but not too frequently to avoid slowing down)
        current_time = time.time()
        if frame_idx % UPDATE_FREQUENCY == 0 or frame_idx == video_frames:
            # Calculate progress and time estimates
            elapsed = current_time - start_time
            progress = frame_idx / video_frames
            video_eta = elapsed / progress - elapsed if progress > 0 else 0

            overall_progress = total_processed_frames / total_frames

            # Create progress indicators
            video_progress = create_progress_bar(frame_idx, video_frames, width=30)
            overall_progress_bar = create_progress_bar(total_processed_frames, total_frames)

            # Calculate processing speed
            fps_processing = frame_idx / elapsed if elapsed > 0 else 0

            # Clear and update the line
            clear_current_line()
            status_line = (
                f"\r{Colors.INFO}Processing video:{Colors.END} {video_progress} | "
                f"{Colors.SUBHEADER}Overall:{Colors.END} {overall_progress_bar} | "
                f"{Colors.SUCCESS}Speed:{Colors.END} {fps_processing:.1f} fps | "
                f"{Colors.WARNING}ETA:{Colors.END} {format_time(video_eta)}"
            )
            sys.stdout.write(status_line)
            sys.stdout.flush()

            last_update_time = current_time

    # Complete the progress bar
    clear_current_line()
    processing_time = time.time() - start_time
    print_success(f"Completed in {format_time(processing_time)} ({frame_idx / processing_time:.1f} fps)")

    video.release()
    return frame_data


def analyze_detections(frame_data, class_names):
    """
    Analyze frame detections to determine video classification.
    Implements the sorting algorithm with the specified rules.
    Uses simplified logic for single-frame images.
    """
    total_frames = len(frame_data)

    # Check if this is an image (single frame) and use simplified logic
    if total_frames == 1:
        return analyze_image_detections(frame_data[0], class_names)

    # Original video analysis logic below
    # Initialize counters and tracking variables
    frames_with_detections = 0
    species_counts = {name: 0 for name in class_names.values()}
    species_frames = {name: [] for name in class_names.values()}

    # Temporal pattern tracking
    current_species = None
    species_transitions = 0
    frames_without_detection = 0
    clusters = []
    current_cluster = {'species': None, 'start': 0, 'end': 0, 'frames': 0}

    # Analyze each frame
    for i, frame in enumerate(frame_data):
        detections = frame['detections']

        if not detections:
            frames_without_detection += 1

            # Check if this breaks a detection cluster
            if frames_without_detection >= CONSECUTIVE_EMPTY_FRAMES and current_cluster['species']:
                current_cluster['end'] = i - frames_without_detection
                clusters.append(current_cluster)
                current_cluster = {'species': None, 'start': 0, 'end': 0, 'frames': 0}

            continue

        frames_with_detections += 1
        frames_without_detection = 0

        # Count detections by species
        frame_species = {}
        for detection in detections:
            species = detection['class_name']
            species_counts[species] += 1
            if i not in species_frames[species]:
                species_frames[species].append(i)

            if species not in frame_species:
                frame_species[species] = 0
            frame_species[species] += 1

        # Determine dominant species in this frame
        if frame_species:
            dominant_species = max(frame_species, key=frame_species.get)

            # Track species transitions
            if current_species and dominant_species != current_species:
                species_transitions += 1

            current_species = dominant_species

            # Update or start new cluster
            if current_cluster['species'] != current_species:
                if current_cluster['species']:
                    current_cluster['end'] = i - 1
                    clusters.append(current_cluster)

                current_cluster = {
                    'species': current_species,
                    'start': i,
                    'end': i,
                    'frames': 1
                }
            else:
                current_cluster['end'] = i
                current_cluster['frames'] += 1

    # Add final cluster if it exists
    if current_cluster['species']:
        clusters.append(current_cluster)

    # Calculate percentages
    total_detections = sum(species_counts.values())
    if total_detections > 0:
        species_percentages = {
            species: count / total_detections
            for species, count in species_counts.items() if count > 0
        }
    else:
        species_percentages = {}

    # Frame coverage percentages
    species_frame_percentages = {
        species: len(frames) / total_frames
        for species, frames in species_frames.items() if frames
    }

    # Sort clusters by frame count (largest first)
    sorted_clusters = sorted(clusters, key=lambda x: x['frames'], reverse=True)

    # Determine dominant species overall
    dominant_species = max(species_counts, key=species_counts.get) if species_counts else None

    # Check for predator-prey conflict
    has_predator = any(species_counts[predator] > 0 for predator in PREDATORS if predator in species_counts)
    has_prey = any(species_counts[prey] > 0 for prey in PREY if prey in species_counts)
    predator_prey_conflict = has_predator and has_prey

    # Check for significant species (>10% of frames with detections)
    significant_species = [s for s, p in species_frame_percentages.items() if p >= 0.1]

    # Detection rate
    detection_rate = frames_with_detections / total_frames if total_frames > 0 else 0

    # Make classification decision - STRICTER NO_ANIMAL rule (ZERO animals detected)
    if frames_with_detections == 0:  # Strict: No animals at all
        classification = "No_Animal"
        reason = "No animals detected in any frames"
    elif predator_prey_conflict:
        classification = "Unsorted"
        reason = "Both predator and prey detected"
    elif species_transitions > MAX_SPECIES_TRANSITIONS and len(significant_species) > 1:
        classification = "Unsorted"
        reason = f"Too many species transitions ({species_transitions})"
    elif dominant_species and species_percentages.get(dominant_species, 0) >= DOMINANT_SPECIES_THRESHOLD:
        classification = dominant_species
        reason = f"Dominant species ({dominant_species}) with {species_percentages[dominant_species] * 100:.1f}% of detections"
    else:
        classification = "Unsorted"
        reason = "No clear dominant species"

    return {
        'total_frames': total_frames,
        'frames_with_detections': frames_with_detections,
        'detection_rate': detection_rate,
        'species_counts': species_counts,
        'species_percentages': species_percentages,
        'species_frame_percentages': species_frame_percentages,
        'species_transitions': species_transitions,
        'clusters': sorted_clusters,
        'classification': classification,
        'reason': reason
    }


def analyze_image_detections(frame_data, class_names):
    """
    Simplified analysis for single-frame images.
    Uses image-specific thresholds and logic.
    """
    detections = frame_data['detections']

    # Count detections by species
    species_counts = {}
    species_confidences = {}

    for detection in detections:
        species = detection['class_name']
        confidence = detection['confidence']

        if species not in species_counts:
            species_counts[species] = 0
            species_confidences[species] = []

        species_counts[species] += 1
        species_confidences[species].append(confidence)

    # Calculate average confidence per species
    species_avg_confidence = {}
    for species, confidences in species_confidences.items():
        species_avg_confidence[species] = sum(confidences) / len(confidences)

    # Calculate percentages
    total_detections = sum(species_counts.values())
    if total_detections > 0:
        species_percentages = {
            species: count / total_detections
            for species, count in species_counts.items()
        }
    else:
        species_percentages = {}

    # NEW: Check for confidence-based unsorted classification
    max_confidence = max((conf for confs in species_confidences.values() for conf in confs), default=0)
    confidence_unsorted = (IMAGE_UNSORTED_MIN_CONFIDENCE <= max_confidence <= IMAGE_UNSORTED_MAX_CONFIDENCE)

    # Make classification decision for images
    if total_detections < IMAGE_MIN_DETECTIONS:
        classification = "No_Animal"
        reason = f"Insufficient detections (found {total_detections}, need {IMAGE_MIN_DETECTIONS})"
    elif confidence_unsorted:
        classification = "Unsorted"
        reason = f"Confidence between unsorted range ({max_confidence:.2f} between {IMAGE_UNSORTED_MIN_CONFIDENCE:.2f}-{IMAGE_UNSORTED_MAX_CONFIDENCE:.2f})"
    elif len(species_counts) == 1:
        # Single species detected
        classification = list(species_counts.keys())[0]
        confidence = species_avg_confidence[classification]
        reason = f"Single species detected with {confidence:.2f} confidence"
    elif len(species_counts) > 1:
        # Multiple species detected - check confidence difference
        sorted_species = sorted(species_avg_confidence.items(), key=lambda x: x[1], reverse=True)
        highest_species, highest_conf = sorted_species[0]
        second_species, second_conf = sorted_species[1]

        # Check for predator-prey conflict
        detected_species = list(species_counts.keys())
        has_predator = any(species in PREDATORS for species in detected_species)
        has_prey = any(species in PREY for species in detected_species)

        if has_predator and has_prey:
            classification = "Unsorted"
            reason = "Both predator and prey detected in image"
        elif highest_conf - second_conf >= IMAGE_MULTI_SPECIES_THRESHOLD:
            classification = highest_species
            reason = f"Clear winner: {highest_species} ({highest_conf:.2f}) vs {second_species} ({second_conf:.2f})"
        else:
            classification = "Unsorted"
            reason = f"Multiple species with similar confidence: {highest_species} ({highest_conf:.2f}) vs {second_species} ({second_conf:.2f})"
    else:
        classification = "No_Animal"
        reason = "No valid detections found"

    return {
        'total_frames': 1,
        'frames_with_detections': 1 if total_detections > 0 else 0,
        'detection_rate': 1.0 if total_detections > 0 else 0.0,
        'species_counts': species_counts,
        'species_percentages': species_percentages,
        'species_frame_percentages': species_percentages,  # Same as percentages for single frame
        'species_transitions': 0,  # N/A for images
        'clusters': [],  # N/A for images
        'classification': classification,
        'reason': reason
    }


def get_species_folder_path(base_path, species, taxonomy):
    """Get the folder path for a species based on the taxonomy."""
    # Special cases
    if species == "Unsorted":
        return os.path.join(base_path, "Unsorted")
    elif species == "No_Animal":
        return os.path.join(base_path, "No_Animal")

    # Find the species in the taxonomy
    for category, subcategories in taxonomy.items():
        if species in subcategories:
            return os.path.join(base_path, "Sorted", category, species)

    # NEW: If not found in taxonomy, create dynamic folder in "Other" category
    other_category_path = os.path.join(base_path, "Sorted", "Other", species)

    # Create the "Other" category and species folder if they don't exist
    os.makedirs(other_category_path, exist_ok=True)

    return other_category_path


def create_folder_structure(base_path, taxonomy):
    """Create the folder structure based on the taxonomy."""
    print_subheader(f"Creating folder structure in {truncate_path(base_path)}")

    # Create base directories
    os.makedirs(os.path.join(base_path, "Sorted"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "Unsorted"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "No_Animal"), exist_ok=True)  # No animal detections

    # Create taxonomy-based directories (simplified, no redundant subfolders)
    for category, subcategories in taxonomy.items():
        category_path = os.path.join(base_path, "Sorted", category)
        os.makedirs(category_path, exist_ok=True)

        for species, _ in subcategories.items():
            species_path = os.path.join(category_path, species)
            os.makedirs(species_path, exist_ok=True)

    # NEW: Create "Other" category for dynamic species
    other_category_path = os.path.join(base_path, "Sorted", "Other")
    os.makedirs(other_category_path, exist_ok=True)

    print_success("Folder structure created successfully")


def process_all_files(input_folder, output_folder, model_path, config):
    """Process all videos and images in the folder and sort them."""
    # Load model
    print_subheader("Loading YOLO model")
    model = load_yolo_model(model_path)
    print_success(f"Model loaded successfully from: {truncate_path(model_path)}")

    # Get class names from config
    class_names = config.get('names', {})
    print_info(f"Loaded {len(class_names)} species classifications")

    # Extract taxonomy from config
    taxonomy = extract_taxonomy_from_config(config)

    # Create folder structure
    create_folder_structure(output_folder, taxonomy)

    # Get all video files - check both upper and lowercase extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []

    # Get all image files - check both upper and lowercase extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []

    # Debug print to see what folder we're checking
    print_info(f"Looking for files in: {input_folder}")

    # Find video files
    for ext in video_extensions:
        # Check both lowercase and uppercase
        video_files.extend(list(Path(input_folder).glob(f'*{ext}')))
        video_files.extend(list(Path(input_folder).glob(f'*{ext.upper()}')))

    # Find image files
    for ext in image_extensions:
        # Check both lowercase and uppercase
        image_files.extend(list(Path(input_folder).glob(f'*{ext}')))
        image_files.extend(list(Path(input_folder).glob(f'*{ext.upper()}')))

    # Remove duplicates
    video_files = list(set(video_files))
    image_files = list(set(image_files))

    # If no files found, show what's in the folder
    if not video_files and not image_files:
        print_warning("No video or image files found. Contents of input folder:")
        try:
            all_files = list(Path(input_folder).glob('*'))
            for f in all_files[:10]:  # Show first 10 files
                print(f"  - {f.name}")
            if len(all_files) > 10:
                print(f"  ... and {len(all_files) - 10} more files")
        except Exception as e:
            print_error(f"Could not list directory contents: {e}")

    if not video_files and not image_files:
        print_warning("No video or image files found in the specified folder.")
        return []

    total_files = len(video_files) + len(image_files)
    print_success(f"Found {len(video_files)} videos and {len(image_files)} images to process ({total_files} total)")

    # Pre-scan files to get total frame count for overall progress
    total_frames = pre_scan_files(video_files, image_files)

    # Process each file
    results = []
    total_processed_frames = 0
    start_time = time.time()

    print_fancy_header("PROCESSING FILES")

    file_index = 0

    # Process videos
    for video_file in video_files:
        video_path = str(video_file)
        file_index += 1

        # Process video with YOLO
        print_subheader(f"Processing file {file_index} of {total_files} (VIDEO)")
        frame_data = process_video_with_yolo(video_path, model, class_names,
                                             total_processed_frames, total_frames)

        if not frame_data:
            print_error(f"Skipping {os.path.basename(video_path)} due to processing error")
            continue

        total_processed_frames += len(frame_data)

        # Analyze detections
        print_info("Analyzing detections...")
        analysis = analyze_detections(frame_data, class_names)
        classification = analysis['classification']

        # Display classification result
        if classification == "No_Animal":
            result_color = Colors.BLUE
        elif classification == "Unsorted":
            result_color = Colors.YELLOW
        else:
            result_color = Colors.GREEN

        print_result(f"Classification: {result_color}{classification}{Colors.END} ({analysis['reason']})")

        # Show species percentages
        if analysis['species_percentages']:
            species_info = []
            for species, percent in sorted(analysis['species_percentages'].items(), key=lambda x: x[1], reverse=True):
                if percent > 0:
                    species_info.append(f"{species}: {percent * 100:.1f}%")

            print_info("Species Detection: " + ", ".join(species_info))

        print_info(f"Detection rate: {analysis['detection_rate'] * 100:.1f}% "
                   f"({analysis['frames_with_detections']:,}/{analysis['total_frames']:,} frames)")

        # Sort video
        target_path = sort_file(video_path, classification, output_folder, taxonomy)

        # Store result for summary
        results.append({
            'original_path': video_path,
            'target_path': target_path,
            'classification': classification,
            'reason': analysis['reason'],
            'species_percentages': analysis['species_percentages'],
            'detection_rate': analysis['detection_rate'],
            'file_type': 'video'
        })

        # Print separator between files
        width = get_terminal_width()
        print(f"{Colors.SUBTLE}{BOX_CHARS['h_line'] * width}{Colors.END}")

    # Process images
    for image_file in image_files:
        image_path = str(image_file)
        file_index += 1

        # Process image with YOLO
        print_subheader(f"Processing file {file_index} of {total_files} (IMAGE)")
        frame_data = process_image_with_yolo(image_path, model, class_names)

        if not frame_data:
            print_error(f"Skipping {os.path.basename(image_path)} due to processing error")
            continue

        total_processed_frames += 1  # Images are 1 frame each

        # Analyze detections
        print_info("Analyzing detections...")
        analysis = analyze_detections(frame_data, class_names)
        classification = analysis['classification']

        # Display classification result
        if classification == "No_Animal":
            result_color = Colors.BLUE
        elif classification == "Unsorted":
            result_color = Colors.YELLOW
        else:
            result_color = Colors.GREEN

        print_result(f"Classification: {result_color}{classification}{Colors.END} ({analysis['reason']})")

        # Show species percentages
        if analysis['species_percentages']:
            species_info = []
            for species, percent in sorted(analysis['species_percentages'].items(), key=lambda x: x[1], reverse=True):
                if percent > 0:
                    species_info.append(f"{species}: {percent * 100:.1f}%")

            print_info("Species Detection: " + ", ".join(species_info))

        print_info(f"Detection rate: {analysis['detection_rate'] * 100:.1f}% (1 frame)")

        # Sort image
        target_path = sort_file(image_path, classification, output_folder, taxonomy)

        # Store result for summary
        results.append({
            'original_path': image_path,
            'target_path': target_path,
            'classification': classification,
            'reason': analysis['reason'],
            'species_percentages': analysis['species_percentages'],
            'detection_rate': analysis['detection_rate'],
            'file_type': 'image'
        })

        # Print separator between files
        width = get_terminal_width()
        print(f"{Colors.SUBTLE}{BOX_CHARS['h_line'] * width}{Colors.END}")

    # Total processing time
    total_time = time.time() - start_time
    print_success(f"All files processed in {format_time(total_time)}")
    print_info(f"Average processing speed: {total_processed_frames / total_time:.1f} frames per second")

    # Generate summary report
    generate_summary_report(results, output_folder)

    return results


def load_yolo_model(model_path):
    """Load the YOLO model."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print_error(f"Error loading YOLO model: {e}")
        sys.exit(1)


def sort_file(file_path, classification, base_path, taxonomy):
    """Move the file to the appropriate folder based on classification."""
    target_folder = get_species_folder_path(base_path, classification, taxonomy)
    file_filename = os.path.basename(file_path)
    target_path = os.path.join(target_folder, file_filename)

    # Create unique filename if target already exists
    if os.path.exists(target_path):
        name, ext = os.path.splitext(file_filename)
        target_path = os.path.join(target_folder, f"{name}_{os.path.getmtime(file_path):.0f}{ext}")

    print_info(f"Moving file to: {truncate_path(target_path)}")
    shutil.copy2(file_path, target_path)
    return target_path


def generate_summary_report(results, output_folder):
    """Generate a summary report of processing results."""
    report_path = os.path.join(output_folder, "processing_report.txt")

    print_subheader("Generating Summary Report")

    with open(report_path, 'w') as f:
        f.write("WolfVue: Wildlife Video Classifier - Processing Report\n")
        f.write("====================================================\n\n")
        f.write("Created by Nathan Bluto\n")
        f.write("Data from The Gray Wolf Research Project\n")
        f.write("Facilitated by Dr. Ausband\n\n")  # Removed asterisks

        # Count by file type
        video_count = sum(1 for r in results if r.get('file_type') == 'video')
        image_count = sum(1 for r in results if r.get('file_type') == 'image')

        f.write(f"Processed {len(results)} files ({video_count} videos, {image_count} images)\n\n")

        # Count by classification
        classifications = {}
        for result in results:
            classification = result['classification']
            if classification not in classifications:
                classifications[classification] = 0
            classifications[classification] += 1

        f.write("Classification Summary:\n")
        for classification, count in sorted(classifications.items()):
            f.write(f"  {classification}: {count} files ({count / len(results) * 100:.1f}%)\n")

        f.write("\nDetailed Results:\n")
        for i, result in enumerate(results, 1):
            file_type = result.get('file_type', 'unknown').upper()
            f.write(f"\n{i}. {os.path.basename(result['original_path'])} ({file_type})\n")
            f.write(f"   Classification: {result['classification']}\n")
            f.write(f"   Reason: {result['reason']}\n")
            f.write(f"   Detection Rate: {result['detection_rate'] * 100:.1f}%\n")
            f.write(
                f"   Species: {', '.join([f'{k} ({v * 100:.1f}%)' for k, v in result['species_percentages'].items() if v > 0])}\n")

    print_success(f"Summary report generated at {truncate_path(report_path)}")

    # Print summary to console in a nice box
    print_fancy_header("PROCESSING SUMMARY")

    # Count by file type
    video_count = sum(1 for r in results if r.get('file_type') == 'video')
    image_count = sum(1 for r in results if r.get('file_type') == 'image')

    # Create a neat summary box
    summary_lines = [
        f"Total files processed: {len(results)}",
        f"Videos: {video_count}, Images: {image_count}",
        "",
        "Classification Results:"
    ]

    # Add classification counts
    for classification, count in sorted(classifications.items()):
        percent = count / len(results) * 100
        classification_str = classification.ljust(15)
        summary_lines.append(f"  {classification_str}: {count} files ({percent:.1f}%)")

    # Create the box and center it as a whole block
    boxed_summary = draw_box('\n'.join(summary_lines), title="Results", style='double')
    print(center_text_block(boxed_summary))


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


def display_splash_screen():
    """Display a splash screen with wolf ASCII art and app title."""
    # Clear screen
    os.system('cls' if IS_WINDOWS else 'clear')

    # Get terminal width
    width = get_terminal_width()

    # Print wolf ASCII art (as a single block to preserve formatting)
    print(center_text_block(WOLF_ASCII_ART))

    # Print centered title
    print(center_text_block(TITLE_DISPLAY))

    # Add a small delay for effect
    time.sleep(0.5)


def main():
    """Main function to run the script."""
    # Initialize colors
    init_colors()

    # Display fancy splash screen
    display_splash_screen()

    # Get configuration file path
    config_path = input(
        f"{Colors.BOLD}Enter the path to the YAML configuration file (or press Enter to use default): {Colors.END}").strip()
    config_path = clean_path(config_path)  # Handle quoted paths
    if not config_path:
        config_path = CONFIG_FILE

    # Load configuration
    print_info(f"Loading configuration from: {truncate_path(config_path)}")
    config = load_config(config_path)

    # Get input/output paths
    input_folder = input(
        f"{Colors.BOLD}Enter the folder path containing videos and images to process (or press Enter to use default): {Colors.END}").strip()
    input_folder = clean_path(input_folder)  # Handle quoted paths
    if not input_folder:
        input_folder = VIDEO_PATH

    output_folder = input(
        f"{Colors.BOLD}Enter the output folder path (or press Enter to use default: {truncate_path(OUTPUT_PATH)}): {Colors.END}").strip()
    output_folder = clean_path(output_folder)  # Handle quoted paths
    if not output_folder:
        output_folder = OUTPUT_PATH

    # Get model path
    model_path = input(f"{Colors.BOLD}Enter the YOLO model path (or press Enter to use default): {Colors.END}").strip()
    model_path = clean_path(model_path)  # Handle quoted paths
    if not model_path:
        model_path = DEFAULT_MODEL_PATH

    # Show settings summary
    settings = [
        f"Input folder: {truncate_path(input_folder)}",
        f"Output folder: {truncate_path(output_folder)}",
        f"Model path: {truncate_path(model_path)}",
        f"Config file: {truncate_path(config_path)}"
    ]

    # Create and center the settings box as a whole
    settings_box = draw_box('\n'.join(settings), title="Settings", style='double')
    print(center_text_block(settings_box))

    # Confirm to proceed
    proceed = input(
        f"\n{Colors.BOLD}Ready to process videos and images. Press Enter to continue or Ctrl+C to cancel...{Colors.END}")

    # Process files
    process_all_files(input_folder, output_folder, model_path, config)

    # Final success message with fancy box
    success_message = """
    All videos and images have been processed and sorted into their respective folders.
    Check the processing_report.txt file for detailed results.
    
    Thank you for using WolfVue: Wildlife Video Classifier!
    
    Created by Nathan Bluto
    Data from The Gray Wolf Research Project
    Facilitated by Dr. Ausband
    """

    print_fancy_header("PROCESSING COMPLETE!")
    # Create and center the success box as a whole
    success_box = draw_box(success_message, style='double')
    print(center_text_block(success_box))


if __name__ == "__main__":
    main()
