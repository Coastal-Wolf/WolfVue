#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    WOLF RANK  –  Wildlife Directory Analyzer                 ║
║                  Rank camera-trap folders by species concentration           ║
║               Based on WolfVue • Built for The Gray Wolf Research Project    ║
║                            Powered by YOLOv8/v10 • 2025                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import platform
import shutil
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

import cv2
import yaml
from ultralytics import YOLO

# Optional but glorious
try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False

# ─────────────────────────────────────────────────────────────
# CONFIGURATION (Still easy to tweak)
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()

DEFAULT_CONFIG = SCRIPT_DIR / "WlfCamData.yaml"
DEFAULT_MODEL = SCRIPT_DIR / "weights" / "WolfVue_Beta1" / "best.pt"

CONF_VIDEO = 0.40
CONF_IMAGE = 0.65
SAMPLE_EVERY_N_FRAMES = 30
TOP_N_RANK = 5

# ─────────────────────────────────────────────────────────────
# Terminal Beauty Engine (Now with colorama fallback)
# ─────────────────────────────────────────────────────────────
IS_WINDOWS = platform.system() == "Windows"

try:
    from colorama import init
    init(autoreset=True)
    class C:
        BOLD = "\033[1m"
        RED = "\033[91m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        MAGENTA = "\033[95m"
        CYAN = "\033[96m"
        WHITE = "\033[97m"
        GRAY = "\033[90m"
        END = "\033[0m"
except ImportError:
    class C:
        BOLD = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = GRAY = END = ""

# ─────────────────────────────────────────────────────────────
# ASCII Majesty
# ─────────────────────────────────────────────────────────────
WOLFRANK_ASCII = r"""
██╗    ██╗ ██████╗ ██╗     ███████╗  ██████╗  █████╗ ███╗   ██╗██╗  ██╗
██║    ██║██╔═══██╗██║     ██╔════╝  ██╔══██╗██╔══██╗████╗  ██║██║ ██╔╝
██║ █╗ ██║██║   ██║██║     █████╗    ██████╔╝███████║██╔██╗ ██║█████╔╝ 
██║███╗██║██║   ██║██║     ██╔══╝    ██╔══██╗██╔══██║██║╚██╗██║██╔═██╗ 
╚███╔███╔╝╚██████╔╝███████╗██║       ██║  ██║██║  ██║██║ ╚████║██║  ██╗
 ╚══╝╚══╝  ╚═════╝ ╚══════╝╚═╝       ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝
"""

TITLE_BOX = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                      WILDLIFE DIRECTORY ANALYZER                             ║
║                                                                              ║
║            Analyze Multiple Directories for Species Concentration            ║
║                                                                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def clear_screen():
    os.system("cls" if IS_WINDOWS else "clear")

def center(text: str) -> str:
    width = shutil.get_terminal_size().columns
    return text.center(width)

def box(lines: list[str], title: str = "") -> str:
    width = max(len(line) for line in lines) + 4
    if title:
        width = max(width, len(title) + 6)
    h = "═" if not title else "═"
    out = ["╔" + h * (width - 2) + "╗"]
    if title:
        out[-1] = out[-1][:width//2 - len(title)//2 - 1] + f" {title} " + out[-1][width//2 + len(title)//2 + 2:]
    for line in lines:
        out.append("║ " + line.center(width - 4) + " ║")
    out.append("╚" + h * (width - 2) + "╝")
    return "\n".join(out)

def truncate(path: Path, max_len: int = 60) -> str:
    s = str(path)
    if len(s) <= max_len:
        return s
    return "..." + s[-(max_len-3):]

# ─────────────────────────────────────────────────────────────
# Core Analysis
# ─────────────────────────────────────────────────────────────
def load_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        print(f"{C.RED}YAML load failed: {e}{C.END}")
        sys.exit(1)

def load_model(path: Path) -> YOLO:
    print(f"{C.CYAN}Loading model: {path}{C.END}")
    return YOLO(str(path))

def get_media_files(folder: Path):
    exts = {"*.mp4", "*.avi", "*.mov", "*.mkv", "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"}
    files = []
    for ext in exts:
        files.extend(folder.rglob(ext))
        files.extend(folder.rglob(ext.upper()))
    return sorted(set(files))

def analyze_file(path: Path, model: YOLO, names: dict):
    counts = defaultdict(int)
    ext = path.suffix.lower()

    try:
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}:
            img = cv2.imread(str(path))
            if img is None: return counts
            results = model(img, verbose=False)
            for r in results:
                for box in r.boxes:
                    if box.conf.item() >= CONF_IMAGE:
                        counts[names[int(box.cls.item())]] += 1

        else:  # video
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened(): return counts
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame_idx += 1
                if frame_idx % SAMPLE_EVERY_N_FRAMES == 0:
                    results = model(frame, verbose=False)
                    for r in results:
                        for box in r.boxes:
                            if box.conf.item() >= CONF_VIDEO:
                                counts[names[int(box.cls.item())]] += 1
            cap.release()
    except Exception as e:
        print(f"{C.YELLOW}Warning: Failed {path.name}: {e}{C.END}")

    return counts

def analyze_directory(folder: Path, model: YOLO, names: dict) -> dict:
    print(f"\n{C.BOLD}{C.BLUE}Analyzing:{C.END} {truncate(folder)}")
    files = get_media_files(folder)
    if not files:
        print(f"{C.YELLOW}No media found{C.END}")
        return {}

    print(f"{C.CYAN}Found {len(files)} files ({files.count(lambda f: f.suffix.lower() in ['.mp4','.avi','.mov','.mkv'])} videos){C.END}")

    total_counts = defaultdict(int)
    iterator = tqdm(files, desc="Processing", unit="file") if TQDM else files

    for file in iterator:
        if not TQDM:
            if (files.index(file) + 1) % 10 == 0 or file == files[-1]:
                print(f"\r{C.GRAY}Processed {files.index(file)+1}/{len(files)}{C.END}", end="", flush=True)
        counts = analyze_file(file, model, names)
        for k, v in counts.items():
            total_counts[k] += v

    if not TQDM: print()
    print(f"{C.GREEN}Success: {sum(total_counts.values())} detections, {len(total_counts)} species{C.END}")
    return dict(total_counts)

# ─────────────────────────────────────────────────────────────
# Reports
# ─────────────────────────────────────────────────────────────
def print_species_ranking(results: dict, names: dict):
    print(f"\n{C.BOLD}{C.MAGENTA}{"═" * 80}{C.END}")
    print(center(f"{C.BOLD}{C.YELLOW}SPECIES CONCENTRATION RANKINGS{C.END}"))
    print(f"{C.BOLD}{C.MAGENTA}{"═" * 80}{C.END}\n")

    all_species = sorted({s for counts in results.values() for s in counts})
    for species in all_species:
        ranked = sorted(
            [(p, c[species]) for p, c in results.items() if species in c],
            key=lambda x: x[1], reverse=True
        )
        print(f"{C.BOLD}{C.CYAN}Top {species}{C.END}")
        for i, (path, count) in enumerate(ranked[:TOP_N_RANK], 1):
            medal = ["", "1st", "2nd", "3rd", "4th"][i] if i <= 3 else f"#{i}"
            color = [C.GREEN, C.YELLOW, C.MAGENTA, C.WHITE, C.GRAY][min(i-1, 4)]
            print(f"  {color}{medal:>3} {path.name:<40} → {count:>6} detections{C.END}")
        if len(ranked) > TOP_N_RANK:
            print(f"  {C.GRAY}... and {len(ranked) - TOP_N_RANK} more{C.END}")
        print()

def save_report(results: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WOLF RANK – Wildlife Directory Analysis Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for folder, counts in results.items():
            f.write(f"{folder}\n")
            f.write(f"  Total detections: {sum(counts.values())}\n")
            for species, count in sorted(counts.items(), key=lambda x: -x[1]):
                pct = count / sum(counts.values()) * 100
                f.write(f"  • {species}: {count} ({pct:.1f}%)\n")
            f.write("\n")
    print(f"{C.GREEN}Report saved: {path}{C.END}")

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    clear_screen()
    print(f"{C.GREEN}{center(WOLFRANK_ASCII)}{C.END}")
    print(f"{C.CYAN}{center(TITLE_BOX)}{C.END}")
    time.sleep(1.5)

    # Config & Model
    cfg_path = Path(input(f"{C.BOLD}YAML path [{DEFAULT_CONFIG}]: {C.END}") or DEFAULT_CONFIG)
    model_path = Path(input(f"{C.BOLD}Model path [{DEFAULT_MODEL}]: {C.END}") or DEFAULT_MODEL)

    config = load_yaml(cfg_path)
    names = {int(k): v for k, v in config.get("names", {}).items()}
    model = load_model(model_path)

    # Directories
    print(f"\n{C.BOLD}{C.BLUE}Enter folders to analyze (drag & drop or paste). Empty line = done.{C.END}")
    folders = []
    while True:
        line = input(f"{C.WHITE}> {C.END}").strip().strip('"\'')
        if not line: break
        p = Path(line)
        if p.is_dir():
            folders.append(p.resolve())
            print(f"{C.GREEN}Added: {p.name}{C.END}")
        else:
            print(f"{C.RED}Not found: {p}{C.END}")

    if not folders:
        print(f"{C.RED}No folders selected. Exiting.{C.END}")
        return

    # Analyze
    results = {}
    start = time.time()
    for i, folder in enumerate(folders, 1):
        print(f"\n{C.BOLD}{C.MAGENTA}{'═' * 80}{C.END}")
        print(center(f"{C.WHITE}FOLDER {i}/{len(folders)} – {folder.name}{C.END}"))
        print(f"{C.BOLD}{C.MAGENTA}{'═' * 80}{C.END}")
        results[folder] = analyze_directory(folder, model, names)

    # Final Report
    total_time = int(time.time() - start)
    print_species_ranking(results, names)
    report_file = Path.cwd() / f"WolfRank_Report_{int(time.time())}.txt"
    save_report(results, report_file)

    summary = [
        f"Directories analyzed: {len(folders)}",
        f"With wildlife: {sum(1 for c in results.values() if c)}",
        f"Total detections: {sum(sum(c.values()) for c in results.values()):,}",
        f"Total time: {str(timedelta(seconds=total_time))}",
        "",
        f"Full report → {report_file.name}",
    ]
    print(f"\n{C.GREEN}{box(summary, title='WOLF RANK COMPLETE')}{C.END}")

if __name__ == "__main__":
    main()
