#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                             WOLF FORGE – YOLO Trainer                        ║
║                    Train custom wildlife models in minutes                   ║
║                   Built for The Gray Wolf Research Project                   ║
║                            Powered by Ultralytics • 2025                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
from collections import Counter
import argparse
from datetime import datetime

# Optional: Logging & Viz
try:
    from tensorboard import program
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# CONFIG (Easy tweaks at top)
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()

DEFAULT_YAML = SCRIPT_DIR / "WlfCamData.yaml"
DEFAULT_DATASET = Path.cwd() / "dataset"  # Assumes YOLO structure: dataset/train/images etc.

# Training defaults (tune these!)
EPOCHS = 100
BATCH_SIZE = 16  # Auto-adjusts based on GPU VRAM
IMG_SIZE = 640
LR = 0.01
WEIGHT_DECAY = 0.0005
PATIENCE = 20  # Early stopping

# ─────────────────────────────────────────────────────────────
# Terminal Magic (from our previous tools)
# ─────────────────────────────────────────────────────────────
try:
    from colorama import init
    init(autoreset=True)
    class C:
        BOLD = "\033[1m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        MAGENTA = "\033[95m"
        CYAN = "\033[96m"
        WHITE = "\033[97m"
        END = "\033[0m"
except ImportError:
    class C:
        BOLD = GREEN = RED = YELLOW = BLUE = MAGENTA = CYAN = WHITE = END = ""

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def box(content: str, title: str = "") -> str:
    lines = content.split("\n")
    width = max(len(l) for l in lines) + 4
    if title: width = max(width, len(title) + 6)
    h = "═" * (width - 2)
    out = [f"╔{h}╗"]
    if title:
        pad = (width - len(title) - 2) // 2
        out[-1] = f"╔{'═'*pad} {title} {'═'*(width - len(title) - 2 - pad)}╗"
    for line in lines:
        out.append(f"║ {line:<{width-4}} ║")
    out.append(f"╚{h}╝")
    return "\n".join(out)

# ─────────────────────────────────────────────────────────────
# Dataset Validation & Prep
# ─────────────────────────────────────────────────────────────
def validate_dataset(dataset_path: Path, yaml_path: Path) -> bool:
    print(f"{C.CYAN}Validating dataset: {dataset_path}{C.END}")
    
    # Check structure
    for split in ["train", "val", "test"]:
        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"
        if not (img_dir.exists() and lbl_dir.exists()):
            print(f"{C.RED}Missing {split}/images or {split}/labels{C.END}")
            return False
    
    # Load YAML & check classes
    config = yaml.safe_load(yaml_path.read_text())
    classes = config.get("names", {})
    if not classes:
        print(f"{C.RED}No classes in YAML: {yaml_path}{C.END}")
        return False
    
    # Quick class balance check
    all_labels = Counter()
    for split in ["train", "val"]:
        lbl_dir = dataset_path / split / "labels"
        for lbl in lbl_dir.glob("*.txt"):
            with open(lbl) as f:
                for line in f:
                    if line.strip():
                        cls_id = int(line.split()[0])
                        all_labels[cls_id] += 1
    
    print(f"{C.GREEN}Dataset OK: {sum(all_labels.values())} labels across {len(all_labels)} classes{C.END}")
    if max(all_labels.values()) / min(all_labels.values() if all_labels.values() else [1]) > 10:
        print(f"{C.YELLOW}Warning: Class imbalance detected (consider augmentation){C.END}")
    
    return True

def auto_adjust_batch_size(model: YOLO) -> int:
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        if gpu_mem > 8:
            return 32
        elif gpu_mem > 4:
            return 16
        else:
            return 8
    return 16  # CPU fallback

# ─────────────────────────────────────────────────────────────
# Core Training
# ─────────────────────────────────────────────────────────────
def train_model(dataset_yaml: Path, epochs: int = EPOCHS, batch: int = BATCH_SIZE, **kwargs):
    print(f"{C.BOLD}{C.MAGENTA}{'═' * 60}{C.END}")
    print(f"{C.BOLD}{C.YELLOW}Starting YOLO training...{C.END}")
    print(f"{C.BOLD}{C.MAGENTA}{'═' * 60}{C.END}")
    
    # Load or create model
    model = YOLO("yolov8n.pt")  # Nano base – swap for yolov8s/m/l if needed
    
    # Train!
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=IMG_SIZE,
        lr0=LR,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        device=0 if torch.cuda.is_available() else "cpu",
        project="runs/train",
        name=f"wolf_{datetime.now().strftime('%Y%m%d_%H%M')}",
        **kwargs  # Plots, save, etc.
    )
    
    # Validate
    metrics = model.val()
    print(f"{C.GREEN}Validation mAP@50: {metrics.box.map50:.3f}{C.END}")
    
    # Export options
    model.export(format="onnx")  # For edge deployment
    print(f"{C.GREEN}Model exported to ONNX{C.END}")
    
    return model

# ─────────────────────────────────────────────────────────────
# Tensorboard Launcher (Bonus)
# ─────────────────────────────────────────────────────────────
def launch_tensorboard(log_dir: str = "runs"):
    if TENSORBOARD_AVAILABLE:
        tb = program.TensorBoard()
        tb.configure(bind_all=True, logdir=log_dir)
        url = tb.launch()
        print(f"{C.CYAN}TensorBoard: {url}{C.END}")
        print(f"{C.YELLOW}Training curves live at {url}#scalars&run=runs/train/wolf_*{C.END}")

# ─────────────────────────────────────────────────────────────
# CLI Magic
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="WolfForge – Train YOLO wildlife models")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to YOLO dataset")
    parser.add_argument("--yaml", type=Path, default=DEFAULT_YAML, help="Dataset config YAML")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size (auto if 0)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    
    clear_screen()
    print(f"{C.BOLD}{C.GREEN}{box('WOLF FORGE ACTIVATED', title='Train Your Model')}{C.END}")
    
    # Validate
    if not validate_dataset(args.dataset, args.yaml):
        print(f"{C.RED}Dataset validation failed. Fix and retry.{C.END}")
        sys.exit(1)
    
    # Auto-batch
    batch = auto_adjust_batch_size(None) if args.batch == 0 else args.batch
    print(f"{C.CYAN}Using batch size: {batch} (GPU: {torch.cuda.is_available()}){C.END}")
    
    # Train
    model = train_model(args.yaml, epochs=args.epochs, batch=batch)
    
    # Viz
    launch_tensorboard()
    
    print(f"\n{C.BOLD}{C.GREEN}{box('Training Complete!', title='Model Ready')}{C.END}")
    print(f"{C.YELLOW}Best model: runs/train/wolf_*/weights/best.pt{C.END}")

if __name__ == "__main__":
    main()
