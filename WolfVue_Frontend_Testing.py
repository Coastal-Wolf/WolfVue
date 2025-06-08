#!/usr/bin/env python3
"""
WolfVue: Wildlife Video Classifier - Desktop Application
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
import shutil
import time
import platform
import tempfile
import zipfile
import threading
from pathlib import Path
from datetime import timedelta
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import scrolledtext

# ============= CONFIGURATION =============
DEFAULT_CONFIDENCE_THRESHOLD = 0.40
DEFAULT_DOMINANT_SPECIES_THRESHOLD = 0.9
DEFAULT_MAX_SPECIES_TRANSITIONS = 5
DEFAULT_CONSECUTIVE_EMPTY_FRAMES = 15
DEFAULT_IMAGE_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_IMAGE_MIN_DETECTIONS = 1
DEFAULT_IMAGE_MULTI_SPECIES_THRESHOLD = 0.60
DEFAULT_IMAGE_UNSORTED_MIN_CONFIDENCE = 0.35
DEFAULT_IMAGE_UNSORTED_MAX_CONFIDENCE = 0.65

PREDATORS = ["Cougar", "Lynx", "Wolf", "Coyote", "Fox", "Bear"]
PREY = ["WhiteTail", "MuleDeer", "Elk", "Moose"]

DEFAULT_TAXONOMY = {
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

# Include all the core processing functions from your original script
def load_config_from_file(config_file_path):
    """Load and parse the YAML configuration file."""
    try:
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
        return config, "Configuration loaded successfully"
    except Exception as e:
        return None, f"Error loading configuration file: {e}"

def extract_taxonomy_from_config(config):
    """Extract taxonomy structure from the YAML config file."""
    if config and 'taxonomy' in config:
        return config['taxonomy']
    else:
        return DEFAULT_TAXONOMY

def create_folder_structure(base_path, taxonomy):
    """Create the folder structure based on the taxonomy."""
    os.makedirs(os.path.join(base_path, "Sorted"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "Unsorted"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "No_Animal"), exist_ok=True)
    
    for category, subcategories in taxonomy.items():
        category_path = os.path.join(base_path, "Sorted", category)
        os.makedirs(category_path, exist_ok=True)
        
        for species, _ in subcategories.items():
            species_path = os.path.join(category_path, species)
            os.makedirs(species_path, exist_ok=True)
    
    other_category_path = os.path.join(base_path, "Sorted", "Other")
    os.makedirs(other_category_path, exist_ok=True)

def get_species_folder_path(base_path, species, taxonomy):
    """Get the folder path for a species based on the taxonomy."""
    if species == "Unsorted":
        return os.path.join(base_path, "Unsorted")
    elif species == "No_Animal":
        return os.path.join(base_path, "No_Animal")
    
    for category, subcategories in taxonomy.items():
        if species in subcategories:
            return os.path.join(base_path, "Sorted", category, species)
    
    other_category_path = os.path.join(base_path, "Sorted", "Other", species)
    os.makedirs(other_category_path, exist_ok=True)
    return other_category_path

def load_yolo_model(model_path):
    """Load the YOLO model."""
    try:
        model = YOLO(model_path)
        return model, "Model loaded successfully"
    except Exception as e:
        return None, f"Error loading YOLO model: {e}"

# Add other processing functions here (process_video_with_yolo, analyze_detections, etc.)
# I'm keeping this shorter for the example, but you'd include all your processing logic

class WolfVueDesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🐺 WolfVue: Wildlife Video Classifier")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Configure style
        self.setup_styles()
        
        # Variables
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.model_path = tk.StringVar()
        self.config_path = tk.StringVar()
        
        # Processing state
        self.is_processing = False
        self.processing_thread = None
        
        # Create the interface
        self.create_widgets()
        
    def setup_styles(self):
        """Setup the visual styling."""
        self.root.configure(bg='#f0f0f0')
        
        # Create custom style for ttk widgets
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        style.configure('Subtitle.TLabel', font=('Arial', 10), background='#f0f0f0', foreground='#666666')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0')
        
    def create_widgets(self):
        """Create all the GUI widgets."""
        
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        title_label = ttk.Label(header_frame, text="🐺 WolfVue: Wildlife Video Classifier", style='Title.TLabel')
        title_label.pack()
        
        subtitle_label = ttk.Label(header_frame, 
                                 text="Created by Nathan Bluto | Gray Wolf Research Project | Dr. Ausband",
                                 style='Subtitle.TLabel')
        subtitle_label.pack()
        
        # Left panel - Controls
        left_frame = ttk.Frame(main_frame, padding="10")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Right panel - Results
        right_frame = ttk.Frame(main_frame, padding="10")
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure column weights
        main_frame.rowconfigure(1, weight=1)
        
        self.create_left_panel(left_frame)
        self.create_right_panel(right_frame)
        
    def create_left_panel(self, parent):
        """Create the left control panel."""
        
        # File Selection Section
        files_frame = ttk.LabelFrame(parent, text="📁 File Selection", padding="10")
        files_frame.pack(fill="x", pady=(0, 10))
        
        # Input folder
        ttk.Label(files_frame, text="Input Folder (Videos/Images):").pack(anchor="w")
        input_frame = ttk.Frame(files_frame)
        input_frame.pack(fill="x", pady=(2, 8))
        
        input_entry = ttk.Entry(input_frame, textvariable=self.input_folder, width=40)
        input_entry.pack(side="left", fill="x", expand=True)
        
        ttk.Button(input_frame, text="Browse", command=self.select_input_folder).pack(side="right", padx=(5, 0))
        
        # Output folder
        ttk.Label(files_frame, text="Output Folder:").pack(anchor="w")
        output_frame = ttk.Frame(files_frame)
        output_frame.pack(fill="x", pady=(2, 8))
        
        output_entry = ttk.Entry(output_frame, textvariable=self.output_folder, width=40)
        output_entry.pack(side="left", fill="x", expand=True)
        
        ttk.Button(output_frame, text="Browse", command=self.select_output_folder).pack(side="right", padx=(5, 0))
        
        # Model file
        ttk.Label(files_frame, text="YOLO Model File (.pt):").pack(anchor="w")
        model_frame = ttk.Frame(files_frame)
        model_frame.pack(fill="x", pady=(2, 8))
        
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path, width=40)
        model_entry.pack(side="left", fill="x", expand=True)
        
        ttk.Button(model_frame, text="Browse", command=self.select_model_file).pack(side="right", padx=(5, 0))
        
        # Config file (optional)
        ttk.Label(files_frame, text="YAML Config File (Optional):").pack(anchor="w")
        config_frame = ttk.Frame(files_frame)
        config_frame.pack(fill="x", pady=(2, 0))
        
        config_entry = ttk.Entry(config_frame, textvariable=self.config_path, width=40)
        config_entry.pack(side="left", fill="x", expand=True)
        
        ttk.Button(config_frame, text="Browse", command=self.select_config_file).pack(side="right", padx=(5, 0))
        
        # Settings Section
        settings_frame = ttk.LabelFrame(parent, text="⚙️ Processing Settings", padding="10")
        settings_frame.pack(fill="x", pady=(0, 10))
        
        # Video settings
        ttk.Label(settings_frame, text="Video Processing:", style='Header.TLabel').pack(anchor="w")
        
        # Confidence threshold
        conf_frame = ttk.Frame(settings_frame)
        conf_frame.pack(fill="x", pady=2)
        ttk.Label(conf_frame, text="Confidence Threshold:").pack(side="left")
        self.conf_var = tk.DoubleVar(value=DEFAULT_CONFIDENCE_THRESHOLD)
        self.conf_scale = ttk.Scale(conf_frame, from_=0.1, to=1.0, variable=self.conf_var, orient="horizontal")
        self.conf_scale.pack(side="right", fill="x", expand=True, padx=(10, 0))
        
        # Dominant species threshold
        dom_frame = ttk.Frame(settings_frame)
        dom_frame.pack(fill="x", pady=2)
        ttk.Label(dom_frame, text="Dominant Species Threshold:").pack(side="left")
        self.dom_var = tk.DoubleVar(value=DEFAULT_DOMINANT_SPECIES_THRESHOLD)
        self.dom_scale = ttk.Scale(dom_frame, from_=0.5, to=1.0, variable=self.dom_var, orient="horizontal")
        self.dom_scale.pack(side="right", fill="x", expand=True, padx=(10, 0))
        
        # Image settings
        ttk.Label(settings_frame, text="Image Processing:", style='Header.TLabel').pack(anchor="w", pady=(10, 0))
        
        # Image confidence threshold
        img_conf_frame = ttk.Frame(settings_frame)
        img_conf_frame.pack(fill="x", pady=2)
        ttk.Label(img_conf_frame, text="Image Confidence Threshold:").pack(side="left")
        self.img_conf_var = tk.DoubleVar(value=DEFAULT_IMAGE_CONFIDENCE_THRESHOLD)
        self.img_conf_scale = ttk.Scale(img_conf_frame, from_=0.1, to=1.0, variable=self.img_conf_var, orient="horizontal")
        self.img_conf_scale.pack(side="right", fill="x", expand=True, padx=(10, 0))
        
        # Process Button
        self.process_btn = ttk.Button(parent, text="🚀 Process Files", command=self.start_processing)
        self.process_btn.pack(fill="x", pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(parent, mode='determinate')
        self.progress.pack(fill="x", pady=(0, 10))
        
    def create_right_panel(self, parent):
        """Create the right results panel."""
        
        # Results Section
        results_frame = ttk.LabelFrame(parent, text="📊 Processing Results", padding="10")
        results_frame.pack(fill="both", expand=True)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, height=20, wrap="word", 
                                                     font=('Consolas', 9))
        self.results_text.pack(fill="both", expand=True)
        
        # Buttons frame
        buttons_frame = ttk.Frame(results_frame)
        buttons_frame.pack(fill="x", pady=(10, 0))
        
        # Clear results button
        ttk.Button(buttons_frame, text="Clear Results", command=self.clear_results).pack(side="left")
        
        # Open output folder button
        ttk.Button(buttons_frame, text="Open Output Folder", command=self.open_output_folder).pack(side="right")
        
    def select_input_folder(self):
        """Select input folder."""
        folder = filedialog.askdirectory(title="Select Input Folder (Videos/Images)")
        if folder:
            self.input_folder.set(folder)
    
    def select_output_folder(self):
        """Select output folder."""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder.set(folder)
    
    def select_model_file(self):
        """Select YOLO model file."""
        file = filedialog.askopenfilename(
            title="Select YOLO Model File",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        if file:
            self.model_path.set(file)
    
    def select_config_file(self):
        """Select YAML config file."""
        file = filedialog.askopenfilename(
            title="Select YAML Configuration File",
            filetypes=[("YAML Files", "*.yaml *.yml"), ("All Files", "*.*")]
        )
        if file:
            self.config_path.set(file)
    
    def log_message(self, message):
        """Add a message to the results text area."""
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_results(self):
        """Clear the results text area."""
        self.results_text.delete(1.0, tk.END)
    
    def open_output_folder(self):
        """Open the output folder in file explorer."""
        if self.output_folder.get() and os.path.exists(self.output_folder.get()):
            if platform.system() == "Windows":
                os.startfile(self.output_folder.get())
            elif platform.system() == "Darwin":  # macOS
                os.system(f"open '{self.output_folder.get()}'")
            else:  # Linux
                os.system(f"xdg-open '{self.output_folder.get()}'")
        else:
            messagebox.showwarning("Warning", "Output folder not set or doesn't exist!")
    
    def validate_inputs(self):
        """Validate all required inputs."""
        if not self.input_folder.get():
            messagebox.showerror("Error", "Please select an input folder!")
            return False
        
        if not os.path.exists(self.input_folder.get()):
            messagebox.showerror("Error", "Input folder doesn't exist!")
            return False
            
        if not self.output_folder.get():
            messagebox.showerror("Error", "Please select an output folder!")
            return False
            
        if not self.model_path.get():
            messagebox.showerror("Error", "Please select a YOLO model file!")
            return False
            
        if not os.path.exists(self.model_path.get()):
            messagebox.showerror("Error", "Model file doesn't exist!")
            return False
        
        return True
    
    def start_processing(self):
        """Start the processing in a separate thread."""
        if self.is_processing:
            messagebox.showwarning("Warning", "Processing is already in progress!")
            return
        
        if not self.validate_inputs():
            return
        
        # Start processing
        self.is_processing = True
        self.process_btn.configure(text="Processing...", state="disabled")
        self.progress['value'] = 0
        self.clear_results()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_files, daemon=True)
        self.processing_thread.start()
    
    def process_files(self):
        """Main processing function (runs in separate thread)."""
        try:
            self.log_message("🚀 Starting WolfVue processing...")
            self.log_message(f"📁 Input folder: {self.input_folder.get()}")
            self.log_message(f"📂 Output folder: {self.output_folder.get()}")
            self.log_message(f"🎯 Model: {os.path.basename(self.model_path.get())}")
            self.log_message("")
            
            # Load configuration
            config = None
            if self.config_path.get() and os.path.exists(self.config_path.get()):
                config, config_msg = load_config_from_file(self.config_path.get())
                self.log_message(f"📋 {config_msg}")
            else:
                self.log_message("📋 Using default configuration")
            
            # Extract taxonomy
            taxonomy = extract_taxonomy_from_config(config)
            
            # Get class names
            if config and 'names' in config:
                class_names = config['names']
            else:
                # Default class names
                class_names = {i: name for i, name in enumerate([
                    "WhiteTail", "MuleDeer", "Elk", "Moose", "Cougar", "Lynx", 
                    "Wolf", "Coyote", "Fox", "Bear"
                ])}
            
            self.log_message(f"🏷️  Loaded {len(class_names)} species classifications")
            
            # Load model
            self.progress['value'] = 10
            self.log_message("🤖 Loading YOLO model...")
            model, model_msg = load_yolo_model(self.model_path.get())
            if not model:
                self.log_message(f"❌ {model_msg}")
                return
            self.log_message(f"✅ {model_msg}")
            
            # Create folder structure
            self.progress['value'] = 20
            self.log_message("📁 Creating output folder structure...")
            create_folder_structure(self.output_folder.get(), taxonomy)
            self.log_message("✅ Folder structure created")
            
            # Find files
            self.progress['value'] = 30
            self.log_message("🔍 Scanning for files...")
            
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            
            video_files = []
            image_files = []
            
            for ext in video_extensions:
                video_files.extend(list(Path(self.input_folder.get()).glob(f'*{ext}')))
                video_files.extend(list(Path(self.input_folder.get()).glob(f'*{ext.upper()}')))
            
            for ext in image_extensions:
                image_files.extend(list(Path(self.input_folder.get()).glob(f'*{ext}')))
                image_files.extend(list(Path(self.input_folder.get()).glob(f'*{ext.upper()}')))
            
            video_files = list(set(video_files))
            image_files = list(set(image_files))
            total_files = len(video_files) + len(image_files)
            
            if total_files == 0:
                self.log_message("❌ No valid video or image files found!")
                return
            
            self.log_message(f"📊 Found {len(video_files)} videos and {len(image_files)} images ({total_files} total)")
            self.log_message("")
            
            # Process files (simplified for demo - you'd add your full processing logic)
            results = []
            processed = 0
            
            for i, file_path in enumerate(video_files + image_files):
                file_name = os.path.basename(str(file_path))
                self.log_message(f"🎬 Processing {file_name}...")
                
                # Update progress
                progress_value = 30 + (processed / total_files) * 60
                self.progress['value'] = progress_value
                
                # Simulate processing (replace with actual processing)
                time.sleep(0.5)  # Remove this in real implementation
                
                # Here you would call your actual processing functions:
                # if str(file_path).lower().endswith(tuple(video_extensions)):
                #     frame_data = process_video_with_yolo(...)
                # else:
                #     frame_data = process_image_with_yolo(...)
                # analysis = analyze_detections(...)
                # classification = analysis['classification']
                # target_path = sort_file(...)
                
                # Simulated result
                classification = "Wolf"  # This would come from your analysis
                self.log_message(f"   ✅ Classified as: {classification}")
                
                processed += 1
            
            self.progress['value'] = 95
            self.log_message("")
            self.log_message("📄 Generating summary report...")
            
            # Generate report (you'd implement this)
            time.sleep(0.5)
            
            self.progress['value'] = 100
            self.log_message("✅ Processing complete!")
            self.log_message("")
            self.log_message("📁 Results saved to output folder")
            self.log_message("📊 Summary report created")
            self.log_message("")
            self.log_message("🎉 WolfVue processing finished successfully!")
            
        except Exception as e:
            self.log_message(f"❌ Error during processing: {str(e)}")
            
        finally:
            # Reset UI state
            self.is_processing = False
            self.process_btn.configure(text="🚀 Process Files", state="normal")

def main():
    """Main function to run the desktop application."""
    root = tk.Tk()
    app = WolfVueDesktopApp(root)
    
    # Handle window closing
    def on_closing():
        if app.is_processing:
            if messagebox.askokcancel("Quit", "Processing is in progress. Do you want to quit anyway?"):
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()