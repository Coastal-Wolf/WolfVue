import os
import cv2
import yaml
import random
import string
import subprocess
import sys
import shutil
from pathlib import Path
from collections import defaultdict, Counter

class TrailCamProcessor:
    def __init__(self):
        self.default_yaml_path = r"C:\Users\Coastal_wolf\Documents\GitHub\TrailCamAi\Datasets\Scripts\WlfCamData.yaml"
        self.species_names = {}
        self.memory_file = "trailcam_memory.txt"
        self.last_directory = self.load_last_directory()
        self.load_yaml()
    
    def save_last_directory(self, directory):
        """Save the last used directory to memory file"""
        try:
            with open(self.memory_file, 'w') as f:
                f.write(directory)
            self.last_directory = directory
        except Exception:
            pass
    
    def load_last_directory(self):
        """Load the last used directory from memory file"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    return f.read().strip()
        except Exception:
            pass
        return None
    
    def load_yaml(self, yaml_path=None):
        """Load species names from YAML file"""
        if yaml_path is None:
            yaml_path = self.default_yaml_path
        
        try:
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)
                # Handle both dict and list formats, ensure integer keys
                names_data = data.get('names', {})
                if isinstance(names_data, dict):
                    # Convert string keys to integers
                    self.species_names = {int(k): v for k, v in names_data.items()}
                elif isinstance(names_data, list):
                    # Convert list to dict with indices
                    self.species_names = {i: name for i, name in enumerate(names_data)}
                else:
                    self.species_names = {}
            print(f"✓ Loaded {len(self.species_names)} species from YAML")
            
            # Show loaded species for verification
            if self.species_names:
                print("📋 Available species:")
                for class_id, name in sorted(self.species_names.items()):
                    print(f"   {class_id}: {name}")
        except Exception as e:
            print(f"✗ Error loading YAML: {e}")
            self.species_names = {}
    
    def normalize_input(self, user_input):
        """Normalize user input - remove quotes, parentheses, extra spaces"""
        if not user_input:
            return ""
        
        # Remove quotes and parentheses manually
        cleaned = user_input.strip()
        
        # Remove leading/trailing quotes
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        elif cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1]
        
        # Remove parentheses
        cleaned = cleaned.replace('(', '').replace(')', '')
        
        # Normalize spaces
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
    
    def get_file_path(self, prompt_text, default_path=None):
        """Get file path with normalization and memory"""
        print(f"\n{prompt_text}")
        
        # Show default path
        if default_path:
            print(f"(Press Enter for default: {default_path})")
        elif self.last_directory:
            print(f"(Press Enter for last used: {self.last_directory})")
        
        user_input = input("➤ ").strip()
        
        # Use appropriate default
        if not user_input:
            if default_path:
                return default_path
            elif self.last_directory:
                return self.last_directory
            else:
                return ""
        
        # Normalize and save the new directory
        cleaned_path = self.normalize_input(user_input)
        if os.path.isdir(cleaned_path):
            self.save_last_directory(cleaned_path)
        elif os.path.isfile(cleaned_path):
            self.save_last_directory(os.path.dirname(cleaned_path))
        
        return cleaned_path
    
    def comprehensive_dataset_analysis(self, directory):
        """Comprehensive analysis that detects dataset structure and provides appropriate analysis"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE DATASET ANALYSIS")
        print("="*60)
        
        dataset_path = Path(directory)
        if not dataset_path.exists():
            print(f"❌ Directory not found: {directory}")
            return False
        
        print(f"🔍 Analyzing: {directory}")
        
        # Check for YOLO training dataset structure (train/val/test)
        splits = ['train', 'val', 'test']
        valid_splits = []
        
        for split in splits:
            split_path = dataset_path / split
            labels_path = split_path / "labels"
            images_path = split_path / "images"
            
            if split_path.exists() and labels_path.exists() and images_path.exists():
                valid_splits.append(split)
        
        # Check for simple structure (images/ and labels/ or images with labels/ subfolder)
        simple_images_dir = dataset_path / "images"
        simple_labels_dir = dataset_path / "labels"
        root_labels_dir = dataset_path / "labels"
        
        has_simple_structure = False
        if simple_images_dir.exists() and simple_labels_dir.exists():
            has_simple_structure = True
            structure_type = "YOLO Simple (images/ + labels/)"
            images_dir_to_analyze = simple_images_dir
            labels_dir_to_analyze = simple_labels_dir
        elif root_labels_dir.exists():
            # Images in root, labels in subfolder
            has_simple_structure = True
            structure_type = "Mixed (root images + labels/)"
            images_dir_to_analyze = dataset_path
            labels_dir_to_analyze = root_labels_dir
        
        # Determine analysis type
        if len(valid_splits) >= 2:  # At least train and one other split
            print(f"✅ Detected: TRAINING DATASET structure")
            print(f"   📂 Found splits: {', '.join(valid_splits)}")
            print(f"   🎯 Analysis: Species distribution across train/val/test")
            return self._analyze_training_dataset(dataset_path, valid_splits)
            
        elif has_simple_structure:
            print(f"✅ Detected: {structure_type}")
            print(f"   📂 Images: {images_dir_to_analyze}")
            print(f"   📂 Labels: {labels_dir_to_analyze}")
            print(f"   🎯 Analysis: Image counting + species breakdown")
            return self._analyze_simple_dataset(images_dir_to_analyze, labels_dir_to_analyze)
            
        else:
            print(f"❌ Unrecognized dataset structure!")
            print(f"   Expected structures:")
            print(f"   1. Training dataset:")
            print(f"      dataset/train/images/ + dataset/train/labels/")
            print(f"      dataset/val/images/ + dataset/val/labels/")
            print(f"      dataset/test/images/ + dataset/test/labels/")
            print(f"   2. Simple dataset:")
            print(f"      dataset/images/ + dataset/labels/")
            print(f"   3. Root dataset:")
            print(f"      dataset/*.jpg + dataset/labels/")
            return False
    
    def _analyze_training_dataset(self, dataset_path, valid_splits):
        """Analyze training dataset with species distribution"""
        if not self.species_names:
            print("⚠️  No species classes loaded!")
            load_yaml = input("Load YAML file first? (y/n): ").strip().lower()
            if load_yaml in ['y', 'yes']:
                self.load_yaml_menu()
                if not self.species_names:
                    print("❌ Cannot proceed without species classes")
                    return False
            else:
                print("❌ Cannot proceed without species classes")
                return False
        
        print(f"✓ Using {len(self.species_names)} species classes from YAML")
        
        # Initialize counters
        split_stats = {}
        all_class_ids = set()
        
        # Analyze each split
        for split in valid_splits:
            print(f"\n🔍 Analyzing {split} split...")
            labels_dir = dataset_path / split / "labels"
            images_dir = dataset_path / split / "images"
            
            # Count images
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            total_images = 0
            for ext in image_extensions:
                total_images += len(list(images_dir.glob(f"*{ext}")))
                total_images += len(list(images_dir.glob(f"*{ext.upper()}")))
            
            # Get all annotation files
            annotation_files = []
            system_files = {"predefined_classes.txt", "classes.txt", "obj.names", "obj.data", "dataset.yaml"}
            
            for txt_file in labels_dir.glob("*.txt"):
                if txt_file.name.lower() not in system_files:
                    annotation_files.append(txt_file)
            
            print(f"   📸 Images: {total_images}")
            print(f"   📄 Annotation files: {len(annotation_files)}")
            
            # Count class occurrences
            class_counts = Counter()
            total_annotations = 0
            valid_files = 0
            
            for ann_file in annotation_files:
                try:
                    file_has_annotations = False
                    with open(ann_file, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if line and not line.startswith('#'):
                                parts = line.split()
                                if len(parts) >= 5:  # YOLO format: class x y w h
                                    try:
                                        class_id = int(float(parts[0]))
                                        
                                        # Validate coordinates (should be 0-1)
                                        x, y, w, h = map(float, parts[1:5])
                                        if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                                            class_counts[class_id] += 1
                                            all_class_ids.add(class_id)
                                            total_annotations += 1
                                            file_has_annotations = True
                                        else:
                                            print(f"      ⚠️  Invalid coordinates in {ann_file.name}:{line_num}")
                                    except (ValueError, IndexError):
                                        print(f"      ⚠️  Invalid format in {ann_file.name}:{line_num}: {line}")
                    
                    if file_has_annotations:
                        valid_files += 1
                        
                except Exception as e:
                    print(f"      ❌ Error reading {ann_file.name}: {e}")
            
            split_stats[split] = {
                'class_counts': dict(class_counts),
                'total_annotations': total_annotations,
                'valid_files': valid_files,
                'total_files': len(annotation_files),
                'total_images': total_images
            }
            
            print(f"   📊 Results: {total_annotations} annotations in {valid_files} files")
        
        # Display comprehensive results
        self._display_training_dataset_results(split_stats, all_class_ids, valid_splits, dataset_path)
        return True
    
    def _analyze_simple_dataset(self, images_dir, labels_dir):
        """Analyze simple dataset structure with image counting and species breakdown"""
        print(f"\n🔍 Analyzing simple dataset structure...")
        
        # Count images by extension
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        extension_counts = {}
        total_images = 0
        
        print(f"   📸 Counting images in: {images_dir}")
        
        for ext in image_extensions:
            count_lower = len(list(Path(images_dir).glob(f"*{ext}")))
            count_upper = len(list(Path(images_dir).glob(f"*{ext.upper()}")))
            total_count = count_lower + count_upper
            
            if total_count > 0:
                extension_counts[ext.upper()] = total_count
                total_images += total_count
        
        # Count and analyze annotations
        annotation_count = 0
        class_counts = Counter()
        species_breakdown = {}
        
        if labels_dir.exists():
            print(f"   📄 Analyzing annotations in: {labels_dir}")
            
            system_files = {"predefined_classes.txt", "classes.txt", "obj.names", "obj.data"}
            annotation_files = [f for f in labels_dir.glob("*.txt") 
                              if f.name.lower() not in system_files]
            annotation_count = len(annotation_files)
            
            # If we have species names loaded, do species analysis
            if self.species_names and annotation_files:
                print(f"   🦌 Performing species analysis...")
                total_annotations = 0
                
                for ann_file in annotation_files:
                    try:
                        with open(ann_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    parts = line.split()
                                    if len(parts) >= 5:
                                        try:
                                            class_id = int(float(parts[0]))
                                            class_counts[class_id] += 1
                                            total_annotations += 1
                                        except (ValueError, IndexError):
                                            continue
                    except Exception:
                        continue
                
                # Create species breakdown
                for class_id, count in class_counts.items():
                    species_name = self.species_names.get(class_id, f"Unknown_{class_id}")
                    species_breakdown[species_name] = count
        
        # Display results
        self._display_simple_dataset_results(images_dir, labels_dir, total_images, 
                                            extension_counts, annotation_count, 
                                            species_breakdown, dict(class_counts))
        return True
    
    def _display_training_dataset_results(self, split_stats, all_class_ids, valid_splits, dataset_dir):
        """Display detailed training dataset results"""
        print(f"\n" + "="*80)
        print("🎯 TRAINING DATASET ANALYSIS RESULTS")
        print("="*80)
        
        print(f"📁 Dataset: {dataset_dir}")
        print(f"📋 Splits analyzed: {', '.join(valid_splits)}")
        print(f"🦌 Species found: {len(all_class_ids)}")
        
        # Overall statistics
        total_annotations_all = sum(stats['total_annotations'] for stats in split_stats.values())
        total_files_all = sum(stats['valid_files'] for stats in split_stats.values())
        total_images_all = sum(stats['total_images'] for stats in split_stats.values())
        
        print(f"📸 Total images: {total_images_all}")
        print(f"📄 Total annotation files: {total_files_all}")
        print(f"🏷️  Total annotations: {total_annotations_all}")
        
        if total_annotations_all == 0:
            print("❌ No valid annotations found in dataset!")
            return
        
        # Split overview
        print(f"\n📊 SPLIT OVERVIEW:")
        print(f"{'Split':<8} {'Images':<8} {'Files':<8} {'Annotations':<12} {'Percentage':<12}")
        print("-" * 55)
        
        for split in valid_splits:
            stats = split_stats[split]
            images = stats['total_images']
            files = stats['valid_files']
            annotations = stats['total_annotations']
            percentage = (annotations / total_annotations_all) * 100 if total_annotations_all > 0 else 0
            print(f"{split:<8} {images:<8} {files:<8} {annotations:<12} {percentage:<12.1f}%")
        
        # Species-by-species breakdown
        print(f"\n🦌 SPECIES DISTRIBUTION BREAKDOWN:")
        print("="*80)
        
        # Create header
        header = f"{'Species':<20} {'ID':<4}"
        for split in valid_splits:
            header += f"{split.title():<10}"
        header += f"{'Total':<8} {'%':<8}"
        print(header)
        print("-" * len(header))
        
        # Sort species by total count (descending)
        species_totals = {}
        for class_id in all_class_ids:
            total = sum(split_stats[split]['class_counts'].get(class_id, 0) for split in valid_splits)
            species_totals[class_id] = total
        
        sorted_species = sorted(species_totals.items(), key=lambda x: x[1], reverse=True)
        
        for class_id, total_count in sorted_species:
            species_name = self.species_names.get(class_id, f"Unknown_{class_id}")
            
            # Truncate long species names
            display_name = species_name[:19] if len(species_name) > 19 else species_name
            
            row = f"{display_name:<20} {class_id:<4}"
            
            # Add counts for each split
            for split in valid_splits:
                count = split_stats[split]['class_counts'].get(class_id, 0)
                row += f"{count:<10}"
            
            # Add total and percentage
            percentage = (total_count / total_annotations_all) * 100 if total_annotations_all > 0 else 0
            row += f"{total_count:<8} {percentage:<8.1f}%"
            print(row)
        
        # Balance analysis
        print(f"\n⚖️  DATASET BALANCE ANALYSIS:")
        print("-" * 40)
        
        if len(sorted_species) > 1:
            max_count = sorted_species[0][1]
            min_count = sorted_species[-1][1]
            balance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            print(f"Most common species: {max_count} annotations")
            print(f"Least common species: {min_count} annotations")
            print(f"Balance ratio: {balance_ratio:.1f}:1")
            
            if balance_ratio > 10:
                print("⚠️  WARNING: Dataset is highly imbalanced!")
            elif balance_ratio > 3:
                print("⚠️  NOTE: Dataset has moderate imbalance")
            else:
                print("✅ GOOD: Dataset is relatively balanced")
        
        # Split balance analysis
        print(f"\n📊 SPLIT BALANCE ANALYSIS:")
        print("-" * 30)
        
        split_percentages = {}
        for split in valid_splits:
            annotations = split_stats[split]['total_annotations']
            percentage = (annotations / total_annotations_all) * 100 if total_annotations_all > 0 else 0
            split_percentages[split] = percentage
            print(f"{split.title()}: {percentage:.1f}%")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 20)
        
        # Check for classes with very few samples
        low_sample_species = [(cid, cnt) for cid, cnt in sorted_species if cnt < 50]
        if low_sample_species:
            print(f"⚠️  Species with <50 samples (may need more data):")
            for class_id, count in low_sample_species[:5]:  # Show first 5
                species_name = self.species_names.get(class_id, f"Unknown_{class_id}")
                print(f"   • {species_name}: {count} samples")
        
        # Check split balance
        if 'train' in split_percentages:
            train_pct = split_percentages['train']
            if train_pct < 60:
                print(f"⚠️  Training split only {train_pct:.1f}% - consider increasing for better model performance")
            elif train_pct > 85:
                print(f"⚠️  Training split {train_pct:.1f}% - consider more validation/test data")
        
        # Check for missing species in specific splits
        for split in valid_splits:
            split_classes = set(split_stats[split]['class_counts'].keys())
            missing_in_split = all_class_ids - split_classes
            if missing_in_split:
                missing_names = [self.species_names.get(cid, f"Unknown_{cid}") for cid in missing_in_split]
                print(f"⚠️  Species missing from {split} split: {', '.join(missing_names[:3])}")
                if len(missing_names) > 3:
                    print(f"     ... and {len(missing_names) - 3} more")
        
        # Summary
        print(f"\n📋 SUMMARY:")
        print(f"✅ Training dataset contains {len(all_class_ids)} species across {len(valid_splits)} splits")
        print(f"📊 Total: {total_annotations_all} annotations in {total_files_all} files from {total_images_all} images")
        balance_ratio = sorted_species[0][1] / sorted_species[-1][1] if len(sorted_species) > 1 and sorted_species[-1][1] > 0 else 1
        if balance_ratio <= 3:
            print(f"⚖️  Dataset appears well-balanced for training")
        else:
            print(f"⚠️  Consider balancing dataset or using class weights during training")
    
    def _display_simple_dataset_results(self, images_dir, labels_dir, total_images, 
                                       extension_counts, annotation_count, species_breakdown, class_counts):
        """Display simple dataset analysis results"""
        print(f"\n" + "="*60)
        print("📊 SIMPLE DATASET ANALYSIS RESULTS")
        print("="*60)
        
        print(f"📁 Images directory: {images_dir}")
        print(f"📁 Labels directory: {labels_dir}")
        print(f"📸 Total images: {total_images:,}")
        
        if extension_counts:
            print(f"\n📊 Images by format:")
            for ext, count in sorted(extension_counts.items()):
                percentage = (count / total_images) * 100
                print(f"   {ext}: {count:,} ({percentage:.1f}%)")
        
        # Annotation status
        if labels_dir.exists():
            print(f"\n🏷️  Annotation status:")
            print(f"   Annotation files: {annotation_count:,}")
            if total_images > 0:
                annotated_percentage = (annotation_count / total_images) * 100
                remaining = total_images - annotation_count
                print(f"   Progress: {annotated_percentage:.1f}% annotated")
                print(f"   Remaining: {remaining:,} images to annotate")
        else:
            print(f"\n🏷️  Annotations: No labels directory found")
        
        # Species breakdown if available
        if species_breakdown:
            total_annotations = sum(species_breakdown.values())
            print(f"\n🦌 SPECIES BREAKDOWN:")
            print(f"   Total annotations: {total_annotations}")
            print("-" * 40)
            
            # Sort by count (descending)
            sorted_species = sorted(species_breakdown.items(), key=lambda x: x[1], reverse=True)
            
            for species_name, count in sorted_species:
                percentage = (count / total_annotations) * 100 if total_annotations > 0 else 0
                print(f"   {species_name:<20} {count:<8} ({percentage:.1f}%)")
            
            # Balance analysis for species
            if len(sorted_species) > 1:
                max_count = sorted_species[0][1]
                min_count = sorted_species[-1][1]
                balance_ratio = max_count / min_count if min_count > 0 else float('inf')
                
                print(f"\n⚖️  Species balance ratio: {balance_ratio:.1f}:1")
                if balance_ratio > 10:
                    print("   ⚠️  Dataset is highly imbalanced!")
                elif balance_ratio > 3:
                    print("   ⚠️  Dataset has moderate imbalance")
                else:
                    print("   ✅ Dataset is relatively balanced")
        
        # Storage estimate
        if total_images > 0:
            avg_size_mb = 2.5  # Rough estimate for typical camera images
            estimated_size_gb = (total_images * avg_size_mb) / 1024
            print(f"\n💾 Estimated storage: ~{estimated_size_gb:.1f} GB")
        
        print(f"\n📋 SUMMARY:")
        if species_breakdown:
            print(f"✅ Dataset contains {len(species_breakdown)} species with {sum(species_breakdown.values())} total annotations")
        else:
            print(f"✅ Dataset contains {total_images} images with {annotation_count} annotation files")
        print(f"📊 Structure: Simple dataset (good for annotation or single-split training)")
    
    def dataset_analysis_menu(self):
        """Enhanced menu for comprehensive dataset analysis"""
        print("\n" + "-"*50)
        print("📊 COMPREHENSIVE DATASET ANALYSIS")
        print("-"*50)
        print("Automatically detects and analyzes:")
        print("• Training datasets (train/val/test structure)")
        print("• Simple datasets (images + labels)")
        print("• Species distribution and balance")
        
        # Get dataset directory
        dataset_dir = self.get_file_path("Enter dataset directory to analyze:")
        if not dataset_dir:
            print("❌ No path provided")
            return
        
        if not os.path.exists(dataset_dir):
            print("❌ Dataset directory does not exist")
            return
        
        # Run comprehensive analysis
        self.comprehensive_dataset_analysis(dataset_dir)
    
    def create_classes_file(self, output_path):
        """Create predefined_classes.txt file for labelImg"""
        if not self.species_names:
            print("✗ No species loaded from YAML")
            return False
        
        try:
            with open(output_path, 'w') as f:
                for class_id in sorted(self.species_names.keys()):
                    f.write(f"{self.species_names[class_id]}\n")
            print(f"✓ Created classes file: {output_path}")
            return True
        except Exception as e:
            print(f"✗ Error creating classes file: {e}")
            return False
    
    def auto_rename_by_species_from_annotations(self, directory):
        """Automatically detect species from annotations and rename files accordingly"""
        labels_dir = Path(directory) / "labels"
        
        if not labels_dir.exists():
            print(f"✗ No labels directory found: {labels_dir}")
            print("   This feature requires existing annotation files")
            return False
        
        # Check for standard YOLO structure with separate images/ and labels/ folders
        images_dir = Path(directory) / "images"
        if images_dir.exists():
            print(f"✅ Detected YOLO dataset structure:")
            print(f"   📁 Images: {images_dir}")
            print(f"   📄 Labels: {labels_dir}")
        else:
            # Fallback: images in same directory as labels folder
            images_dir = Path(directory)
            print(f"✅ Using single directory structure:")
            print(f"   📁 Directory: {directory}")
        
        # Get all annotation files (excluding system files)
        system_files = {"predefined_classes.txt", "classes.txt", "obj.names", "obj.data"}
        annotation_files = []
        
        for txt_file in labels_dir.glob("*.txt"):
            if txt_file.name.lower() not in system_files:
                annotation_files.append(txt_file)
        
        if not annotation_files:
            print("✗ No annotation files found to process")
            return False
        
        print(f"🔍 Analyzing {len(annotation_files)} annotation files...")
        
        # Debug: Show what images are available in the images directory
        available_images = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Use a set to avoid duplicates and then convert to list
        image_paths_set = set()
        for ext in image_extensions:
            image_paths_set.update(images_dir.glob(f"*{ext}"))
            image_paths_set.update(images_dir.glob(f"*{ext.upper()}"))
        
        available_images = list(image_paths_set)
        
        print(f"📁 Found {len(available_images)} unique images in {images_dir}")
        if len(available_images) > 0:
            print(f"   Example images: {[img.name for img in sorted(available_images)[:3]]}")
        
        # Create a lookup dictionary for faster matching
        image_lookup = {}
        for img in available_images:
            image_lookup[img.stem.lower()] = img
        
        # Parse annotations and group by detected species
        species_groups = {}  # species_name -> list of (image_file, annotation_file)
        unmatched_files = []
        matched_count = 0
        
        for annotation_file in annotation_files:
            try:
                # Find corresponding image file using the lookup dictionary
                image_stem = annotation_file.stem
                corresponding_image = None
                
                print(f"\n🔍 Processing: {annotation_file.name}")
                print(f"   Looking for image stem: '{image_stem}'")
                
                # Use case-insensitive lookup first
                if image_stem.lower() in image_lookup:
                    corresponding_image = image_lookup[image_stem.lower()]
                    print(f"   ✓ QUICK MATCH: {corresponding_image.name}")
                else:
                    # Fallback to manual search if lookup fails
                    print(f"   No quick match, trying manual search...")
                    for img in available_images:
                        if img.stem == image_stem:  # Exact case match
                            corresponding_image = img
                            print(f"   ✓ EXACT MATCH: {corresponding_image.name}")
                            break
                        elif img.stem.lower() == image_stem.lower():  # Case-insensitive match
                            corresponding_image = img
                            print(f"   ✓ CASE-INSENSITIVE MATCH: {corresponding_image.name}")
                            break
                
                if not corresponding_image:
                    print(f"   ✗ NO MATCH FOUND for: {annotation_file.name}")
                    print(f"   📊 Total available images: {len(available_images)}")
                    
                    # Show what files actually exist with similar names
                    similar_files = []
                    for img in available_images:
                        if (image_stem.lower() in img.stem.lower() or 
                            img.stem.lower() in image_stem.lower() or
                            abs(len(img.stem) - len(image_stem)) <= 2):  # Similar length
                            similar_files.append(img.name)
                    
                    if similar_files:
                        print(f"   📁 Similar files: {similar_files[:5]}")
                    else:
                        print(f"   📁 No similar files found")
                        # Show some example files for comparison
                        example_files = [img.name for img in sorted(available_images)[:5]]
                        print(f"   📋 Example images: {example_files}")
                    
                    unmatched_files.append((None, annotation_file))
                    continue
                
                matched_count += 1
                
                # Parse annotation file to extract class IDs
                detected_classes = set()
                annotation_content = ""
                try:
                    with open(annotation_file, 'r') as f:
                        annotation_content = f.read().strip()
                        for line_num, line in enumerate(annotation_content.split('\n'), 1):
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) >= 5:  # YOLO format: class_id x_center y_center width height
                                    try:
                                        class_id = int(parts[0])
                                        # Validate that coordinates are reasonable (0-1 range for YOLO)
                                        x_center, y_center, width, height = map(float, parts[1:5])
                                        if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
                                            detected_classes.add(class_id)
                                        else:
                                            print(f"   ⚠️  Invalid coordinates on line {line_num}: {line}")
                                    except ValueError as ve:
                                        print(f"   ⚠️  Invalid format on line {line_num}: {line} - {ve}")
                                else:
                                    print(f"   ⚠️  Incomplete annotation on line {line_num}: {line} (need 5 values)")
                except Exception as e:
                    print(f"   ✗ Error reading annotation file: {e}")
                
                if not detected_classes:
                    print(f"   ⚠️  No valid annotations found in: {annotation_file.name}")
                    print(f"   📄 File content preview: {annotation_content[:100]}...")
                    unmatched_files.append((corresponding_image, annotation_file))
                    continue
                
                # Handle multiple species in one image
                if len(detected_classes) > 1:
                    print(f"   📋 Multiple species detected: {detected_classes}")
                    # Use the first (primary) species for naming
                    primary_class = min(detected_classes)
                else:
                    primary_class = list(detected_classes)[0]
                
                # Look up species name from loaded YAML
                if primary_class in self.species_names:
                    species_name = self.species_names[primary_class]
                    print(f"   ✅ Detected: Class {primary_class} = {species_name}")
                    
                    # Clean species name for filename
                    clean_species_name = "".join(c for c in species_name if c.isalnum() or c in "_ ").replace(" ", "_")
                    
                    if clean_species_name not in species_groups:
                        species_groups[clean_species_name] = []
                    
                    species_groups[clean_species_name].append((corresponding_image, annotation_file))
                else:
                    print(f"   ⚠️  Unknown class ID {primary_class} in {annotation_file.name}")
                    print(f"   📋 Available classes: {list(self.species_names.keys())}")
                    print(f"   📄 Annotation content: {annotation_content[:50]}...")
                    unmatched_files.append((corresponding_image, annotation_file))
                    
            except Exception as e:
                print(f"✗ Error processing {annotation_file.name}: {e}")
                if 'corresponding_image' in locals() and corresponding_image:
                    unmatched_files.append((corresponding_image, annotation_file))
        
        if not species_groups:
            print("✗ No valid species annotations found")
            print(f"💡 Debug Info:")
            print(f"   - Annotation files: {len(annotation_files)}")
            print(f"   - Images found: {len(available_images)}")
            print(f"   - Successful matches: {matched_count}")
            print(f"   - Available species in YAML: {list(self.species_names.items())}")
            return False
        
        # Show detection results
        print(f"\n📊 SPECIES DETECTION RESULTS:")
        total_files = 0
        for species_name, file_pairs in species_groups.items():
            print(f"   🦌 {species_name}: {len(file_pairs)} files")
            total_files += len(file_pairs)
        
        if unmatched_files:
            print(f"   ❓ Unmatched/Invalid: {len(unmatched_files)} files")
        
        # Confirm renaming
        print(f"\n🏷️  RENAMING PREVIEW:")
        print(f"   Will rename {total_files} image-annotation pairs")
        print(f"   Format: [SpeciesName]_[Number].[extension]")
        print(f"   Example: WhiteTail_1.jpg, Elk_23.jpg")
        
        confirm = input(f"\n🔄 Proceed with automatic species renaming? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("❌ Renaming cancelled")
            return False
        
        # Perform renaming
        print(f"\n🚀 Starting automatic species renaming...")
        total_renamed = 0
        renamed_log = []  # Keep track of what was renamed
        
        for species_name, file_pairs in species_groups.items():
            counter = 1
            print(f"\n🦌 Renaming {species_name} files...")
            
            for image_file, annotation_file in sorted(file_pairs, key=lambda x: x[0].name):
                try:
                    # Generate new names with simple numbering
                    new_base_name = f"{species_name}_{counter}"
                    new_image_name = f"{new_base_name}{image_file.suffix}"
                    new_annotation_name = f"{new_base_name}.txt"
                    
                    new_image_path = image_file.parent / new_image_name
                    new_annotation_path = annotation_file.parent / new_annotation_name
                    
                    # Check if destination already exists and increment counter if needed
                    while new_image_path.exists():
                        counter += 1
                        new_base_name = f"{species_name}_{counter}"
                        new_image_name = f"{new_base_name}{image_file.suffix}"
                        new_annotation_name = f"{new_base_name}.txt"
                        new_image_path = image_file.parent / new_image_name
                        new_annotation_path = annotation_file.parent / new_annotation_name
                    
                    # Store the rename operation for logging
                    renamed_log.append({
                        'old_image': image_file.name,
                        'new_image': new_image_name,
                        'old_annotation': annotation_file.name,
                        'new_annotation': new_annotation_name,
                        'species': species_name
                    })
                    
                    # Rename image file
                    image_file.rename(new_image_path)
                    
                    # Rename annotation file
                    annotation_file.rename(new_annotation_path)
                    
                    if counter <= 3 or counter % 10 == 0:  # Show some progress
                        print(f"   📝 {image_file.name} → {new_image_name}")
                    
                    counter += 1
                    total_renamed += 1
                    
                except Exception as e:
                    print(f"   ✗ Error renaming {image_file.name}: {e}")
        
        print(f"\n✅ AUTOMATIC RENAMING COMPLETE!")
        print(f"   📊 Renamed: {total_renamed} image-annotation pairs")
        print(f"   🦌 Species detected: {len(species_groups)}")
        
        # Show summary by species
        print(f"\n📋 RENAMING SUMMARY BY SPECIES:")
        species_actual_counts = {}
        for entry in renamed_log:
            species = entry['species']
            species_actual_counts[species] = species_actual_counts.get(species, 0) + 1
        
        for species_name in sorted(species_actual_counts.keys()):
            print(f"   🦌 {species_name}: {species_actual_counts[species_name]} files")
        
        # Offer to show detailed log
        if renamed_log:
            show_log = input(f"\n📄 Show detailed rename log? (y/n): ").strip().lower()
            if show_log in ['y', 'yes']:
                print(f"\n🔍 DETAILED RENAME LOG:")
                for i, entry in enumerate(renamed_log, 1):
                    print(f"   [{i:2d}] {entry['old_image']} → {entry['new_image']} ({entry['species']})")
                    if i >= 20:  # Limit to first 20 entries
                        remaining = len(renamed_log) - 20
                        if remaining > 0:
                            print(f"   ... and {remaining} more entries")
                        break
        
        if unmatched_files:
            print(f"\n⚠️  UNMATCHED FILES:")
            print(f"   📄 {len(unmatched_files)} files could not be auto-renamed")
            print(f"   💡 Reasons: invalid annotations, unknown species, or file format issues")
            
            show_unmatched = input("   Show detailed unmatched files analysis? (y/n): ").strip().lower()
            if show_unmatched in ['y', 'yes']:
                print(f"\n🔍 DETAILED UNMATCHED FILES ANALYSIS:")
                for i, (image_file, annotation_file) in enumerate(unmatched_files[:10], 1):
                    print(f"\n   [{i}] {annotation_file.name}:")
                    
                    if image_file:
                        print(f"      📁 Image: {image_file.name} ✓")
                    else:
                        print(f"      📁 Image: NOT FOUND ✗")
                        
                    # Show annotation content
                    try:
                        with open(annotation_file, 'r') as f:
                            content = f.read().strip()
                            if content:
                                lines = content.split('\n')
                                print(f"      📄 Annotation lines: {len(lines)}")
                                for line_num, line in enumerate(lines[:3], 1):  # Show first 3 lines
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        class_id = parts[0]
                                        if class_id.isdigit():
                                            class_name = self.species_names.get(int(class_id), "UNKNOWN")
                                            print(f"         Line {line_num}: Class {class_id} ({class_name}) + coords")
                                        else:
                                            print(f"         Line {line_num}: Invalid class ID '{class_id}'")
                                    else:
                                        print(f"         Line {line_num}: Incomplete ({len(parts)} parts): {line}")
                                if len(lines) > 3:
                                    print(f"         ... and {len(lines) - 3} more lines")
                            else:
                                print(f"      📄 Annotation: EMPTY FILE ✗")
                    except Exception as e:
                        print(f"      📄 Annotation: ERROR READING - {e}")
                        
                if len(unmatched_files) > 10:
                    print(f"\n   ... and {len(unmatched_files) - 10} more files")
                    
                print(f"\n💡 TROUBLESHOOTING TIPS:")
                print(f"   • Check annotation files are not empty")
                print(f"   • Ensure class IDs exist in YAML: {list(self.species_names.keys())}")
                print(f"   • Verify YOLO format: class_id x_center y_center width height")
                print(f"   • Coordinates should be between 0.0 and 1.0")
        
        return True
    
    def find_resume_image(self, images_dir, labels_dir):
        """Find the exact image to resume annotation from"""
        import time
        
        # Get all annotation files with their modification times
        # Filter out system files that aren't actual annotations
        system_files = {"predefined_classes.txt", "classes.txt", "obj.names", "obj.data"}
        annotation_files = []
        
        for txt_file in Path(labels_dir).glob("*.txt"):
            # Skip system/config files
            if txt_file.name.lower() in system_files:
                continue
            
            # Additional check: make sure it's not a config file by checking if corresponding image exists
            image_stem = txt_file.stem
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            has_corresponding_image = False
            
            for ext in image_extensions:
                potential_image = Path(images_dir) / f"{image_stem}{ext}"
                if potential_image.exists():
                    has_corresponding_image = True
                    break
                potential_image = Path(images_dir) / f"{image_stem}{ext.upper()}"
                if potential_image.exists():
                    has_corresponding_image = True
                    break
            
            # Only include if it has a corresponding image (real annotation)
            if has_corresponding_image:
                try:
                    mod_time = txt_file.stat().st_mtime
                    annotation_files.append((txt_file, mod_time))
                except OSError:
                    continue
        
        if not annotation_files:
            print("📊 No real annotations found - will start from first image")
            # Find first image file
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            for ext in image_extensions:
                first_images = list(Path(images_dir).glob(f"*{ext}"))
                if first_images:
                    return str(sorted(first_images)[0])
                first_images = list(Path(images_dir).glob(f"*{ext.upper()}"))
                if first_images:
                    return str(sorted(first_images)[0])
            return None
        
        # Sort by modification time (most recent first)
        annotation_files.sort(key=lambda x: x[1], reverse=True)
        most_recent_annotation = annotation_files[0][0]
        most_recent_time = time.ctime(annotation_files[0][1])
        
        print(f"📊 Annotation Analysis:")
        print(f"   Most recent annotation: {most_recent_annotation.name}")
        print(f"   Created/Modified: {most_recent_time}")
        print(f"   Total real annotations: {len(annotation_files)}")
        
        # Find the corresponding image for the most recent annotation
        image_stem = most_recent_annotation.stem
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        corresponding_image = None
        for ext in image_extensions:
            potential_image = Path(images_dir) / f"{image_stem}{ext}"
            if potential_image.exists():
                corresponding_image = potential_image
                break
            potential_image = Path(images_dir) / f"{image_stem}{ext.upper()}"
            if potential_image.exists():
                corresponding_image = potential_image
                break
        
        if not corresponding_image:
            print(f"⚠️  Could not find image for annotation: {most_recent_annotation.name}")
            print("   Starting from first image instead")
            # Find first image file as fallback
            all_images = []
            for ext in image_extensions:
                all_images.extend(Path(images_dir).glob(f"*{ext}"))
                all_images.extend(Path(images_dir).glob(f"*{ext.upper()}"))
            if all_images:
                return str(sorted(all_images)[0])
            return None
        
        # Get all image files to find next unannotated
        all_images = []
        for ext in image_extensions:
            all_images.extend(Path(images_dir).glob(f"*{ext}"))
            all_images.extend(Path(images_dir).glob(f"*{ext.upper()}"))
        
        all_images = sorted(all_images)
        annotated_stems = {ann[0].stem for ann in annotation_files}
        
        # Find the next unannotated image after the most recent work
        next_unannotated = None
        found_recent = False
        
        for img in all_images:
            if img.stem == image_stem:
                found_recent = True
                continue
            if found_recent and img.stem not in annotated_stems:
                next_unannotated = img
                break
        
        # If no unannotated after recent, find first unannotated overall
        if not next_unannotated:
            for img in all_images:
                if img.stem not in annotated_stems:
                    next_unannotated = img
                    break
        
        if next_unannotated:
            print(f"✅ Resume from: {next_unannotated.name}")
            print(f"   (Last worked on: {corresponding_image.name})")
            return str(next_unannotated)
        else:
            print(f"✅ All images annotated! Reviewing most recent: {corresponding_image.name}")
            return str(corresponding_image)
    
    def cleanup_and_merge_annotations(self, original_labels_dir, temp_labels_dir, temp_workspace_dir):
        """Copy new annotations back to original location and clean up"""
        try:
            # Copy any new/updated annotations back
            new_annotations = 0
            if os.path.exists(temp_labels_dir):
                for txt_file in Path(temp_labels_dir).glob("*.txt"):
                    if txt_file.name != "predefined_classes.txt":
                        original_path = Path(original_labels_dir) / txt_file.name
                        shutil.copy2(txt_file, original_path)
                        new_annotations += 1
            
            # Calculate space freed
            space_freed = self.calculate_directory_size(temp_workspace_dir)
            
            # Clean up temporary directory
            if os.path.exists(temp_workspace_dir):
                shutil.rmtree(temp_workspace_dir)
            
            print(f"\n✅ Workspace cleanup complete!")
            print(f"   📄 Copied back: {new_annotations} annotation files")
            print(f"   🗑️  Removed temporary workspace")
            print(f"   💾 Space freed: {self.format_file_size(space_freed)}")
            
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")
            print(f"   📁 Temp workspace: {temp_workspace_dir}")
            print(f"   🔧 You may need to manually delete the temp folder")
    
    def calculate_directory_size(self, directory):
        """Calculate total size of directory in bytes"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception:
            pass
        return total_size
    
    def format_file_size(self, size_bytes):
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def cleanup_old_workspaces(self, images_dir):
        """Clean up any leftover temporary workspaces from previous sessions"""
        temp_workspace_pattern = "temp_annotation_workspace*"
        cleanup_count = 0
        total_space_freed = 0
        
        try:
            # Look for temp workspaces in the images directory
            for item in Path(images_dir).glob(temp_workspace_pattern):
                if item.is_dir():
                    space_freed = self.calculate_directory_size(str(item))
                    shutil.rmtree(str(item))
                    cleanup_count += 1
                    total_space_freed += space_freed
                    print(f"   🗑️  Cleaned up old workspace: {item.name}")
            
            if cleanup_count > 0:
                print(f"✅ Cleaned up {cleanup_count} old workspace(s)")
                print(f"   💾 Space freed: {self.format_file_size(total_space_freed)}")
            
        except Exception as e:
            print(f"⚠️  Error cleaning old workspaces: {e}")
    
    def diagnose_file_matching(self, directory):
        """Diagnostic function to debug file matching issues"""
        print("\n" + "-"*50)
        print("🔬 FILE MATCHING DIAGNOSTIC")
        print("-"*50)
        
        labels_dir = Path(directory) / "labels"
        images_dir = Path(directory) / "images"
        
        if not labels_dir.exists():
            print(f"❌ Labels directory not found: {labels_dir}")
            return
            
        if not images_dir.exists():
            print(f"❌ Images directory not found: {images_dir}")
            return
        
        print(f"📁 Checking: {images_dir}")
        print(f"📄 Checking: {labels_dir}")
        
        # Get all files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        all_images = []
        for ext in image_extensions:
            all_images.extend(images_dir.glob(f"*{ext}"))
            all_images.extend(images_dir.glob(f"*{ext.upper()}"))
        
        all_annotations = list(labels_dir.glob("*.txt"))
        system_files = {"predefined_classes.txt", "classes.txt", "obj.names", "obj.data"}
        annotation_files = [f for f in all_annotations if f.name.lower() not in system_files]
        
        print(f"\n📊 SUMMARY:")
        print(f"   Images found: {len(all_images)}")
        print(f"   Annotation files: {len(annotation_files)}")
        
        # Show specific problematic case
        test_stem = "0u40fbady8o9_0092"
        print(f"\n🔍 TESTING SPECIFIC CASE: {test_stem}")
        
        # Check if annotation exists
        test_annotation = labels_dir / f"{test_stem}.txt"
        print(f"   Annotation file: {test_annotation}")
        print(f"   Annotation exists: {test_annotation.exists()}")
        
        # Check all possible image matches
        print(f"   Looking for matching images:")
        for ext in image_extensions:
            test_image_lower = images_dir / f"{test_stem}{ext}"
            test_image_upper = images_dir / f"{test_stem}{ext.upper()}"
            
            print(f"      {test_image_lower} → {test_image_lower.exists()}")
            print(f"      {test_image_upper} → {test_image_upper.exists()}")
        
        # Show actual files that contain this stem
        print(f"   Files containing '{test_stem}':")
        matching_images = [img for img in all_images if test_stem in img.name]
        for img in matching_images:
            print(f"      📁 {img.name}")
            
        # Show first few images for comparison
        print(f"\n📋 FIRST 10 IMAGES IN DIRECTORY:")
        for i, img in enumerate(sorted(all_images)[:10], 1):
            print(f"   {i:2d}. {img.name}")
            
        # Show first few annotations for comparison  
        print(f"\n📋 FIRST 10 ANNOTATIONS:")
        for i, ann in enumerate(sorted(annotation_files)[:10], 1):
            corresponding_images = [img for img in all_images if img.stem == ann.stem]
            status = "✓" if corresponding_images else "✗"
            print(f"   {i:2d}. {ann.name} {status}")
            if corresponding_images:
                print(f"       → {corresponding_images[0].name}")
    
    def manual_cleanup_menu(self):
        """Manual cleanup menu for leftover temporary files"""
        print("\n" + "-"*30)
        print("🗑️  MANUAL WORKSPACE CLEANUP")
        print("-"*30)
        
        # Get directory
        directory = self.get_file_path("Enter directory to check for leftover workspaces:")
        if not directory or not os.path.exists(directory):
            print("❌ Invalid directory path")
            return
        
        # Look for temp workspaces
        temp_workspace_pattern = "temp_annotation_workspace*"
        temp_workspaces = list(Path(directory).glob(temp_workspace_pattern))
        temp_workspaces = [w for w in temp_workspaces if w.is_dir()]
        
        if not temp_workspaces:
            print("✅ No temporary workspaces found - directory is clean!")
            return
        
        print(f"\n📁 Found {len(temp_workspaces)} temporary workspace(s):")
        total_size = 0
        
        for workspace in temp_workspaces:
            size = self.calculate_directory_size(str(workspace))
            total_size += size
            print(f"   📂 {workspace.name} - {self.format_file_size(size)}")
        
        print(f"\n💾 Total space used: {self.format_file_size(total_size)}")
        
        # Confirm cleanup
        confirm = input(f"\n🗑️  Delete all temporary workspaces? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            cleaned_count = 0
            for workspace in temp_workspaces:
                try:
                    shutil.rmtree(str(workspace))
                    cleaned_count += 1
                    print(f"   ✅ Deleted: {workspace.name}")
                except Exception as e:
                    print(f"   ❌ Error deleting {workspace.name}: {e}")
            
            print(f"\n✅ Cleanup complete!")
            print(f"   🗑️  Deleted: {cleaned_count}/{len(temp_workspaces)} workspaces")
            print(f"   💾 Space freed: {self.format_file_size(total_size)}")
        else:
            print("❌ Cleanup cancelled")
    
    def create_resume_workspace(self, images_dir, labels_dir, resume_image_path):
        """Create a temporary workspace starting from resume point"""
        import tempfile
        
        # Clean up any old workspaces first
        print(f"🧹 Checking for old temporary workspaces...")
        self.cleanup_old_workspaces(images_dir)
        
        # Create temporary directory
        temp_dir = os.path.join(images_dir, "temp_annotation_workspace")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        temp_images_dir = temp_dir
        temp_labels_dir = os.path.join(temp_dir, "labels")
        os.makedirs(temp_labels_dir, exist_ok=True)
        
        # Get all images sorted
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        all_images = []
        for ext in image_extensions:
            all_images.extend(Path(images_dir).glob(f"*{ext}"))
            all_images.extend(Path(images_dir).glob(f"*{ext.upper()}"))
        
        all_images = sorted(all_images)
        
        # Find the resume image index
        resume_image = Path(resume_image_path)
        resume_index = -1
        for i, img in enumerate(all_images):
            if img.name == resume_image.name:
                resume_index = i
                break
        
        if resume_index == -1:
            print("⚠️  Could not find resume image in directory")
            return None, None
        
        # Images to copy (from resume point onwards)
        images_to_copy = all_images[resume_index:]
        total_images = len(images_to_copy)
        
        print(f"🔧 Creating resume workspace...")
        print(f"   📊 Images to copy: {total_images}")
        
        if total_images > 50:
            print(f"   ⏳ This may take a moment - copying {total_images} images...")
            
        # Estimate space usage
        if total_images > 0:
            sample_size = os.path.getsize(images_to_copy[0]) if os.path.exists(images_to_copy[0]) else 2.5 * 1024 * 1024
            estimated_space = (sample_size * total_images)
            print(f"   💾 Estimated workspace size: {self.format_file_size(estimated_space)}")
        
        # Copy images starting from resume point with progress
        copied_images = 0
        last_percentage = -1
        
        for i, img in enumerate(images_to_copy):
            try:
                dest_path = Path(temp_images_dir) / img.name
                shutil.copy2(img, dest_path)
                copied_images += 1
                
                # Also copy existing annotation if it exists
                annotation_file = Path(labels_dir) / f"{img.stem}.txt"
                if annotation_file.exists():
                    dest_annotation = Path(temp_labels_dir) / annotation_file.name
                    shutil.copy2(annotation_file, dest_annotation)
                
                # Show progress bar every 5% or every 10 files for smaller batches
                current_percentage = int((i + 1) / total_images * 100)
                
                if (total_images > 20 and current_percentage != last_percentage and current_percentage % 5 == 0) or \
                   (total_images <= 20 and (i + 1) % max(1, total_images // 10) == 0) or \
                   (i + 1) == total_images:
                    
                    # Create progress bar
                    bar_length = 20
                    filled_length = int(bar_length * (i + 1) / total_images)
                    bar = "█" * filled_length + "░" * (bar_length - filled_length)
                    
                    print(f"\r   📁 [{bar}] {current_percentage:3d}% ({i + 1}/{total_images})", end="", flush=True)
                    last_percentage = current_percentage
                    
            except Exception as e:
                print(f"\n⚠️  Error copying {img.name}: {e}")
        
        print()  # New line after progress bar
        
        # Copy predefined classes file
        classes_file = Path(labels_dir) / "predefined_classes.txt"
        if classes_file.exists():
            dest_classes = Path(temp_labels_dir) / "predefined_classes.txt"
            shutil.copy2(classes_file, dest_classes)
        
        # Calculate actual workspace size
        workspace_size = self.calculate_directory_size(temp_dir)
        
        print(f"✅ Workspace created successfully!")
        print(f"   📍 Starting with: {resume_image.name}")
        print(f"   📊 Images ready to annotate: {copied_images}")
        print(f"   💾 Workspace size: {self.format_file_size(workspace_size)}")
        print(f"   ⚠️  Workspace will be auto-deleted when you close labelImg")
        
        return temp_images_dir, temp_labels_dir
    
    def launch_labelimg(self, images_path, is_directory=True, resume_from_last=False):
        """Launch labelImg for annotation with proper resume functionality"""
        
        # Check if labelImg is installed
        try:
            result = subprocess.run(['labelImg', '--help'], capture_output=True, text=True)
        except FileNotFoundError:
            print("❌ labelImg not found!")
            print("   Install it with: pip install labelImg")
            print("   Or: conda install -c conda-forge labelimg")
            return False
        
        if is_directory:
            images_dir = images_path
            labels_dir = os.path.join(images_dir, "labels")
        else:
            images_dir = os.path.dirname(images_path)
            labels_dir = os.path.join(images_dir, "labels")
        
        # Create labels directory if it doesn't exist
        os.makedirs(labels_dir, exist_ok=True)
        
        # Create predefined classes file
        classes_file = os.path.join(labels_dir, "predefined_classes.txt")
        if not self.create_classes_file(classes_file):
            return False
        
        # Save this directory for memory
        self.save_last_directory(images_dir)
        
        # Handle resume functionality
        if is_directory and resume_from_last:
            # Find the exact image to resume from
            resume_image = self.find_resume_image(images_dir, labels_dir)
            if resume_image:
                temp_images_dir, temp_labels_dir = self.create_resume_workspace(images_dir, labels_dir, resume_image)
                
                if temp_images_dir and temp_labels_dir:
                    # Launch labelImg on temporary workspace
                    temp_classes_file = os.path.join(temp_labels_dir, "predefined_classes.txt")
                    cmd = ['labelImg', temp_images_dir, temp_classes_file, temp_labels_dir]
                    
                    print(f"\n🚀 Launching labelImg from resume point...")
                    print(f"   📍 Starting directly at your next image!")
                    
                    print("\n💡 labelImg Controls:")
                    print("   • W = Draw bounding box")
                    print("   • A/D = Previous/Next image") 
                    print("   • Ctrl+S = Save annotation")
                    print("   • Space = Mark as verified")
                    print("   • Del = Delete selected box")
                    
                    try:
                        # Launch labelImg and wait for it to close
                        process = subprocess.run(cmd)
                        print("\n🔄 labelImg closed. Processing annotations...")
                        
                    except Exception as e:
                        print(f"✗ Error running labelImg: {e}")
                        
                    finally:
                        # Always cleanup, even if there was an error
                        print("🧹 Cleaning up workspace...")
                        self.cleanup_and_merge_annotations(labels_dir, temp_labels_dir, temp_images_dir)
                    
                    return True
                else:
                    print("⚠️  Could not create workspace, falling back to normal mode")
        
        # Normal mode (non-resume or fallback)
        cmd = ['labelImg', images_dir, classes_file, labels_dir]
        
        print(f"\n🚀 Launching labelImg...")
        print(f"   Images: {images_dir}")
        print(f"   Labels: {labels_dir}")
        print(f"   Classes: {classes_file}")
        
        print("\n💡 labelImg Controls:")
        print("   • W = Draw bounding box")
        print("   • A/D = Previous/Next image") 
        print("   • Ctrl+S = Save annotation")
        print("   • Space = Mark as verified")
        print("   • Del = Delete selected box")
        
        try:
            # Launch labelImg
            process = subprocess.Popen(cmd)
            print("✅ labelImg launched successfully!")
            return True
        except Exception as e:
            print(f"✗ Error launching labelImg: {e}")
            return False
    
    def extract_frames_from_video(self, video_path, output_dir, randomize=False, extract_all=False, interval=30):
        """Extract frames from a single video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"✗ Could not open video: {video_path}")
            return 0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0:
            print(f"✗ Invalid FPS for video: {video_path}")
            cap.release()
            return 0
            
        video_name = Path(video_path).stem
        
        print(f"📹 Processing: {video_name}")
        print(f"   FPS: {fps:.1f}, Total frames: {total_frames}, Duration: {total_frames/fps:.1f}s")
        
        if extract_all:
            print(f"   Extracting ALL {total_frames} frames")
            frame_interval = 1  # Every frame
        else:
            frame_interval = max(1, int(fps * interval))
            expected_frames = total_frames // frame_interval
            print(f"   Extracting every {interval} seconds → ~{expected_frames} frames")
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if extract_all or (frame_count % frame_interval == 0):
                if randomize:
                    # Generate random filename
                    random_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
                    filename = f"{random_name}.jpg"
                else:
                    # Sequential naming with video name
                    filename = f"{video_name}_{extracted_count:04d}.jpg"
                
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, frame)
                extracted_count += 1
                
                # Show progress for large extractions
                if extract_all and extracted_count % 500 == 0:
                    print(f"   📸 {extracted_count} frames extracted...")
            
            frame_count += 1
        
        cap.release()
        print(f"   ✓ {extracted_count} frames extracted")
        return extracted_count

    def extract_frames_from_directory(self, directory, randomize=False, extract_all=False, interval=30):
        """Extract frames from all videos in a directory"""
        if not os.path.exists(directory):
            print(f"✗ Directory not found: {directory}")
            return False
        
        # Comprehensive list of video extensions
        video_extensions = [
            '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.mpg', '.mpeg', 
            '.3gp', '.webm', '.ogg', '.ogv', '.ts', '.mts', '.m2ts', '.vob', 
            '.asf', '.rm', '.rmvb', '.divx', '.xvid'
        ]
        
        # Find all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(Path(directory).glob(f"*{ext}"))
            video_files.extend(Path(directory).glob(f"*{ext.upper()}"))
        
        if not video_files:
            print(f"✗ No video files found in {directory}")
            print(f"   Supported formats: {', '.join(video_extensions)}")
            return False
        
        print(f"🎥 Found {len(video_files)} video files")
        
        # Create output directory for frames
        output_dir = os.path.join(directory, "extracted_frames")
        os.makedirs(output_dir, exist_ok=True)
        
        total_extracted = 0
        
        for i, video_file in enumerate(sorted(video_files), 1):
            print(f"\n[{i}/{len(video_files)}] Processing video...")
            extracted = self.extract_frames_from_video(str(video_file), output_dir, randomize, extract_all, interval)
            total_extracted += extracted
        
        print(f"\n🎉 EXTRACTION COMPLETE!")
        print(f"   Processed: {len(video_files)} videos")
        print(f"   Total frames extracted: {total_extracted}")
        print(f"   Frames saved to: {output_dir}")
        return True
    
    def rename_by_species(self, directory, species_id):
        """Rename files by species and keep annotations paired"""
        if species_id not in self.species_names:
            print(f"✗ Species ID {species_id} not found in YAML")
            return False
        
        species_name = self.species_names[species_id]
        
        # Check if this is a YOLO dataset structure
        yolo_images_dir = Path(directory) / "images"
        yolo_labels_dir = Path(directory) / "labels"
        
        if yolo_images_dir.exists() and yolo_labels_dir.exists():
            print("✅ Detected YOLO dataset structure!")
            print(f"   📁 Images: {yolo_images_dir}")
            print(f"   📄 Labels: {yolo_labels_dir}")
            actual_images_dir = yolo_images_dir
            actual_labels_dir = yolo_labels_dir
            structure_type = "YOLO"
        else:
            print("✅ Using simple directory structure")
            actual_images_dir = Path(directory)
            actual_labels_dir = Path(directory) / "labels"
            structure_type = "simple"
        
        # Get all image/video files
        extensions = ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov']
        files = []
        
        for ext in extensions:
            files.extend(actual_images_dir.glob(f"*{ext}"))
            files.extend(actual_images_dir.glob(f"*{ext.upper()}"))
        
        if not files:
            print(f"✗ No image/video files found in {actual_images_dir}")
            return False
        
        # Check for labels directory
        has_labels = actual_labels_dir.exists()
        
        if has_labels:
            print(f"✓ Found labels directory - will rename annotation files too")
            print(f"   📁 Structure: {structure_type}")
        else:
            print(f"ℹ️  No labels directory found - renaming media files only")
        
        print(f"🏷️  Renaming {len(files)} files with species: {species_name}")
        
        renamed_annotations = 0
        
        for i, file_path in enumerate(sorted(files), 1):
            extension = file_path.suffix
            new_name = f"{species_name}_{i:03d}{extension}"
            new_path = file_path.parent / new_name
            
            # Check for corresponding annotation file
            label_file = actual_labels_dir / f"{file_path.stem}.txt"
            new_label_file = actual_labels_dir / f"{species_name}_{i:03d}.txt"
            
            try:
                # Rename media file
                file_path.rename(new_path)
                
                # Rename annotation file if it exists
                if has_labels and label_file.exists():
                    label_file.rename(new_label_file)
                    renamed_annotations += 1
                    annotation_status = " + annotation"
                else:
                    annotation_status = ""
                
                if i <= 5 or i % 20 == 0:  # Show progress
                    print(f"📝 {file_path.name} → {new_name}{annotation_status}")
                    
            except Exception as e:
                print(f"✗ Error renaming {file_path.name}: {e}")
        
        print(f"✅ RENAMING COMPLETE!")
        print(f"   📁 Structure: {structure_type}")
        print(f"   🏷️  Renamed {len(files)} files with species prefix: {species_name}")
        if has_labels:
            print(f"   📄 Renamed {renamed_annotations} corresponding annotation files")
        return True
    
    def randomize_filenames(self, directory):
        """Randomize all filenames and keep annotations paired"""
        
        # Check if this is a YOLO dataset structure
        yolo_images_dir = Path(directory) / "images"
        yolo_labels_dir = Path(directory) / "labels"
        
        if yolo_images_dir.exists() and yolo_labels_dir.exists():
            print("✅ Detected YOLO dataset structure!")
            print(f"   📁 Images: {yolo_images_dir}")
            print(f"   📄 Labels: {yolo_labels_dir}")
            actual_images_dir = yolo_images_dir
            actual_labels_dir = yolo_labels_dir
            structure_type = "YOLO"
        else:
            print("✅ Using simple directory structure")
            actual_images_dir = Path(directory)
            actual_labels_dir = Path(directory) / "labels"
            structure_type = "simple"
        
        # Find all image/video files in the appropriate directory
        extensions = ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov']
        files = []
        
        for ext in extensions:
            files.extend(actual_images_dir.glob(f"*{ext}"))
            files.extend(actual_images_dir.glob(f"*{ext.upper()}"))
        
        if not files:
            print(f"✗ No image/video files found in {actual_images_dir}")
            return False
        
        # Check for labels directory
        has_labels = actual_labels_dir.exists()
        
        if has_labels:
            print(f"✓ Found labels directory - will randomize annotation files too")
            print(f"   📁 Structure: {structure_type}")
        else:
            print(f"ℹ️  No labels directory found - randomizing media files only")
        
        print(f"🎲 Randomizing {len(files)} filenames...")
        
        # Create a mapping of random names to avoid duplicates
        used_names = set()
        renamed_annotations = 0
        
        for i, file_path in enumerate(files):
            extension = file_path.suffix
            
            # Generate unique random name
            while True:
                random_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
                if random_name not in used_names:
                    used_names.add(random_name)
                    break
            
            new_name = f"{random_name}{extension}"
            new_path = file_path.parent / new_name
            
            # Check for corresponding annotation file
            label_file = actual_labels_dir / f"{file_path.stem}.txt"
            new_label_file = actual_labels_dir / f"{random_name}.txt"
            
            try:
                # Rename media file
                file_path.rename(new_path)
                
                # Rename annotation file if it exists
                if has_labels and label_file.exists():
                    label_file.rename(new_label_file)
                    renamed_annotations += 1
                    annotation_status = " + annotation"
                else:
                    annotation_status = ""
                
                if i <= 5 or i % 20 == 0:  # Show progress
                    print(f"🔀 {file_path.name} → {new_name}{annotation_status}")
                    
            except Exception as e:
                print(f"✗ Error renaming {file_path.name}: {e}")
        
        print(f"✅ RANDOMIZATION COMPLETE!")
        print(f"   📁 Structure: {structure_type}")
        print(f"   🔀 Randomized {len(files)} filenames")
        if has_labels:
            print(f"   📄 Randomized {renamed_annotations} corresponding annotation files")
        return True
    
    def move_annotated_files(self, source_dir):
        """Move all annotated images and labels to a new 'annotated' folder"""
        if not os.path.exists(source_dir):
            print(f"✗ Directory not found: {source_dir}")
            return False
        
        labels_dir = os.path.join(source_dir, "labels")
        if not os.path.exists(labels_dir):
            print(f"✗ No labels directory found: {labels_dir}")
            return False
        
        # Create annotated folders
        annotated_dir = os.path.join(source_dir, "annotated")
        annotated_images_dir = os.path.join(annotated_dir, "images")
        annotated_labels_dir = os.path.join(annotated_dir, "labels")
        
        os.makedirs(annotated_images_dir, exist_ok=True)
        os.makedirs(annotated_labels_dir, exist_ok=True)
        
        # Find all label files
        label_files = list(Path(labels_dir).glob("*.txt"))
        # Remove predefined_classes.txt from the list
        label_files = [f for f in label_files if f.name != "predefined_classes.txt"]
        
        if not label_files:
            print("✗ No annotation files found")
            return False
        
        print(f"📦 Found {len(label_files)} annotated files to move")
        
        # Get all image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        moved_count = 0
        missing_images = []
        
        for label_file in label_files:
            # Find corresponding image file
            image_stem = label_file.stem
            corresponding_image = None
            
            for ext in image_extensions:
                potential_image = Path(source_dir) / f"{image_stem}{ext}"
                if potential_image.exists():
                    corresponding_image = potential_image
                    break
                # Also check uppercase extensions
                potential_image = Path(source_dir) / f"{image_stem}{ext.upper()}"
                if potential_image.exists():
                    corresponding_image = potential_image
                    break
            
            if corresponding_image:
                try:
                    # Move image file
                    new_image_path = Path(annotated_images_dir) / corresponding_image.name
                    corresponding_image.rename(new_image_path)
                    
                    # Move label file
                    new_label_path = Path(annotated_labels_dir) / label_file.name
                    label_file.rename(new_label_path)
                    
                    moved_count += 1
                    if moved_count <= 5 or moved_count % 20 == 0:  # Show progress
                        print(f"📁 Moved: {corresponding_image.name} + {label_file.name}")
                        
                except Exception as e:
                    print(f"✗ Error moving {corresponding_image.name}: {e}")
            else:
                missing_images.append(image_stem)
        
        print(f"\n✅ MOVE COMPLETE!")
        print(f"   Moved: {moved_count} image-label pairs")
        print(f"   Destination: {annotated_dir}")
        
        if missing_images:
            print(f"   ⚠️  {len(missing_images)} label files had no corresponding images:")
            for missing in missing_images[:5]:  # Show first 5
                print(f"      {missing}.txt")
            if len(missing_images) > 5:
                print(f"      ... and {len(missing_images) - 5} more")
        
        # Copy predefined_classes.txt to annotated labels directory
        predefined_classes = Path(labels_dir) / "predefined_classes.txt"
        if predefined_classes.exists():
            try:
                shutil.copy2(predefined_classes, annotated_labels_dir)
                print(f"   ✓ Copied predefined_classes.txt to annotated folder")
            except Exception as e:
                print(f"   ⚠️  Could not copy predefined_classes.txt: {e}")
        
        return True
    
    def _count_images_simple(self, images_dir, labels_dir):
        """Simple image counting for annotation menu"""
        # Count images by extension
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        total_images = 0
        
        for ext in image_extensions:
            total_images += len(list(Path(images_dir).glob(f"*{ext}")))
            total_images += len(list(Path(images_dir).glob(f"*{ext.upper()}")))
        
        # Count annotations
        annotation_count = 0
        if Path(labels_dir).exists():
            annotation_files = [f for f in Path(labels_dir).glob("*.txt") 
                              if f.name.lower() not in {"predefined_classes.txt", "classes.txt", "obj.names", "obj.data"}]
            annotation_count = len(annotation_files)
        
        print(f"   📸 Total images: {total_images}")
        print(f"   📄 Annotations: {annotation_count}")
        if total_images > 0 and annotation_count > 0:
            progress = (annotation_count / total_images) * 100
            print(f"   📈 Progress: {progress:.1f}% annotated")
        
        return total_images, annotation_count
    
    def show_species_menu(self):
        """Display available species"""
        print("\n🦌 Available Species:")
        for id_num, name in self.species_names.items():
            print(f"  {id_num}: {name}")
        print()
    
    def extract_frames_menu(self):
        """Frame extraction submenu"""
        print("\n" + "-"*30)
        print("📸 FRAME EXTRACTION FROM DIRECTORY")
        print("-"*30)
        
        # Get directory path containing videos
        directory = self.get_file_path("Enter directory containing video files:")
        if not directory:
            print("❌ No path provided")
            return
        
        # Choose extraction method
        print("\n🎯 EXTRACTION OPTIONS:")
        print("1: Extract ALL frames from each video (warning: creates LOTS of files)")
        print("2: Extract frames at regular intervals (recommended)")
        
        extraction_choice = input("➤ Choose option (1 or 2, default: 2): ").strip()
        extract_all = extraction_choice == '1'
        
        interval = 30  # Default
        
        if not extract_all:
            # Explain interval option with examples
            print("\n⏱️  INTERVAL EXPLANATION:")
            print("   Think of this as 'how often to take a snapshot while the video plays'")
            print("   📊 Examples for a 10-minute (600 second) video:")
            print("   • 30 seconds → 20 frames (1 every 30 sec)")
            print("   • 10 seconds → 60 frames (1 every 10 sec)")
            print("   • 60 seconds → 10 frames (1 every minute)")
            print("   • 5 seconds → 120 frames (1 every 5 sec)")
            print("\n   💡 For trail cams: 30-60 seconds is usually perfect")
            print("      (captures animal behavior without too many similar frames)")
            
            # Get extraction interval
            try:
                interval_input = input("\nEnter interval in seconds (default: 30): ").strip()
                interval = int(interval_input) if interval_input else 30
                if interval <= 0:
                    interval = 30
                    print("⚠️  Invalid interval, using default: 30 seconds")
            except ValueError:
                interval = 30
                print("⚠️  Invalid input, using default: 30 seconds")
            
            print(f"✓ Will take 1 frame every {interval} seconds from each video")
        
        # Ask about randomization
        randomize_choice = input("\nRandomize extracted frame filenames? (y/n, default: n): ").strip().lower()
        randomize = randomize_choice in ['y', 'yes', '1', 'true']
        
        if randomize:
            print("✓ Frame filenames will be randomized")
        else:
            print("✓ Frame filenames will include original video name + number")
        
        # Show what will happen
        if extract_all:
            print("\n⚠️  WARNING: Extracting ALL frames will create thousands of files!")
            confirm = input("   Continue? (y/n): ").strip().lower()
            if confirm not in ['y', 'yes']:
                print("❌ Operation cancelled")
                return
        
        # Extract frames from all videos in directory
        print(f"\n🚀 Starting extraction from directory: {directory}")
        self.extract_frames_from_directory(directory, randomize, extract_all, interval)
    
    def rename_menu(self):
        """File renaming submenu"""
        print("\n" + "-"*30)
        print("🏷️  SMART FILE RENAMING (WITH ANNOTATION PAIRING)")
        print("-"*30)
        
        # Get directory
        directory = self.get_file_path("Enter directory path:")
        if not directory:
            print("❌ No path provided")
            return
        
        # Check for labels directory and annotations
        labels_dir = Path(directory) / "labels"
        annotation_files = []
        
        if labels_dir.exists():
            print("✅ Labels directory detected!")
            
            # Check if there are actual annotation files
            annotation_files = [f for f in labels_dir.glob("*.txt") 
                              if f.name.lower() not in {"predefined_classes.txt", "classes.txt", "obj.names", "obj.data"}]
            
            if annotation_files:
                print(f"📊 Found {len(annotation_files)} annotation files")
                print("🤖 Auto-detection will be used for smart renaming!")
            else:
                print("ℹ️  No annotation files found in labels directory")
        else:
            print("ℹ️  No labels directory found")
        
        # Auto-detection is the primary option if annotations exist
        if len(annotation_files) > 0:
            print(f"\n🎯 SMART RENAMING OPTIONS:")
            print("1: Auto-detect species from annotations ⭐ (Recommended)")
            print("2: Manual species selection from YAML")
            print("3: Randomize all filenames")
            
            choice = input("➤ Choose option (1, 2, or 3, default: 1): ").strip()
            if not choice:
                choice = '1'  # Default to auto-detection
                
        else:
            print(f"\n🔧 RENAMING OPTIONS:")
            print("1: By Species (manual selection from YAML)")
            print("2: Randomize all filenames")
            
            choice = input("➤ Choose option (1 or 2): ").strip()
        
        # Execute chosen option
        if choice == '1':
            if len(annotation_files) > 0:
                # Auto-detection
                print(f"\n🤖 AUTOMATIC SPECIES DETECTION")
                print(f"   Analyzing annotation files to detect species...")
                print(f"   Will rename files like: Elk_001.jpg, Bear_023.jpg")
                print(f"   📋 Annotation files to analyze: {len(annotation_files)}")
                
                confirm = input(f"\n🚀 Proceed with smart auto-detection? (y/n, default: y): ").strip().lower()
                if not confirm or confirm in ['y', 'yes']:
                    self.auto_rename_by_species_from_annotations(directory)
                else:
                    print("❌ Auto-detection cancelled")
            else:
                # Manual selection (fallback when no annotations)
                self.show_species_menu()
                try:
                    species_id = int(input("Enter species ID number: ").strip())
                    self.rename_by_species(directory, species_id)
                except ValueError:
                    print("❌ Invalid species ID")
                    
        elif choice == '2':
            if len(annotation_files) > 0:
                # Manual selection (when annotations exist but user chooses manual)
                self.show_species_menu()
                try:
                    species_id = int(input("Enter species ID number: ").strip())
                    self.rename_by_species(directory, species_id)
                except ValueError:
                    print("❌ Invalid species ID")
            else:
                # Randomize (when no annotations)
                confirm = input("⚠️  This will randomize ALL filenames. Continue? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    self.randomize_filenames(directory)
                else:
                    print("❌ Operation cancelled")
                    
        elif choice == '3' and len(annotation_files) > 0:
            # Randomize (when annotations exist)
            confirm = input("⚠️  This will randomize ALL filenames. Continue? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                self.randomize_filenames(directory)
            else:
                print("❌ Operation cancelled")
        else:
            print("❌ Invalid choice")
    
    def annotation_menu(self):
        """Image annotation submenu"""
        print("\n" + "-"*30)
        print("🏷️  IMAGE ANNOTATION WITH LABELIMG")
        print("-"*30)
        
        # Check if we have species loaded
        if not self.species_names:
            print("⚠️  No species classes loaded!")
            load_yaml = input("Load YAML file first? (y/n): ").strip().lower()
            if load_yaml in ['y', 'yes']:
                self.load_yaml_menu()
                if not self.species_names:
                    print("❌ Cannot proceed without species classes")
                    return
            else:
                print("❌ Cannot proceed without species classes")
                return
        
        print(f"✓ Using {len(self.species_names)} species classes from YAML")
        
        # Choose annotation target
        print("\n📂 ANNOTATION TARGET:")
        print("1: Annotate single image file")
        print("2: Annotate all images in directory")
        
        target_choice = input("➤ Choose option (1 or 2): ").strip()
        
        if target_choice == '1':
            # Single image
            image_path = self.get_file_path("Enter path to image file:")
            if not image_path or not os.path.exists(image_path):
                print("❌ Invalid image path")
                return
            
            self.launch_labelimg(image_path, is_directory=False)
            
        elif target_choice == '2':
            # Directory of images - handle both YOLO structure and simple structure
            images_dir = self.get_file_path("Enter directory containing images (or dataset root with images/ folder):")
            if not images_dir or not os.path.exists(images_dir):
                print("❌ Invalid directory path")
                return
            
            # Check if this is a YOLO dataset structure
            yolo_images_dir = Path(images_dir) / "images"
            yolo_labels_dir = Path(images_dir) / "labels"
            
            if yolo_images_dir.exists() and yolo_labels_dir.exists():
                print("✅ Detected YOLO dataset structure!")
                print(f"   📁 Images: {yolo_images_dir}")
                print(f"   📄 Labels: {yolo_labels_dir}")
                actual_images_dir = str(yolo_images_dir)
                actual_labels_dir = str(yolo_labels_dir)
            else:
                print("✅ Using simple directory structure")
                actual_images_dir = images_dir
                actual_labels_dir = os.path.join(images_dir, "labels")
            
            # Show image statistics first
            print(f"\n📊 Analyzing directory...")
            total_images, annotation_count = self._count_images_simple(actual_images_dir, actual_labels_dir)
            
            if total_images == 0:
                print("❌ No images found in directory")
                return
            
            # Ask about resuming from last annotated
            print(f"\n🔄 ANNOTATION OPTIONS:")
            if annotation_count > 0:
                print("1: Start from beginning")
                print("2: Resume from last annotated image ⭐ (Recommended)")
                resume_choice = input("➤ Choose option (1 or 2, default: 2): ").strip()
                resume_from_last = resume_choice != '1'
            else:
                print("✓ No existing annotations found - will start from first image")
                resume_from_last = False
            
            # For YOLO structure, we need to pass the images directory but use the dataset root for labels
            if yolo_images_dir.exists() and yolo_labels_dir.exists():
                self.launch_labelimg_yolo_structure(actual_images_dir, actual_labels_dir, resume_from_last)
            else:
                self.launch_labelimg(actual_images_dir, is_directory=True, resume_from_last=resume_from_last)
        
        else:
            print("❌ Invalid choice")
    
    def launch_labelimg_yolo_structure(self, images_dir, labels_dir, resume_from_last=False):
        """Launch labelImg for YOLO dataset structure with separate images/ and labels/ folders"""
        
        # Check if labelImg is installed
        try:
            result = subprocess.run(['labelImg', '--help'], capture_output=True, text=True)
        except FileNotFoundError:
            print("❌ labelImg not found!")
            print("   Install it with: pip install labelImg")
            print("   Or: conda install -c conda-forge labelimg")
            return False
        
        # Create labels directory if it doesn't exist
        os.makedirs(labels_dir, exist_ok=True)
        
        # Create predefined classes file
        classes_file = os.path.join(labels_dir, "predefined_classes.txt")
        if not self.create_classes_file(classes_file):
            return False
        
        # Save this directory for memory
        self.save_last_directory(images_dir)
        
        # Handle resume functionality
        if resume_from_last:
            # Find the exact image to resume from
            resume_image = self.find_resume_image(images_dir, labels_dir)
            if resume_image:
                temp_images_dir, temp_labels_dir = self.create_resume_workspace(images_dir, labels_dir, resume_image)
                
                if temp_images_dir and temp_labels_dir:
                    # Launch labelImg on temporary workspace
                    temp_classes_file = os.path.join(temp_labels_dir, "predefined_classes.txt")
                    cmd = ['labelImg', temp_images_dir, temp_classes_file, temp_labels_dir]
                    
                    print(f"\n🚀 Launching labelImg from resume point...")
                    print(f"   📍 Starting directly at your next image!")
                    
                    print("\n💡 labelImg Controls:")
                    print("   • W = Draw bounding box")
                    print("   • A/D = Previous/Next image") 
                    print("   • Ctrl+S = Save annotation")
                    print("   • Space = Mark as verified")
                    print("   • Del = Delete selected box")
                    
                    try:
                        # Launch labelImg and wait for it to close
                        process = subprocess.run(cmd)
                        print("\n🔄 labelImg closed. Processing annotations...")
                        
                    except Exception as e:
                        print(f"✗ Error running labelImg: {e}")
                        
                    finally:
                        # Always cleanup, even if there was an error
                        print("🧹 Cleaning up workspace...")
                        self.cleanup_and_merge_annotations(labels_dir, temp_labels_dir, temp_images_dir)
                    
                    return True
                else:
                    print("⚠️  Could not create workspace, falling back to normal mode")
        
        # Normal mode (non-resume or fallback)
        cmd = ['labelImg', images_dir, classes_file, labels_dir]
        
        print(f"\n🚀 Launching labelImg...")
        print(f"   Images: {images_dir}")
        print(f"   Labels: {labels_dir}")
        print(f"   Classes: {classes_file}")
        
        print("\n💡 labelImg Controls:")
        print("   • W = Draw bounding box")
        print("   • A/D = Previous/Next image") 
        print("   • Ctrl+S = Save annotation")
        print("   • Space = Mark as verified")
        print("   • Del = Delete selected box")
        
        try:
            # Launch labelImg
            process = subprocess.Popen(cmd)
            print("✅ labelImg launched successfully!")
            return True
        except Exception as e:
            print(f"✗ Error launching labelImg: {e}")
            return False
    
    def organize_annotations_menu(self):
        """Menu for organizing annotated files"""
        print("\n" + "-"*30)
        print("📦 ORGANIZE ANNOTATED FILES")
        print("-"*30)
        
        # Get source directory
        source_dir = self.get_file_path("Enter directory containing images and labels:")
        if not source_dir or not os.path.exists(source_dir):
            print("❌ Invalid directory path")
            return
        
        # Check what we'll be moving
        labels_dir = os.path.join(source_dir, "labels")
        if not os.path.exists(labels_dir):
            print(f"❌ No 'labels' folder found in {source_dir}")
            print("   Make sure you're pointing to the directory that contains both images and a 'labels' folder")
            return
        
        # Count annotations
        label_files = list(Path(labels_dir).glob("*.txt"))
        label_files = [f for f in label_files if f.name != "predefined_classes.txt"]
        
        print(f"\n📊 PREVIEW:")
        print(f"   Source directory: {source_dir}")
        print(f"   Annotated files found: {len(label_files)}")
        print(f"   Will create: {source_dir}/annotated/")
        print(f"                ├── images/     (annotated images)")
        print(f"                └── labels/     (annotation files)")
        
        if len(label_files) == 0:
            print("❌ No annotation files found to move")
            return
        
        # Confirm action
        confirm = input(f"\n🔄 Move {len(label_files)} annotated files to 'annotated' folder? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("❌ Operation cancelled")
            return
        
        # Perform the move
        self.move_annotated_files(source_dir)
    
    def load_yaml_menu(self):
        """YAML loading submenu"""
        print("\n" + "-"*30)
        print("📄 LOAD YAML CONFIGURATION")
        print("-"*30)
        yaml_path = self.get_file_path("Enter YAML file path:", self.default_yaml_path)
        self.load_yaml(yaml_path)
    
    def main_menu(self):
        """Main interactive menu"""
        print("\n" + "="*50)
        print("🎥 TRAIL CAM FRAME EXTRACTOR & FILE RENAMER 🎥")
        print("="*50)
        
        while True:
            print("\n📋 MAIN MENU:")
            print("1: Extract frames from ALL videos in directory")
            print("2: Rename photos or videos in directory")
            print("3: Annotate images with labelImg")
            print("4: Move annotated files to separate folder")
            print("5: Analyze dataset (training or simple structure)")
            print("6: Manual cleanup of temporary workspaces")
            print("7: Load different YAML file")
            print("8: Diagnose file matching issues 🔬")
            print("9: Exit")
            
            choice = input("\n➤ Choose option (1-9): ").strip()
            
            if choice == '1':
                self.extract_frames_menu()
            elif choice == '2':
                self.rename_menu()
            elif choice == '3':
                self.annotation_menu()
            elif choice == '4':
                self.organize_annotations_menu()
            elif choice == '5':
                self.dataset_analysis_menu()
            elif choice == '6':
                self.manual_cleanup_menu()
            elif choice == '7':
                self.load_yaml_menu()
            elif choice == '8':
                directory = self.get_file_path("Enter dataset directory to diagnose:")
                if directory:
                    self.diagnose_file_matching(directory)
            elif choice == '9':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-9.")

if __name__ == "__main__":
    processor = TrailCamProcessor()
    processor.main_menu()