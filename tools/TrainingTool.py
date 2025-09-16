import os
import yaml
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import sys
from tqdm import tqdm
import time

class YOLOTrainingTool:
    def __init__(self):
        # Project management
        self.project_file = "yolo_project.json"
        self.projects_dir = Path("yolo_projects")
        self.projects_dir.mkdir(exist_ok=True)
        
        # Default paths
        self.default_paths = {
            'images_dir': r"C:\Users\Coastal_wolf\Desktop\WolfVueTrain\images",
            'labels_dir': r"C:\Users\Coastal_wolf\Desktop\WolfVueTrain\labels", 
            'yaml_path': r"C:\Users\Coastal_wolf\Documents\GitHub\TrailCamAi\Datasets\Scripts\WlfCamData.yaml"
        }
        
        # Current project state
        self.project_state = {
            'project_name': None,
            'created_date': None,
            'last_modified': None,
            'paths': self.default_paths.copy(),
            'species_names': {},
            'dataset_stats': {},
            'selected_classes': {},
            'target_counts': {},
            'dataset_output_dir': None,
            'training_params': {
                'epochs': 100,
                'batch_size': 16,
                'image_size': 640,
                'model_size': 'yolov8s.pt',
                'learning_rate': 0.01,
                'patience': 50,
                'save_period': 10,
                'workers': 8,
                'device': 'auto',
                'augment': True,
                'project_name': 'trail_cam_models'
            }
        }
        
        # FIXED: Load or create project (don't create directly)
        self.load_or_create_project()
    
    def load_or_create_project(self):
        """Load existing project or create new one"""
        print("\n" + "="*80)
        print("YOLO TRAINING TOOL - PROJECT MANAGER")
        print("="*80)
        
        # Check for existing projects
        existing_projects = list(self.projects_dir.glob("*.json"))
        
        if existing_projects:
            print(f"\nFound {len(existing_projects)} existing project(s):")
            for i, project_file in enumerate(existing_projects, 1):
                try:
                    with open(project_file, 'r') as f:
                        project_data = json.load(f)
                    name = project_data.get('project_name', project_file.stem)
                    modified = project_data.get('last_modified', 'Unknown')
                    print(f"  {i}: {name} (Modified: {modified})")
                except Exception:
                    print(f"  {i}: {project_file.stem} (Corrupted)")
            
            print(f"  {len(existing_projects) + 1}: Create new project")
            
            while True:
                try:
                    choice = input(f"\nSelect project (1-{len(existing_projects) + 1}): ").strip()
                    choice_num = int(choice)
                    
                    if 1 <= choice_num <= len(existing_projects):
                        # Load existing project
                        project_file = existing_projects[choice_num - 1]
                        if self.load_project(project_file):
                            print(f"Loaded project: {self.project_state['project_name']}")
                            return
                        else:
                            print("Failed to load project, creating new one...")
                            break
                    elif choice_num == len(existing_projects) + 1:
                        # Create new project
                        break
                    else:
                        print("Invalid selection")
                except ValueError:
                    print("Please enter a valid number")
        
        # Create new project
        self.create_new_project()

    def create_new_project(self):
        """Create a new project"""
        project_name = input("\nEnter project name: ").strip()
        if not project_name:
            project_name = f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Sanitize project name for filename
        safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        
        self.project_state['project_name'] = project_name
        self.project_state['created_date'] = datetime.now().isoformat()
        self.project_state['last_modified'] = datetime.now().isoformat()
        
        self.project_file = self.projects_dir / f"{safe_name}.json"
        self.save_project()
        
        print(f"Created new project: {project_name}")

    def _fix_integer_keys(self, state):
        """Fix JSON serialization converting integer keys to strings"""
        
        # Fix dataset_stats class_counts
        if 'dataset_stats' in state and 'class_counts' in state['dataset_stats']:
            class_counts = state['dataset_stats']['class_counts']
            if class_counts:
                # Convert string keys back to integers
                fixed_class_counts = {}
                for key, value in class_counts.items():
                    try:
                        fixed_class_counts[int(key)] = value
                    except (ValueError, TypeError):
                        fixed_class_counts[key] = value  # Keep original if conversion fails
                state['dataset_stats']['class_counts'] = fixed_class_counts
        
        # Fix selected_classes keys
        if 'selected_classes' in state:
            selected_classes = state['selected_classes']
            if selected_classes:
                fixed_selected_classes = {}
                for key, value in selected_classes.items():
                    try:
                        fixed_selected_classes[int(key)] = value
                    except (ValueError, TypeError):
                        fixed_selected_classes[key] = value
                    state['selected_classes'] = fixed_selected_classes
        
        # Fix target_counts keys
        if 'target_counts' in state:
            target_counts = state['target_counts']
            if target_counts:
                fixed_target_counts = {}
                for key, value in target_counts.items():
                    try:
                        fixed_target_counts[int(key)] = value
                    except (ValueError, TypeError):
                        fixed_target_counts[key] = value
                state['target_counts'] = fixed_target_counts
        
        # Fix class_mapping keys
        if 'class_mapping' in state:
            class_mapping = state['class_mapping']
            if class_mapping:
                fixed_class_mapping = {}
                for key, value in class_mapping.items():
                    try:
                        fixed_class_mapping[int(key)] = value
                    except (ValueError, TypeError):
                        fixed_class_mapping[key] = value
                state['class_mapping'] = fixed_class_mapping

    def debug_class_ids(self):
        """Debug method to check class ID types and values"""
        print("\n🔍 DEBUG: Checking class ID data types and values...")

        # Check dataset_stats
        stats = self.project_state.get('dataset_stats', {})
        class_counts = stats.get('class_counts', {})
        print(f"\nDataset stats class_counts:")
        for k, v in list(class_counts.items())[:5]:  # Show first 5
            print(f"  Key: {repr(k)} (type: {type(k).__name__}) -> Value: {v}")

        # Check selected_classes
        selected = self.project_state.get('selected_classes', {})
        print(f"\nSelected classes:")
        for k, v in list(selected.items())[:5]:  # Show first 5
            print(f"  Key: {repr(k)} (type: {type(k).__name__}) -> Name: {v.get('name', 'Unknown')}")

        # Check target_counts
        targets = self.project_state.get('target_counts', {})
        print(f"\nTarget counts:")
        for k, v in list(targets.items())[:5]:  # Show first 5
            print(f"  Key: {repr(k)} (type: {type(k).__name__}) -> Count: {v}")

        # Check actual annotation files
        paths = self.project_state['paths']
        labels_dir = Path(paths['labels_dir'])
        print(f"\nSample annotation file content:")

        sample_files = list(labels_dir.glob("*.txt"))[:3]  # Check first 3 files
        for ann_file in sample_files:
            if ann_file.name.lower() not in {"predefined_classes.txt", "classes.txt", "obj.names", "obj.data"}:
                print(f"\nFile: {ann_file.name}")
                try:
                    with open(ann_file, 'r') as f:
                        lines = f.readlines()[:3]  # First 3 lines
                        for i, line in enumerate(lines):
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) >= 5:
                                    print(f"  Line {i+1}: class_id = {repr(parts[0])} (type: {type(parts[0]).__name__})")
                except Exception as e:
                    print(f"  Error reading file: {e}")
                break

    def load_project(self, project_file):
        """Load project from file with proper integer key handling"""
        try:
            with open(project_file, 'r') as f:
                loaded_state = json.load(f)
        
            # Fix JSON serialization issues with integer keys
            self._fix_integer_keys(loaded_state)
        
            # Merge loaded state with current state (preserve structure)
            self.project_state.update(loaded_state)
            self.project_file = project_file
        
            # Load YAML if path exists
            if self.project_state['paths']['yaml_path']:
                self.load_yaml_config()
        
            return True
        except Exception as e:
            print(f"Error loading project: {e}")
            return False
    
    def save_project(self):
        """Save current project state"""
        try:
            self.project_state['last_modified'] = datetime.now().isoformat()
            with open(self.project_file, 'w') as f:
                json.dump(self.project_state, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save project: {e}")
    
    def format_path(self, path):
        """Format path for clean display"""
        if not path:
            return "Not set"
        
        path_obj = Path(path)
        if len(str(path)) > 60:
            parts = path_obj.parts
            if len(parts) > 3:
                return f"{parts[0]}\\...\\{parts[-2]}\\{parts[-1]}"
        return str(path)
    
    def get_user_path(self, prompt, current_path, path_type="directory"):
        """Get path from user with validation"""
        print(f"\n{prompt}")
        print(f"Current: {self.format_path(current_path)}")
        
        user_input = input("New path (Enter to keep current): ").strip()
        if user_input:
            user_input = user_input.replace('"', '').replace("'", "")
            path_obj = Path(user_input)
            
            if path_type == "directory":
                if path_obj.is_dir():
                    return str(path_obj)
                else:
                    print(f"Error: Directory does not exist: {user_input}")
                    create = input("Create directory? (y/n): ").strip().lower()
                    if create in ['y', 'yes']:
                        try:
                            path_obj.mkdir(parents=True, exist_ok=True)
                            return str(path_obj)
                        except Exception as e:
                            print(f"Error creating directory: {e}")
                            return current_path
                    return current_path
            elif path_type == "file":
                if path_obj.is_file():
                    return str(path_obj)
                else:
                    print(f"Error: File does not exist: {user_input}")
                    return current_path
        
        return current_path
    
    def load_yaml_config(self):
        """Load species configuration from YAML with error handling"""
        yaml_path = self.project_state['paths']['yaml_path']
        if not yaml_path or not Path(yaml_path).exists():
            print("YAML configuration file not found")
            return False
        
        try:
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)
                self.project_state['species_names'] = data.get('names', {})
            
            print(f"✓ Loaded {len(self.project_state['species_names'])} species classes")
            self.save_project()  # Auto-save project state
            return True
        except Exception as e:
            print(f"Error loading YAML: {e}")
            return False
    
    def analyze_dataset(self, show_progress=True):
        """Analyze dataset with progress tracking"""
        paths = self.project_state['paths']
        if not paths['images_dir'] or not paths['labels_dir']:
            print("Error: Image or label paths not set")
            return False
        
        print("Analyzing dataset...")
        
        # Get all annotation files
        labels_dir = Path(paths['labels_dir'])
        annotation_files = []
        
        for txt_file in labels_dir.glob("*.txt"):
            # Skip system files
            if txt_file.name.lower() not in {"predefined_classes.txt", "classes.txt", "obj.names", "obj.data"}:
                annotation_files.append(txt_file)
        
        if not annotation_files:
            print("Error: No annotation files found")
            return False
        
        # Count annotations with progress
        class_counts = Counter()
        total_annotations = 0
        valid_files = 0
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF']
        
        iterator = tqdm(annotation_files, desc="Analyzing files") if show_progress else annotation_files
        
        for ann_file in iterator:
            try:
                # Check if corresponding image exists
                image_stem = ann_file.stem
                has_image = False
                
                for ext in image_extensions:
                    image_path = Path(paths['images_dir']) / f"{image_stem}{ext}"
                    if image_path.exists():
                        has_image = True
                        break
                
                if not has_image:
                    continue
                
                # Parse annotations
                with open(ann_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                try:
                                    class_id = int(parts[0])
                                    class_counts[class_id] += 1
                                    total_annotations += 1
                                except (ValueError, IndexError):
                                    continue
                
                valid_files += 1
                
            except Exception:
                continue
        
        self.project_state['dataset_stats'] = {
            'class_counts': dict(class_counts),  # Convert Counter to dict for JSON serialization
            'total_annotations': total_annotations,
            'valid_files': valid_files,
            'total_files': len(annotation_files),
            'analysis_date': datetime.now().isoformat()
        }
        
        self.save_project()  # Auto-save
        return True
    
    def display_dataset_analysis(self):
        """Display comprehensive dataset analysis"""
        stats = self.project_state['dataset_stats']
        if not stats:
            print("No dataset analysis available. Run analysis first.")
            return
        
        print("\n" + "="*80)
        print("DATASET ANALYSIS RESULTS")
        print("="*80)
        
        print(f"Dataset Overview:")
        print(f"  Total annotation files: {stats['total_files']}")
        print(f"  Valid image-annotation pairs: {stats['valid_files']}")
        print(f"  Total object annotations: {stats['total_annotations']}")
        if stats['valid_files'] > 0:
            print(f"  Average objects per image: {stats['total_annotations'] / stats['valid_files']:.1f}")
        
        class_counts = stats['class_counts']
        if not class_counts:
            print("No valid annotations found")
            return
        
        # Sort classes by count (largest first)
        sorted_classes = sorted(class_counts.items(), key=lambda x: (int(x[0]), x[1]), reverse=False)
        sorted_classes = sorted(sorted_classes, key=lambda x: x[1], reverse=True)  # Sort by count
        
        print(f"\nClass Distribution (Largest to Smallest):")
        print("-" * 60)
        print(f"{'Class ID':<10} {'Species Name':<20} {'Count':<10} {'Percentage':<12}")
        print("-" * 60)
        
        species_names = self.project_state['species_names']
        for class_id, count in sorted_classes:
            class_id = int(class_id)  # Ensure int for lookup
            species_name = species_names.get(class_id, f"Unknown_{class_id}")
            percentage = (count / stats['total_annotations']) * 100
            print(f"{class_id:<10} {species_name:<20} {count:<10} {percentage:<12.1f}%")
        
        # Dataset balance analysis
        self._show_balance_analysis(sorted_classes)
        self._show_recommendations(sorted_classes)
    
    def _show_balance_analysis(self, sorted_classes):
        """Show dataset balance analysis"""
        print(f"\nDataset Balance Analysis:")
        if len(sorted_classes) > 1:
            max_count = sorted_classes[0][1]
            min_count = sorted_classes[-1][1]
            balance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            print(f"  Most common class: {max_count} annotations")
            print(f"  Least common class: {min_count} annotations")
            print(f"  Balance ratio: {balance_ratio:.1f}:1")
            
            if balance_ratio > 10:
                print("  ⚠️  WARNING: Dataset is highly imbalanced!")
            elif balance_ratio > 3:
                print("  ⚠️  NOTE: Dataset has moderate imbalance")
            else:
                print("  ✓ GOOD: Dataset is relatively balanced")
    
    def _show_recommendations(self, sorted_classes):
        """Show training recommendations"""
        print(f"\nRecommendations:")
        species_names = self.project_state['species_names']
        
        for class_id, count in sorted_classes:
            class_id = int(class_id)
            species_name = species_names.get(class_id, f"Unknown_{class_id}")
            if count < 50:
                print(f"  ⚠️  {species_name}: Only {count} samples (minimum 50-100 recommended)")
            elif count < 100:
                print(f"  ⚠️  {species_name}: {count} samples (100+ recommended for good results)")
    
    def select_classes_for_training(self):
        """Interactive class selection with smart defaults"""
        stats = self.project_state['dataset_stats']
        if not stats:
            print("Error: No dataset analysis available")
            return False
        
        print("\n" + "="*80)
        print("CLASS SELECTION FOR TRAINING")
        print("="*80)
        
        class_counts = stats['class_counts']
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        species_names = self.project_state['species_names']
        
        print("Select classes to include in training:")
        print("(Press 'a' for auto-select recommended classes, or choose manually)")
        print()
        
        # Auto-select option
        auto_choice = input("Auto-select classes with 50+ samples? (y/n/manual): ").strip().lower()
        
        selected_classes = {}
        
        if auto_choice in ['y', 'yes']:
            # Auto-select classes with sufficient samples
            for class_id, count in sorted_classes:
                class_id = int(class_id)
                species_name = species_names.get(class_id, f"Unknown_{class_id}")
                
                included = count >= 50  # Auto-include if 50+ samples
                selected_classes[class_id] = {
                    'name': species_name,
                    'count': count,
                    'included': included
                }
                
                status = "✓ INCLUDED" if included else "✗ EXCLUDED"
                reason = "sufficient samples" if included else "too few samples"
                print(f"  {species_name}: {count} samples - {status} ({reason})")
        else:
            # Manual selection
            for class_id, count in sorted_classes:
                class_id = int(class_id)
                species_name = species_names.get(class_id, f"Unknown_{class_id}")
                
                # Provide recommendation
                if count < 50:
                    recommendation = "NOT RECOMMENDED (too few samples)"
                elif count < 100:
                    recommendation = "CAUTION (may need more samples)"
                else:
                    recommendation = "RECOMMENDED"
                
                print(f"\nClass {class_id}: {species_name} ({count} samples) - {recommendation}")
                
                while True:
                    default = "y" if count >= 50 else "n"
                    choice = input(f"Include {species_name}? (y/n, default {default}): ").strip().lower()
                    
                    if not choice:
                        choice = default
                    
                    if choice in ['y', 'yes']:
                        selected_classes[class_id] = {
                            'name': species_name,
                            'count': count,
                            'included': True
                        }
                        break
                    elif choice in ['n', 'no']:
                        selected_classes[class_id] = {
                            'name': species_name,
                            'count': count,
                            'included': False
                        }
                        break
                    else:
                        print("Please enter 'y' or 'n'")
        
        self.project_state['selected_classes'] = selected_classes
        
        # Show selection summary
        included = {k: v for k, v in selected_classes.items() if v['included']}
        excluded = {k: v for k, v in selected_classes.items() if not v['included']}
        
        print(f"\n📊 Selection Summary:")
        print(f"  ✓ Included classes: {len(included)}")
        for class_id, info in included.items():
            print(f"    {class_id}: {info['name']} ({info['count']} samples)")
        
        if excluded:
            print(f"  ✗ Excluded classes: {len(excluded)}")
            for class_id, info in excluded.items():
                print(f"    {class_id}: {info['name']} ({info['count']} samples)")
        
        self.save_project()  # Auto-save
        return len(included) > 0
    
    def set_balanced_dataset_targets(self):
        """Set target counts with smart defaults"""
        selected_classes = self.project_state['selected_classes']
        included_classes = {k: v for k, v in selected_classes.items() if v['included']}
        
        if not included_classes:
            print("Error: No classes selected")
            return False
        
        print("\n" + "="*80)
        print("BALANCED DATASET CONFIGURATION")
        print("="*80)
        
        # Show current counts and suggest strategy
        counts = [info['count'] for info in included_classes.values()]
        min_count = min(counts)
        max_count = max(counts)
        avg_count = sum(counts) // len(counts)
        
        print("Current class distribution:")
        for class_id, info in included_classes.items():
            print(f"  {info['name']}: {info['count']} samples")
        
        print(f"\nStatistics:")
        print(f"  Minimum: {min_count}, Maximum: {max_count}, Average: {avg_count}")
        
        # Suggest optimal strategy
        if max_count / min_count > 3:
            suggested_target = min_count
            suggestion = f"Recommended: {suggested_target} (balance to smallest class)"
        else:
            suggested_target = avg_count
            suggestion = f"Recommended: {suggested_target} (use average)"
        
        print(f"\nBalancing Options:")
        print(f"1: Use suggested target ({suggestion})")
        print(f"2: Set custom target for all classes")
        print(f"3: Set individual targets")
        print(f"4: Use current counts (no balancing)")
        
        while True:
            choice = input("Choose strategy (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                break
            print("Please enter 1-4")
        
        target_counts = {}
        
        if choice == '1':
            # Use suggested target
            for class_id in included_classes.keys():
                target_counts[class_id] = suggested_target
            print(f"Set target to {suggested_target} for all classes")
        
        elif choice == '2':
            # Custom target for all
            while True:
                try:
                    target = input(f"Enter target count for all classes: ").strip()
                    target_count = int(target)
                    
                    if target_count <= 0:
                        print("Target must be positive")
                        continue
                    
                    for class_id in included_classes.keys():
                        target_counts[class_id] = target_count
                    break
                except ValueError:
                    print("Please enter a valid number")
        
        elif choice == '3':
            # Individual targets
            for class_id, info in included_classes.items():
                while True:
                    try:
                        prompt = f"Target for {info['name']} (available: {info['count']}): "
                        target = input(prompt).strip()
                        target_count = int(target) if target else info['count']
                        
                        if target_count <= 0:
                            print("Target must be positive")
                            continue
                        
                        target_counts[class_id] = target_count
                        break
                    except ValueError:
                        print("Please enter a valid number")
        
        else:
            # Use current counts
            for class_id, info in included_classes.items():
                target_counts[class_id] = info['count']
        
        self.project_state['target_counts'] = target_counts
        
        # Show final targets
        print(f"\n📋 Final Target Counts:")
        total_target = 0
        for class_id, target in target_counts.items():
            class_name = included_classes[class_id]['name']
            available = included_classes[class_id]['count']
            status = "✓ OK" if target <= available else f"⚠️ NEED {target - available} MORE"
            print(f"  {class_name}: {target} (available: {available}) - {status}")
            total_target += target
        
        print(f"  📊 Total target samples: {total_target}")
        
        self.save_project()  # Auto-save
        return True
    
    def create_balanced_dataset(self, output_dir=None):
        """Create balanced dataset with enhanced file copying and automatic class remapping"""
        target_counts = self.project_state['target_counts']
        if not target_counts:
            print("Error: No target counts set")
            return False
        
        # Use project's dataset directory or ask for new one
        if not output_dir:
            if self.project_state['dataset_output_dir']:
                use_existing = input(f"Use existing dataset directory? ({self.format_path(self.project_state['dataset_output_dir'])}) (y/n): ").strip().lower()
                if use_existing in ['y', 'yes']:
                    output_dir = self.project_state['dataset_output_dir']
            
            if not output_dir:
                output_dir = input("Enter output directory for dataset: ").strip()
                if not output_dir:
                    print("Output directory required")
                    return False
                output_dir = output_dir.replace('"', '').replace("'", "")
        
        # Save the output directory for future use
        self.project_state['dataset_output_dir'] = output_dir
        self.save_project()
        
        print(f"\n🔄 Creating balanced dataset in: {self.format_path(output_dir)}")
        print(f"📁 Source data will remain completely unchanged")
        
        # Create class index mapping for consecutive numbering
        selected_class_ids = sorted(target_counts.keys())
        class_mapping = {old_id: new_id for new_id, old_id in enumerate(selected_class_ids)}
        
        print(f"\n🔄 Class Index Remapping (applied only to copied files):")
        for old_id, new_id in class_mapping.items():
            class_name = self.project_state['selected_classes'][old_id]['name']
            print(f"  {class_name}: {old_id} → {new_id}")
        
        # Save mapping to project state for reference
        self.project_state['class_mapping'] = class_mapping
        self.save_project()
        
        # Create output structure
        output_path = Path(output_dir)
        dirs_to_create = [
            output_path / "train" / "images",
            output_path / "train" / "labels", 
            output_path / "val" / "images",
            output_path / "val" / "labels",
            output_path / "test" / "images",
            output_path / "test" / "labels"
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Get split ratios
        train_pct, val_pct, test_pct = self._get_split_ratios()
        
        # Process each class with progress tracking
        print(f"\n🔄 Copying and remapping files...")
        print(f"📝 Note: Original files unchanged, remapping applied only to copies")
        
        total_files_to_copy = sum(min(info['count'], target_counts[class_id]) 
                                 for class_id, info in self.project_state['selected_classes'].items() 
                                 if info['included'])
        
        copied_files = 0
        class_stats = {}
        
        with tqdm(total=total_files_to_copy * 2, desc="Creating dataset", unit="files") as pbar:  # *2 for images and labels
            for class_id, target_count in target_counts.items():
                class_info = self.project_state['selected_classes'][class_id]
                class_name = class_info['name']
                
                # Collect files for this class
                class_files = self._collect_class_files(class_id, target_count)
                actual_count = len(class_files)
                
                if actual_count == 0:
                    print(f"⚠️ No files found for {class_name}")
                    continue
                
                # Split files
                train_files, val_files, test_files = self._split_files(
                    class_files, actual_count, train_pct, val_pct, test_pct
                )
                
                # Copy files with progress and class remapping
                copied_count = self._copy_class_files(
                    train_files, val_files, test_files, 
                    output_path, class_mapping, pbar
                )
                
                copied_files += copied_count
                
                # Use NEW class ID for stats
                new_class_id = class_mapping[class_id]
                class_stats[new_class_id] = {
                    'name': class_name,
                    'train': len(train_files),
                    'val': len(val_files), 
                    'test': len(test_files),
                    'total': actual_count,
                    'original_id': class_id
                }
        
        # Generate summary and YAML
        self._generate_dataset_summary(output_dir, class_stats, copied_files)
        yaml_path = self._create_training_yaml(output_path, class_stats)
        
        # Validate the created dataset
        if yaml_path:
            self._validate_dataset_consistency(output_path, len(class_stats))
        
        print(f"\n✅ Dataset creation complete with automatic class remapping!")
        print(f"📋 Classes have been renumbered 0-{len(class_stats)-1} for YOLO compatibility")
        print(f"📁 Original source data remains completely unchanged")
        return True
    
    def _get_split_ratios(self):
        """Get train/val/test split ratios from user"""
        print(f"\n📊 Dataset Split Configuration:")
        
        # Offer common presets
        print("Preset options:")
        print("1: 70/20/10 (Standard)")
        print("2: 80/15/5 (More training data)")
        print("3: 60/25/15 (More validation/test)")
        print("4: Custom split")
        
        choice = input("Choose preset (1-4): ").strip()
        
        if choice == '1':
            return 70, 20, 10
        elif choice == '2':
            return 80, 15, 5
        elif choice == '3':
            return 60, 25, 15
        else:
            while True:
                try:
                    train_pct = float(input("Training percentage: ") or "70")
                    val_pct = float(input("Validation percentage: ") or "20")
                    test_pct = float(input("Test percentage: ") or "10")
                    
                    if abs(train_pct + val_pct + test_pct - 100) > 0.1:
                        print("Error: Percentages must sum to 100")
                        continue
                    
                    return train_pct, val_pct, test_pct
                except ValueError:
                    print("Please enter valid numbers")
    
    def _collect_class_files(self, class_id, target_count):
        """Collect image-annotation file pairs for a specific class"""
        paths = self.project_state['paths']
        labels_dir = Path(paths['labels_dir'])
        images_dir = Path(paths['images_dir'])
        
        # CRITICAL FIX: Ensure class_id is an integer (fixes JSON serialization issue)
        class_id = int(class_id)
        
        print(f"🔍 Looking for class {class_id} (type: {type(class_id).__name__})")  # Debug
        
        class_files = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF']
        
        files_checked = 0
        files_with_class = 0
        
        for ann_file in labels_dir.glob("*.txt"):
            if ann_file.name.lower() in {"predefined_classes.txt", "classes.txt", "obj.names", "obj.data"}:
                continue
            
            files_checked += 1
            
            try:
                # Check if this annotation file contains the target class
                has_target_class = False
                with open(ann_file, 'r', encoding='utf-8') as f:  # Added encoding
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                try:
                                    file_class_id = int(parts[0])
                                    # FIXED: Both sides are now definitely integers
                                    if file_class_id == class_id:
                                        has_target_class = True
                                        break
                                except ValueError:
                                    # Skip invalid class ID format
                                    continue
                
                if not has_target_class:
                    continue
                
                files_with_class += 1
                
                # Find corresponding image
                image_stem = ann_file.stem
                image_found = False
                for ext in image_extensions:
                    image_path = images_dir / f"{image_stem}{ext}"
                    if image_path.exists():
                        class_files.append((image_path, ann_file))
                        image_found = True
                        break
                
                if not image_found:
                    print(f"      ⚠️ No corresponding image found for {ann_file.name}")
                    
            except Exception as e:
                print(f"      ⚠️ Error processing {ann_file.name}: {e}")
                continue
        
        print(f"    📊 Class {class_id}: Checked {files_checked} files, found {files_with_class} with this class, {len(class_files)} valid pairs")
        
        # Remove duplicates and limit to target count
        class_files = list(set(class_files))
        if len(class_files) > target_count:
            class_files = random.sample(class_files, target_count)
        
        return class_files
    
    def _split_files(self, class_files, actual_count, train_pct, val_pct, test_pct):
        """Split files into train/val/test sets"""
        random.shuffle(class_files)
        
        train_count = int(actual_count * train_pct / 100)
        val_count = int(actual_count * val_pct / 100)
        test_count = actual_count - train_count - val_count
        
        train_files = class_files[:train_count]
        val_files = class_files[train_count:train_count + val_count]
        test_files = class_files[train_count + val_count:]
        
        return train_files, val_files, test_files
    
    def _copy_class_files(self, train_files, val_files, test_files, output_path, class_mapping, pbar):
        """Copy files to appropriate directories with automatic class re-annotation"""
        copied_count = 0
        
        splits = [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files)
        ]
        
        for split_name, split_files in splits:
            img_dir = output_path / split_name / "images"
            lbl_dir = output_path / split_name / "labels"
            
            for img_path, ann_path in split_files:
                try:
                    # Copy image (unchanged)
                    shutil.copy2(img_path, img_dir / img_path.name)
                    pbar.update(1)
                    
                    # Copy and remap annotation file
                    self._copy_and_remap_annotation(ann_path, lbl_dir / ann_path.name, class_mapping)
                    pbar.update(1)
                    
                    copied_count += 1
                except Exception as e:
                    print(f"    ⚠️ Error copying {img_path.name}: {e}")
        
        return copied_count
    
    def _copy_and_remap_annotation(self, source_ann_path, dest_ann_path, class_mapping):
        """Copy annotation file and remap class indices to new consecutive numbering
        
        IMPORTANT: This creates a NEW annotation file with remapped class indices.
        The original annotation file remains completely unchanged.
        """
        try:
            # Read original annotation file (NEVER modified)
            with open(source_ann_path, 'r') as source_file:
                lines = source_file.readlines()
            
            # Process lines and remap class indices for the NEW file
            remapped_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            old_class_id = int(parts[0])
                            # Only include annotations for selected classes
                            if old_class_id in class_mapping:
                                new_class_id = class_mapping[old_class_id]
                                # Replace old class ID with new consecutive ID
                                parts[0] = str(new_class_id)
                                remapped_lines.append(' '.join(parts) + '\n')
                        except (ValueError, IndexError):
                            # Skip invalid lines
                            continue
            
            # Write NEW annotation file with remapped class indices
            with open(dest_ann_path, 'w') as dest_file:
                dest_file.writelines(remapped_lines)
                
        except Exception as e:
            # If remapping fails, copy original without modification and show warning
            shutil.copy2(source_ann_path, dest_ann_path)
            print(f"    ⚠️ Warning: Could not remap {source_ann_path.name}, copied original: {e}")
    
    def _generate_dataset_summary(self, output_dir, class_stats, copied_files):
        """Generate and display dataset creation summary"""
        print(f"\n" + "="*80)
        print("📊 DATASET CREATION SUMMARY")
        print("="*80)
        
        print(f"Output directory: {self.format_path(output_dir)}")
        print(f"Total files copied: {copied_files * 2} (images + labels)")
        
        # Emphasize data safety
        original_images = self.project_state['paths']['images_dir']
        original_labels = self.project_state['paths']['labels_dir']
        print(f"\n📁 DATA SAFETY:")
        print(f"  ✅ Original images directory: UNCHANGED")
        print(f"      {self.format_path(original_images)}")
        print(f"  ✅ Original labels directory: UNCHANGED")
        print(f"      {self.format_path(original_labels)}")
        print(f"  ✅ New dataset: Contains copies with remapped class indices")
        
        print(f"\nDataset Split Summary:")
        print(f"{'Class':<20} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
        print("-" * 55)
        
        total_train = total_val = total_test = total_all = 0
        for class_id, stats in class_stats.items():
            print(f"{stats['name']:<20} {stats['train']:<8} {stats['val']:<8} {stats['test']:<8} {stats['total']:<8}")
            total_train += stats['train']
            total_val += stats['val']
            total_test += stats['test']
            total_all += stats['total']
        
        print("-" * 55)
        print(f"{'TOTAL':<20} {total_train:<8} {total_val:<8} {total_test:<8} {total_all:<8}")
        
        print(f"\n📋 Key Points:")
        print(f"  • Original source data is completely untouched")
        print(f"  • New dataset has consecutive class indices (0, 1, 2, ...)")
        print(f"  • YOLO training will work without class index errors")
        print(f"  • Remapping only applied to copied annotation files")
    
    def _create_training_yaml(self, dataset_dir, class_stats):
        """Create YAML configuration file with consecutive class indices"""
        # Sort by new class ID to ensure proper order (0, 1, 2, ...)
        sorted_class_stats = dict(sorted(class_stats.items()))
        
        yaml_config = {
            'path': str(dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images', 
            'test': 'test/images',
            'nc': len(sorted_class_stats),
            'names': {class_id: stats['name'] for class_id, stats in sorted_class_stats.items()}
        }
        
        yaml_path = dataset_dir / "dataset.yaml"
        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
            
            # Update project state with YAML path
            self.project_state['training_params']['data_yaml'] = str(yaml_path)
            self.save_project()
            
            print(f"\n✅ Training configuration saved: {self.format_path(str(yaml_path))}")
            print(f"📋 YAML contains {len(sorted_class_stats)} classes with indices 0-{len(sorted_class_stats)-1}")
            
            # Show class mapping for verification
            print(f"\n📋 Final Class Mapping:")
            for class_id, stats in sorted_class_stats.items():
                original_id = stats.get('original_id', class_id)
                print(f"  {class_id}: {stats['name']} (was {original_id})")
            
            return str(yaml_path)
        except Exception as e:
            print(f"⚠️ Error saving YAML config: {e}")
            return None
    
    def _validate_dataset_consistency(self, dataset_path, expected_num_classes):
        """Validate that dataset annotations match YAML configuration"""
        print(f"\n🔍 Validating dataset consistency...")
        
        # Check annotation files in train/val/test splits
        issues_found = []
        all_class_ids = set()
        
        for split in ['train', 'val', 'test']:
            labels_dir = dataset_path / split / "labels"
            if not labels_dir.exists():
                continue
                
            for ann_file in labels_dir.glob("*.txt"):
                try:
                    with open(ann_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) >= 5:
                                    try:
                                        class_id = int(parts[0])
                                        all_class_ids.add(class_id)
                                        
                                        # Check if class ID is within expected range
                                        if class_id < 0 or class_id >= expected_num_classes:
                                            issues_found.append(
                                                f"Invalid class ID {class_id} in {ann_file.name}:{line_num} "
                                                f"(expected 0-{expected_num_classes-1})"
                                            )
                                    except ValueError:
                                        issues_found.append(f"Invalid class ID format in {ann_file.name}:{line_num}")
                except Exception as e:
                    issues_found.append(f"Error reading {ann_file.name}: {e}")
        
        # Report validation results
        if issues_found:
            print(f"❌ Found {len(issues_found)} validation issues:")
            for issue in issues_found[:5]:  # Show first 5 issues
                print(f"  • {issue}")
            if len(issues_found) > 5:
                print(f"  • ... and {len(issues_found) - 5} more issues")
            return False
        else:
            max_class_id = max(all_class_ids) if all_class_ids else -1
            min_class_id = min(all_class_ids) if all_class_ids else 0
            
            print(f"✅ Dataset validation passed!")
            print(f"  • Found class IDs: {min_class_id}-{max_class_id}")
            print(f"  • Expected range: 0-{expected_num_classes-1}")
            print(f"  • Total unique classes: {len(all_class_ids)}")
            
            # Check for missing class IDs
            expected_ids = set(range(expected_num_classes))
            missing_ids = expected_ids - all_class_ids
            if missing_ids:
                print(f"  ⚠️ Warning: Missing class IDs in annotations: {sorted(missing_ids)}")
            
            return True
    
    def show_training_guide(self):
        """Show comprehensive training setup guide"""
        dataset_dir = self.project_state.get('dataset_output_dir')
        if not dataset_dir:
            print("❌ No dataset created yet. Create dataset first.")
            return
        
        dataset_yaml = Path(dataset_dir) / "dataset.yaml"
        if not dataset_yaml.exists():
            print("❌ Dataset YAML not found. Recreate dataset.")
            return
        
        print("\n" + "="*80)
        print("🚀 YOLO TRAINING SETUP GUIDE")
        print("="*80)
        
        self._show_prerequisites()
        self._show_training_commands(dataset_yaml)
        self._show_hardware_recommendations()
        self._show_training_tips()
        self._show_post_training_steps(dataset_yaml)
    
    def _show_prerequisites(self):
        """Show training prerequisites"""
        print(f"\n📋 PREREQUISITES:")
        print("-" * 40)
        print("✓ Python 3.8+")
        print("✓ PyTorch with CUDA:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("✓ Ultralytics YOLO:")
        print("  pip install ultralytics")
        print("✓ Additional packages:")
        print("  pip install opencv-python pillow matplotlib tqdm")
        
        print(f"\n🔍 GPU Check:")
        print('  python -c "import torch; print(f\'CUDA: {torch.cuda.is_available()}\')"')
    
    def _show_training_commands(self, dataset_yaml):
        """Show training command options"""
        params = self.project_state['training_params']
        
        print(f"\n🎯 TRAINING COMMANDS:")
        print("-" * 40)
        
        # Quick start command
        quick_cmd = f"yolo train data=\"{dataset_yaml}\" model={params['model_size']} epochs={params['epochs']}"
        print(f"Quick Start:")
        print(f"  {quick_cmd}")
        
        # Recommended command
        rec_cmd = (f"yolo train data=\"{dataset_yaml}\" model={params['model_size']} "
                  f"epochs={params['epochs']} batch={params['batch_size']} "
                  f"imgsz={params['image_size']} lr0={params['learning_rate']} "
                  f"patience={params['patience']} project={params['project_name']}")
        print(f"\nRecommended:")
        print(f"  {rec_cmd}")
        
        # Full command with all parameters
        full_cmd = (f"yolo train data=\"{dataset_yaml}\" model={params['model_size']} "
                   f"epochs={params['epochs']} batch={params['batch_size']} "
                   f"imgsz={params['image_size']} lr0={params['learning_rate']} "
                   f"patience={params['patience']} save_period={params['save_period']} "
                   f"workers={params['workers']} device={params['device']} "
                   f"project={params['project_name']}")
        print(f"\nFull Options:")
        print(f"  {full_cmd}")
    
    def _show_hardware_recommendations(self):
        """Show hardware recommendations"""
        print(f"\n💻 HARDWARE RECOMMENDATIONS:")
        print("-" * 40)
        print("Minimum:")
        print("  • 8GB RAM, 4GB GPU VRAM")
        print("  • GTX 1660 or better")
        print("Recommended:")
        print("  • 16GB+ RAM, 8GB+ GPU VRAM")
        print("  • RTX 3070 or better")
        print("  • SSD storage")
    
    def _show_training_tips(self):
        """Show training tips and monitoring"""
        print(f"\n💡 TRAINING TIPS:")
        print("-" * 40)
        print("• Monitor with TensorBoard: tensorboard --logdir runs/train")
        print("• Use GPU monitoring: nvidia-smi")
        print("• Reduce batch size if CUDA out of memory")
        print("• Training time: 1-12+ hours depending on dataset size")
        print("• Best model saved as 'best.pt' in results folder")
    
    def _show_post_training_steps(self, dataset_yaml):
        """Show post-training validation and deployment steps"""
        print(f"\n✅ AFTER TRAINING:")
        print("-" * 40)
        print("1. Validate model:")
        print(f'   yolo val model=runs/train/exp/weights/best.pt data="{dataset_yaml}"')
        print("2. Test predictions:")
        print("   yolo predict model=runs/train/exp/weights/best.pt source=test_images/")
        print("3. Export for deployment:")
        print("   yolo export model=runs/train/exp/weights/best.pt format=onnx")
        
        print(f"\n🔧 TROUBLESHOOTING:")
        print("-" * 40)
        print("• 'CUDA out of memory' → Reduce batch size (try 8 or 4)")
        print("• 'No module named ultralytics' → pip install ultralytics")
        print("• 'Dataset class indices error' → ✅ AUTOMATICALLY FIXED!")
        print("  (This tool automatically remaps class indices to 0, 1, 2, ...)")
        print("• Training too slow → Check GPU usage with nvidia-smi")
        print("• Poor accuracy → More data, better annotations, longer training")
        print("• 'RuntimeError: No CUDA' → Install PyTorch with CUDA support")
        
        print(f"\n📁 DATA SAFETY GUARANTEE:")
        print("-" * 40)
        print("• Your original image and annotation directories are NEVER modified")
        print("• Tool only copies files TO the new dataset directory")
        print("• Class remapping only happens on the copied annotation files")
        print("• Original source data remains exactly as it was")
    
    def configure_training_parameters(self):
        """Configure training parameters with smart defaults"""
        print("\n" + "="*80)
        print("⚙️ TRAINING PARAMETER CONFIGURATION")
        print("="*80)
        
        params = self.project_state['training_params']
        recommendations = self._get_training_recommendations()
        
        print("Current parameters:")
        for param, value in params.items():
            if param != 'data_yaml':  # Skip internal parameter
                print(f"  {param}: {value}")
        
        print(f"\n🔧 Configure parameters (Enter to keep current):")
        
        # Configure each parameter with recommendations
        self._configure_epochs(params, recommendations)
        self._configure_batch_size(params, recommendations)
        self._configure_image_size(params, recommendations)
        self._configure_model_size(params, recommendations)
        self._configure_learning_rate(params, recommendations)
        self._configure_advanced_params(params)
        
        self.save_project()  # Auto-save
        print(f"\n✅ Training parameters updated!")
    
    def _get_training_recommendations(self):
        """Get intelligent training recommendations based on dataset"""
        recommendations = {}
        
        stats = self.project_state['dataset_stats']
        selected_classes = self.project_state['selected_classes']
        
        total_images = stats.get('valid_files', 0) if stats else 0
        num_classes = len([c for c in selected_classes.values() if c['included']]) if selected_classes else 0
        
        # Epochs
        if total_images < 500:
            recommendations['epochs'] = {'value': 150, 'reason': 'Small dataset needs more epochs'}
        elif total_images < 2000:
            recommendations['epochs'] = {'value': 100, 'reason': 'Medium dataset standard'}
        else:
            recommendations['epochs'] = {'value': 80, 'reason': 'Large dataset fewer epochs'}
        
        # Model size
        if total_images < 1000:
            recommendations['model_size'] = {'value': 'yolov8n.pt', 'reason': 'Nano prevents overfitting'}
        elif total_images < 5000:
            recommendations['model_size'] = {'value': 'yolov8s.pt', 'reason': 'Small good for medium data'}
        else:
            recommendations['model_size'] = {'value': 'yolov8m.pt', 'reason': 'Medium for large data'}
        
        # Learning rate
        recommendations['learning_rate'] = {
            'value': 0.01 if num_classes <= 3 else 0.005,
            'reason': f'Optimized for {num_classes} classes'
        }
        
        return recommendations
    
    def _configure_epochs(self, params, recommendations):
        """Configure training epochs"""
        rec = recommendations.get('epochs', {})
        rec_text = f" (Rec: {rec['value']} - {rec['reason']})" if rec else ""
        
        while True:
            try:
                epochs_input = input(f"Epochs (current: {params['epochs']}){rec_text}: ").strip()
                if epochs_input:
                    epochs = int(epochs_input)
                    if epochs <= 0:
                        print("Epochs must be positive")
                        continue
                    params['epochs'] = epochs
                break
            except ValueError:
                print("Please enter a valid number")
    
    def _configure_batch_size(self, params, recommendations):
        """Configure batch size with GPU memory guidance"""
        print(f"\nBatch size guidelines:")
        print("  4-8: Low GPU memory (4GB)")
        print("  16: Standard GPU memory (8GB)")
        print("  32+: High GPU memory (12GB+)")
        
        while True:
            try:
                batch_input = input(f"Batch size (current: {params['batch_size']}): ").strip()
                if batch_input:
                    batch_size = int(batch_input)
                    if batch_size <= 0:
                        print("Batch size must be positive")
                        continue
                    params['batch_size'] = batch_size
                break
            except ValueError:
                print("Please enter a valid number")
    
    def _configure_image_size(self, params, recommendations):
        """Configure image size"""
        print(f"\nImage size options:")
        print("  416: Faster training, lower accuracy")
        print("  640: Standard (recommended)")
        print("  832: Slower training, higher accuracy")
        
        while True:
            try:
                size_input = input(f"Image size (current: {params['image_size']}): ").strip()
                if size_input:
                    image_size = int(size_input)
                    if image_size < 320 or image_size > 1280:
                        print("Image size should be 320-1280")
                        continue
                    params['image_size'] = image_size
                break
            except ValueError:
                print("Please enter a valid number")
    
    def _configure_model_size(self, params, recommendations):
        """Configure model size"""
        rec = recommendations.get('model_size', {})
        rec_text = f" (Rec: {rec['value']} - {rec['reason']})" if rec else ""
        
        print(f"\nModel options:")
        print("  yolov8n.pt: Nano (fastest)")
        print("  yolov8s.pt: Small (balanced)")
        print("  yolov8m.pt: Medium (accurate)")
        print("  yolov8l.pt: Large (slow but accurate)")
        
        model_input = input(f"Model (current: {params['model_size']}){rec_text}: ").strip()
        if model_input:
            valid_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
            if model_input in valid_models:
                params['model_size'] = model_input
            else:
                print(f"Invalid model. Keeping: {params['model_size']}")
    
    def _configure_learning_rate(self, params, recommendations):
        """Configure learning rate"""
        rec = recommendations.get('learning_rate', {})
        rec_text = f" (Rec: {rec['value']} - {rec['reason']})" if rec else ""
        
        while True:
            try:
                lr_input = input(f"Learning rate (current: {params['learning_rate']}){rec_text}: ").strip()
                if lr_input:
                    learning_rate = float(lr_input)
                    if learning_rate <= 0 or learning_rate > 1:
                        print("Learning rate should be 0-1")
                        continue
                    params['learning_rate'] = learning_rate
                break
            except ValueError:
                print("Please enter a valid number")
    
    def _configure_advanced_params(self, params):
        """Configure advanced parameters"""
        print(f"\n🔧 Advanced options (Enter to skip):")
        
        # Patience
        patience_input = input(f"Early stopping patience (current: {params['patience']}): ").strip()
        if patience_input:
            try:
                params['patience'] = int(patience_input)
            except ValueError:
                pass
        
        # Device
        device_input = input(f"Device [auto/cpu/0/1] (current: {params['device']}): ").strip()
        if device_input:
            params['device'] = device_input
        
        # Project name
        project_input = input(f"Output project name (current: {params['project_name']}): ").strip()
        if project_input:
            params['project_name'] = project_input
    
    def start_training_wizard(self):
        """Comprehensive training wizard"""
        if not self.project_state.get('dataset_output_dir'):
            print("❌ No dataset created. Create dataset first.")
            return
        
        dataset_yaml = Path(self.project_state['dataset_output_dir']) / "dataset.yaml"
        if not dataset_yaml.exists():
            print("❌ Dataset YAML missing. Recreate dataset.")
            return
        
        print("\n" + "="*80)
        print("🚀 TRAINING WIZARD")
        print("="*80)
        
        # Generate training command
        params = self.project_state['training_params']
        training_cmd = self._generate_training_command(dataset_yaml)
        
        print(f"Dataset: {self.format_path(str(dataset_yaml))}")
        print(f"Command: {training_cmd}")
        
        print(f"\nOptions:")
        print("1: 📋 Copy command to clipboard")
        print("2: 🚀 Start training now")
        print("3: 💾 Save training script")
        print("4: 📖 Show full training guide")
        print("5: ⚙️ Modify parameters")
        print("6: 🔙 Back to menu")
        
        while True:
            choice = input("Choose option (1-6): ").strip()
            
            if choice == '1':
                self._copy_to_clipboard(training_cmd)
                break
            elif choice == '2':
                self._start_training_now(training_cmd, dataset_yaml.parent)
                break
            elif choice == '3':
                self._save_training_script(training_cmd, dataset_yaml.parent)
                break
            elif choice == '4':
                self.show_training_guide()
                break
            elif choice == '5':
                self.configure_training_parameters()
                # Regenerate command with new parameters
                training_cmd = self._generate_training_command(dataset_yaml)
                print(f"Updated command: {training_cmd}")
            elif choice == '6':
                break
            else:
                print("Please enter 1-6")
    
    def _generate_training_command(self, dataset_yaml):
        """Generate YOLO training command"""
        params = self.project_state['training_params']
        
        cmd_parts = [
            "yolo", "train",
            f'data="{dataset_yaml}"',
            f"model={params['model_size']}",
            f"epochs={params['epochs']}",
            f"batch={params['batch_size']}",
            f"imgsz={params['image_size']}",
            f"lr0={params['learning_rate']}",
            f"patience={params['patience']}",
            f"device={params['device']}",
            f"project={params['project_name']}"
        ]
        
        return " ".join(cmd_parts)
    
    def _copy_to_clipboard(self, command):
        """Copy command to clipboard"""
        try:
            import pyperclip
            pyperclip.copy(command)
            print("✅ Command copied to clipboard!")
        except ImportError:
            print("⚠️ Install pyperclip for clipboard: pip install pyperclip")
            print(f"\nCopy this command manually:")
            print(command)
    
    def _start_training_now(self, command, dataset_dir):
        """Start training immediately"""
        print(f"\n🚀 Starting YOLO training...")
        print("This may take several hours...")
        
        confirm = input("Start training? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("Training cancelled")
            return
        
        try:
            import subprocess
            original_dir = os.getcwd()
            os.chdir(dataset_dir)
            
            print("Training started...")
            result = subprocess.run(command.split(), capture_output=False, text=True)
            
            os.chdir(original_dir)
            
            if result.returncode == 0:
                print("✅ Training completed successfully!")
            else:
                print("❌ Training had errors. Check output above.")
        except Exception as e:
            print(f"❌ Error starting training: {e}")
            print("Try running the command manually")
    
    def _save_training_script(self, command, dataset_dir):
        """Save training command as script"""
        script_path = dataset_dir / "start_training.bat"
        try:
            with open(script_path, 'w') as f:
                f.write("@echo off\n")
                f.write("echo Starting YOLO Training...\n")
                f.write(f"cd /d \"{dataset_dir}\"\n")
                f.write(f"{command}\n")
                f.write("echo Training complete!\n")
                f.write("pause\n")
            
            print(f"✅ Training script saved: {self.format_path(str(script_path))}")
            print("Double-click the .bat file to start training")
        except Exception as e:
            print(f"❌ Error saving script: {e}")
    
    def show_current_status(self):
        """Show comprehensive project status"""
        print("\n" + "="*80)
        print(f"📊 PROJECT STATUS: {self.project_state['project_name']}")
        print("="*80)
        
        # Project info
        print(f"Created: {self.project_state.get('created_date', 'Unknown')}")
        print(f"Modified: {self.project_state.get('last_modified', 'Unknown')}")
        
        # Paths
        print(f"\n📁 Paths:")
        paths = self.project_state['paths']
        for name, path in paths.items():
            status = "✅" if (Path(path).exists() if path else False) else "❌"
            print(f"  {name}: {status} {self.format_path(path)}")
        
        # Dataset info
        stats = self.project_state['dataset_stats']
        if stats:
            print(f"\n📊 Dataset Analysis:")
            print(f"  Valid files: {stats['valid_files']}")
            print(f"  Total annotations: {stats['total_annotations']}")
            print(f"  Classes found: {len(stats['class_counts'])}")
        
        # Selected classes
        selected = self.project_state['selected_classes']
        if selected:
            included = {k: v for k, v in selected.items() if v['included']}
            print(f"\n🎯 Selected Classes ({len(included)}):")
            for class_id, info in included.items():
                target = self.project_state['target_counts'].get(class_id, 'Not set')
                print(f"  {info['name']}: {info['count']} available, target: {target}")
        
        # Dataset output
        output_dir = self.project_state.get('dataset_output_dir')
        if output_dir:
            status = "✅" if Path(output_dir).exists() else "❌"
            yaml_status = "✅" if (Path(output_dir) / "dataset.yaml").exists() else "❌"
            print(f"\n📦 Dataset Output:")
            print(f"  Directory: {status} {self.format_path(output_dir)}")
            print(f"  YAML config: {yaml_status}")
            
            # Show class remapping if available
            class_mapping = self.project_state.get('class_mapping')
            if class_mapping:
                print(f"  🔄 Class remapping applied:")
                for old_id, new_id in sorted(class_mapping.items(), key=lambda x: x[1]):
                    class_name = self.project_state['selected_classes'][old_id]['name']
                    print(f"    {class_name}: {old_id} → {new_id}")
        
        # Training parameters
        print(f"\n⚙️ Training Configuration:")
        params = self.project_state['training_params']
        key_params = ['model_size', 'epochs', 'batch_size', 'image_size', 'learning_rate']
        for param in key_params:
            print(f"  {param}: {params[param]}")
        
        # Show data YAML if available
        if 'data_yaml' in params and params['data_yaml']:
            print(f"  data_yaml: {self.format_path(params['data_yaml'])}")
    
    def paths_configuration_menu(self):
        """Enhanced path configuration menu"""
        print("\n" + "="*80)
        print("📁 PATH CONFIGURATION")
        print("="*80)
        
        paths = self.project_state['paths']
        
        while True:
            print(f"\nCurrent paths:")
            for i, (name, path) in enumerate(paths.items(), 1):
                status = "✅" if (Path(path).exists() if path else False) else "❌"
                print(f"  {i}: {name}: {status} {self.format_path(path)}")
            
            print(f"\nOptions:")
            print(f"1-3: Update path")
            print(f"4: Reset to defaults")
            print(f"5: Auto-detect paths")
            print(f"6: Validate all paths")
            print(f"7: Back to menu")
            
            choice = input("Choose option (1-7): ").strip()
            
            if choice in ['1', '2', '3']:
                path_names = list(paths.keys())
                path_name = path_names[int(choice) - 1]
                path_type = "file" if "yaml" in path_name else "directory"
                
                new_path = self.get_user_path(
                    f"Enter {path_name.replace('_', ' ')}:",
                    paths[path_name],
                    path_type
                )
                paths[path_name] = new_path
                
                # Auto-reload YAML if changed
                if path_name == 'yaml_path' and new_path != paths[path_name]:
                    self.load_yaml_config()
                
                self.save_project()
                
            elif choice == '4':
                paths.update(self.default_paths)
                self.save_project()
                print("✅ Paths reset to defaults")
                
            elif choice == '5':
                self._auto_detect_paths()
                
            elif choice == '6':
                self._validate_all_paths()
                
            elif choice == '7':
                break
                
            else:
                print("Please enter 1-7")
    
    def _auto_detect_paths(self):
        """Auto-detect common path structures"""
        print("🔍 Auto-detecting paths...")
        
        # Look for common directory structures
        common_patterns = [
            "images", "imgs", "data/images", "dataset/images",
            "labels", "annotations", "data/labels", "dataset/labels"
        ]
        
        detected = {}
        current_dir = Path.cwd()
        
        for pattern in common_patterns:
            path = current_dir / pattern
            if path.exists():
                if "image" in pattern:
                    detected['images_dir'] = str(path)
                elif "label" in pattern or "annotation" in pattern:
                    detected['labels_dir'] = str(path)
        
        # Look for YAML files
        for yaml_file in current_dir.rglob("*.yaml"):
            if any(keyword in yaml_file.name.lower() for keyword in ['data', 'config', 'dataset']):
                detected['yaml_path'] = str(yaml_file)
                break
        
        if detected:
            print("Found potential paths:")
            for name, path in detected.items():
                print(f"  {name}: {path}")
            
            if input("Use detected paths? (y/n): ").strip().lower() in ['y', 'yes']:
                self.project_state['paths'].update(detected)
                self.save_project()
                print("✅ Paths updated")
        else:
            print("No common path structures detected")
    
    def _validate_all_paths(self):
        """Validate all configured paths"""
        print("🔍 Validating paths...")
        
        paths = self.project_state['paths']
        all_valid = True
        
        for name, path in paths.items():
            if not path:
                print(f"❌ {name}: Not set")
                all_valid = False
                continue
            
            path_obj = Path(path)
            if name == 'yaml_path':
                if path_obj.is_file():
                    print(f"✅ {name}: Valid file")
                else:
                    print(f"❌ {name}: File not found")
                    all_valid = False
            else:
                if path_obj.is_dir():
                    # Count files for directories
                    if name == 'images_dir':
                        count = len(list(path_obj.glob("*.jpg")) + list(path_obj.glob("*.png")))
                        print(f"✅ {name}: Valid directory ({count} images)")
                    elif name == 'labels_dir':
                        count = len(list(path_obj.glob("*.txt")))
                        print(f"✅ {name}: Valid directory ({count} text files)")
                else:
                    print(f"❌ {name}: Directory not found")
                    all_valid = False
        
        if all_valid:
            print("✅ All paths are valid!")
        else:
            print("❌ Some paths need attention")
    
    def show_class_remapping(self):
        """Show detailed class remapping information"""
        class_mapping = self.project_state.get('class_mapping')
        if not class_mapping:
            print("❌ No class remapping available. Create a balanced dataset first.")
            return
        
        print("\n" + "="*80)
        print("🔄 CLASS INDEX REMAPPING DETAILS")
        print("="*80)
        
        print("This tool automatically remaps class indices to consecutive numbers (0, 1, 2, ...)")
        print("to ensure YOLO compatibility and prevent training errors.\n")
        
        print(f"{'Original ID':<12} {'New ID':<8} {'Class Name':<25} {'Samples':<10}")
        print("-" * 65)
        
        selected_classes = self.project_state['selected_classes']
        target_counts = self.project_state.get('target_counts', {})
        
        # Sort by new ID for display
        sorted_mapping = sorted(class_mapping.items(), key=lambda x: x[1])
        
        for old_id, new_id in sorted_mapping:
            if old_id in selected_classes:
                class_info = selected_classes[old_id]
                class_name = class_info['name']
                target_count = target_counts.get(old_id, 'N/A')
                print(f"{old_id:<12} {new_id:<8} {class_name:<25} {target_count:<10}")
        
        print(f"\n📋 Summary:")
        print(f"  • Original class range: {min(class_mapping.keys())}-{max(class_mapping.keys())}")
        print(f"  • New class range: 0-{len(class_mapping)-1}")
        print(f"  • Total classes in dataset: {len(class_mapping)}")
        
        # Show YAML info if available
        dataset_dir = self.project_state.get('dataset_output_dir')
        if dataset_dir:
            yaml_path = Path(dataset_dir) / "dataset.yaml"
            if yaml_path.exists():
                print(f"  • YAML file: {self.format_path(str(yaml_path))}")
                print(f"  • Annotation files: Automatically updated with new indices")
        
        print(f"\n💡 Why remapping is important:")
        print("  • YOLO requires consecutive class indices starting from 0")
        print("  • Prevents 'invalid class indices' training errors")
        print("  • Ensures compatibility with YOLO's internal processing")
        print("  • Original class names are preserved in the YAML file")
        print("  • Original source data remains completely unchanged")
    
    def main_menu(self):
        """Enhanced main menu with better organization"""
        # Load YAML on startup if available
        if self.project_state['paths']['yaml_path']:
            self.load_yaml_config()
        
        while True:
            print("\n" + "="*80)
            print(f"🏠 MAIN MENU - {self.project_state['project_name']}")
            print("="*80)
            
            # Show quick status
            stats = self.project_state['dataset_stats']
            selected = self.project_state['selected_classes']
            dataset_dir = self.project_state.get('dataset_output_dir')
            
            status_items = []
            if stats:
                status_items.append(f"📊 {stats['valid_files']} images analyzed")
            if selected:
                included_count = sum(1 for v in selected.values() if v['included'])
                status_items.append(f"🎯 {included_count} classes selected")
            if dataset_dir and Path(dataset_dir).exists():
                status_items.append("📦 Dataset created")
            
            if status_items:
                print("Status: " + " | ".join(status_items))
            
            print(f"\n📋 Project Management:")
            print("1: 📁 Configure paths")
            print("2: 📊 Show project status")
            print("3: 💾 Switch/create project")
            
            print(f"\n🔍 Dataset Preparation:")
            print("4: 📊 Analyze dataset")
            print("5: 🎯 Select training classes") 
            print("6: ⚖️ Set target counts")
            print("7: 📦 Create balanced dataset")
            print("8: 🔄 Show class remapping")
            
            print(f"\n🚀 Training:")
            print("9: ⚙️ Configure training parameters")
            print("10: 🚀 Training wizard")
            print("11: 📖 Show training guide")
            
            print(f"\n🐛 Debug:")
            print("13: 🔍 Debug class IDs")
            
            print(f"\n❌ Exit:")
            print("12: Exit")
            
            choice = input("\nChoose option (1-13): ").strip()
            
            if choice == '1':
                self.paths_configuration_menu()
                
            elif choice == '2':
                self.show_current_status()
                
            elif choice == '3':
                self.load_or_create_project()
                
            elif choice == '4':
                if self.analyze_dataset():
                    self.display_dataset_analysis()
                else:
                    print("❌ Dataset analysis failed. Check paths.")
                    
            elif choice == '5':
                if not self.project_state['dataset_stats']:
                    print("❌ Run dataset analysis first (option 4)")
                else:
                    self.select_classes_for_training()
                    
            elif choice == '6':
                if not any(v['included'] for v in self.project_state['selected_classes'].values()):
                    print("❌ Select classes first (option 5)")
                else:
                    self.set_balanced_dataset_targets()
                    
            elif choice == '7':
                if not self.project_state['target_counts']:
                    print("❌ Set target counts first (option 6)")
                else:
                    if self.create_balanced_dataset():
                        print("✅ Dataset created successfully!")
                        
            elif choice == '8':
                self.show_class_remapping()
                        
            elif choice == '9':
                self.configure_training_parameters()
                
            elif choice == '10':
                self.start_training_wizard()
                
            elif choice == '11':
                self.show_training_guide()
                
            elif choice == '12':
                print("👋 Goodbye!")
                break
                
            elif choice == '13':
                self.debug_class_ids()
                
            else:
                print("Please enter 1-13")

def main():
    """Main function with improved error handling"""
    try:
        # Check for required packages
        required_packages = ['yaml', 'tqdm']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing required packages: {', '.join(missing_packages)}")
            print(f"Install with: pip install {' '.join(missing_packages)}")
            return
        
        print("🔥 YOLO Training Tool v2.0")
        print("Enhanced with project management and robust dataset creation")
        
        tool = YOLOTrainingTool()
        tool.main_menu()
        
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your configuration and try again")

if __name__ == "__main__":
    main()