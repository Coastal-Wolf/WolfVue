# WolfVue: Wildlife Video Classifier

A tool for automatically classifying trail camera videos using YOLO object detection, originally developed for The Gray Wolf Research Project.

## Quick Start

### Prerequisites

Python 3.8 or higher installed on your system.

### Installation

Install the required packages:
(you can do this in CMD)
```bash
pip install ultralytics opencv-python pyyaml tqdm colorama
```

### Basic Usage


1. Place your videos in the `input_videos/` folder
2. Run the script (using CMD):

```bash
python wolfvue.py
```
The script will give you prompts to change the filepath of any core components (if you wanted to test a new model fine-tune for example)
If you dont have anything youd like to edit, press enter after each prompt. it should already be pre-configured for windows systems.

The script will process all videos and sort them into the `output_videos/` folder based on detected species.

youll see lots of lines of frame detection zooming down your screen, that means its running. Depending on your system and your videos, this could take time. 

When the videos are finished processing, you should see text giving you a brief summary of identifications, and a file in your output folder with a more detailed classification report.

## How It Works

WolfVue processes trail camera footage frame by frame, detecting animals using a trained YOLO model. It then analyzes the temporal patterns of detections to classify each video into one of three categories:

- **Species folders**: Videos with a clear dominant species (>70% of detections)
- **Unsorted**: Videos with multiple species, predator-prey conflicts, or unclear patterns to be manually sorted
- **No_Animal**: Videos with zero animal detections

## Configuration

The main parameters you can adjust in the script:

```python
CONFIDENCE_THRESHOLD = 0.25          # Minimum YOLO confidence score
DOMINANT_SPECIES_THRESHOLD = 0.7     # Required percentage for dominant species
MAX_SPECIES_TRANSITIONS = 5          # Maximum allowed species changes
CONSECUTIVE_EMPTY_FRAMES = 30        # Empty frames to break a detection sequence
```
NOTE: this only adjusts the sorting algoritm based on frame detection BY the yolo model, but does not effect the YOLO model itself.
## Advanced Usage

### Understanding the Classification Algorithm

The classifier uses a multi-factor approach to determine video categories:

1. **Detection Aggregation**: All YOLO detections above the confidence threshold are collected frame by frame
2. **Temporal Clustering**: Consecutive frames with the same species are grouped into clusters
3. **Transition Analysis**: The algorithm counts how often the detected species changes throughout the video
4. **Dominance Calculation**: Both total detection count and frame coverage are considered

### Classification Rules

A video is classified as a specific species only if:
- That species represents >70% of all detections
- Species transitions are below the threshold
- No predator-prey conflicts exist (e.g., wolf and deer in same video)

This conservative approach ensures high confidence in single-species classifications.

### Model Requirements

The YOLO model (best.pt) must be trained to detect the species defined in WlfCamData.yaml. The yaml file maps class IDs to species names:

```yaml
names:
  0: "WhiteTail"
  1: "MuleDeer"
  2: "Elk"
  3: "Moose"
  4: "Cougar"
  5: "Lynx"
  6: "Wolf"
  7: "Coyote"
  8: "Fox"
  9: "Bear"
```
Ideally, this list will expand in the future.

### About YOLO Model WolfVue_Beta1

this model is a very rough model that was scraped together with as much data as I could find;

Coyote: 65 instances
Elk: 236 instances
Moose: 40 instances
MuleDeer: 167 instances
WhiteTail: 60 instances
Wolf: 27 instances

this means its only actually able to identify 6 different species, is unbalanced, and highly skewed towards Elk because they make up so much of the dataset.

This is NOT a good model, but its a start. 

I cannot share the data, as its under NDA by the Gray Wolf Research Project, as some of it is on private property, so open weight is the best I can do. 

The goal of open sourcing this is to hopefully get some more trail cam videos that can be fine tuned for more species, more accurately, and maybe more efficiently.
If im being completely honest I hardly know what im doing, so someone who does know what theyre doing might be able to take this to the next level, and make a good model
that researchers and hobbiests may be able to utilize in the future.

you can find more details as to the results of this model training in its respective folder.

### Performance Considerations

Processing speed depends on:
- Video resolution (higher resolution = slower)
- Model complexity (YOLOv8n is fastest, YOLOv8x is most accurate)
- Hardware (GPU acceleration dramatically improves speed)

For GPU acceleration, ensure you have CUDA-enabled PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Batch Processing

The script automatically processes all videos in the input folder. For large datasets:
- The pre-scan estimates total processing time
- Progress bars show both per-video and overall progress
- A processing report is generated with detailed results

### Output Structure

Videos are organized into a taxonomy-based folder structure:

```
output_videos/
├── Sorted/
│   ├── Ungulates/
│   │   ├── WhiteTail/
│   │   ├── MuleDeer/
│   │   ├── Elk/
│   │   └── Moose/
│   └── Predators/
│       ├── Cougar/
│       ├── Lynx/
│       ├── Wolf/
│       ├── Coyote/
│       ├── Fox/
│       └── Bear/
├── Unsorted/
└── No_Animal/
```

### Customizing the Taxonomy

To modify the folder structure, edit the TAXONOMY dictionary in the script:

```python
TAXONOMY = {
    "Category": {
        "Species": ["Species"],
        # Add more as needed
    }
}
```

### Processing Report

After processing, check `processing_report.txt` for:
- Classification summary statistics
- Per-video classification details
- Detection rates and species percentages
- Reasoning for each classification decision

## Troubleshooting

**No videos found**: Ensure videos are in common formats (.mp4, .avi, .mov, .mkv) and check both uppercase and lowercase extensions.

**Path errors**: The script uses relative paths from its location. Keep all folders (input_videos, output_videos, weights) in the same directory as the script.

**Memory issues**: For very long videos, consider splitting them or reducing resolution before processing.

**Slow processing**: Without GPU acceleration, expect approximately 10 frames per second on modern CPUs.

## Technical Details

### Frame-by-Frame Analysis

Each frame undergoes:
1. YOLO inference to detect bounding boxes
2. Confidence filtering
3. Species identification mapping
4. Temporal context integration

### Temporal Consistency

The algorithm maintains temporal consistency by:
- Tracking species across consecutive frames
- Identifying detection clusters
- Penalizing frequent species transitions
- Handling gaps in detections (animal temporarily out of frame)

### Edge Cases

The classifier handles several edge cases:
- Brief appearances by secondary species are ignored if under threshold
- Predator-prey scenarios always result in "Unsorted" classification
- Videos with sparse, intermittent detections are evaluated based on total pattern

### final remarks

I want to preface this by saying most of this was created with AI, the scripting, learning how to train models, etc. While I have a basic understanding of python, I would not have been
Able to achieve this without Claude, that said, coders who do know what theyre doing will probably encounter some odd quirks in the code because of this, and I apoligize in advance.

At first I was conflicted about using AI to code, but ultimately, the means that this is done does not matter so long as the end project benefits scientists. researchers, and hobbiests free of charge as intended.

Also note that I would like this project to be specifically for Trail cameras, so please make sure any data that is fine tuned is done with data FROM trail cameras. 

Thank you for reading, and possibly using this. I think this could make for a great open source project!

## Contributing

When contributing, please maintain the existing code structure and add appropriate error handling for new features. The codebase prioritizes readability and maintainability over premature optimization.

## Credits

Created by Nathan Bluto  
initial data from The Gray Wolf Research Project  
Facilitated by Dr. Ausband

