# ISL to English Translator

Converts Indian Sign Language (ISL) gestures to English text using deep learning with **video-first temporal learning**.

## Video-First Architecture with Sliding Windows

Instead of processing entire videos as single gestures, we extract **multiple overlapping temporal windows** from each video. This captures different parts of a gesture across frames.

```
One Gesture Video
  ↓
Extract all frames (skip frames with no hands)
  ↓
Create sliding windows (default: 10 frames per window, 50% overlap)
  ↓
Multiple gesture samples (one per window)
```

**Why sliding windows?**
- A gesture is typically 10-30 frames of motion
- 10 frames ≈ 0.33 seconds at 30fps
- 50% overlap (stride=5) generates more training data from one video
- Each window = independent training sample for the same gesture class

**Example:**
```
Video: "hello.mp4" (150 frames)
  Window 1: Frames 0-9
  Window 2: Frames 5-14
  Window 3: Frames 10-19
  ... (16 windows total with 50% overlap)
```

All windows saved to `data/dataset/hello/` as separate sequences.

## Quick Start: Complete Workflow

### 1. Organize Videos (Multiple Versions Consolidate Automatically)

Place videos (any format: `.mp4`, `.avi`, `.mov`, `.mkv`) in flat or nested directories:

```
data/handsigns/
  ├── hello(1).mp4            ← All three consolidated → data/dataset/hello/
  ├── hello(2).mp4            ← (gesture name extraction removes version)
  ├── hello_v3.mp4            ← (patterns: (1), _1, _v1 all map to base name)
  ├── goodbye.mp4             ← Class: "goodbye"
  ├── thanks_1.mp4            ← Consolidates with thanks_2, thanks_3...
  ├── nested/
  │   └── wave_hand.avi       ← Class: "wave_hand"
  └── deep_folder/
      └── sign_name.mov       ← Class: "sign_name"
```

**Gesture Name Extraction Rules:**
- `hello.mp4` → gesture class: `hello`
- `hello(1).mp4`, `hello(2).mp4` → both → gesture class: `hello`
- `hello_1.mp4`, `hello_v1.mp4` → both → gesture class: `hello`
- `sign_name_v2.mp4` → gesture class: `sign_name`

**Benefits:** Multiple recordings of same gesture automatically consolidated into one folder → cleaner dataset, no folder duplication.

### 2. Convert Videos to Sequences

```powershell
# Convert all training videos with sliding windows (default: 10-frame windows, stride=5)
python app.py convert-all

# Or customize window parameters:
python app.py convert-all --window-size 15 --stride 7

# Check what was converted
python app.py dataset-stats
```

**Window options:**
- `--window-size`: Frames per sequence (default 10 ≈ 0.33s at 30fps)
- `--stride`: Step between windows (default 5 = 50% overlap)
  - Smaller stride = more overlap = more training data
  - Larger stride = less overlap = fewer training samples

Output structure (consolidation example):
```
Input videos:
  hello(1).mp4  (100 frames)  → 16 windows → sequences 0-15
  hello(2).mp4  (120 frames)  → 19 windows → sequences 16-34
  hello_v1.mp4  (110 frames)  → 17 windows → sequences 35-51
                                ↓
data/dataset/
  ├── hello/               ← All hello variations consolidated here!
  │   ├── sequence_0.csv   ← From hello(1), window 0
  │   ├── sequence_1.csv   ← From hello(1), window 1
  │   ├── ...
  │   ├── sequence_16.csv  ← From hello(2), window 0  (sequence IDs continuous)
  │   └── ...
  ├── goodbye/
  │   ├── sequence_0.csv
  │   └── ...
  └── [other gestures]
```

**Result:** 52 training sequences from 3 video files, all labeled as one gesture class.

### 3. Train Model

```powershell
python app.py train
```

### 4. Evaluate Performance

```powershell
python app.py evaluate
```

### 5. Run Real-Time Translator

```powershell
python app.py run
```

---

## Commands

### Data Management
```powershell
python app.py convert-train      # Convert handsigns/ (images+videos) → dataset/
python app.py convert-test       # Convert fortesting/ (images+videos) → test_dataset/
python app.py convert-all        # Convert both
python app.py dataset-stats      # Show dataset summary
```

### Training & Evaluation
```powershell
python app.py train              # Train model on dataset/
python app.py evaluate           # Evaluate on test_dataset/
python app.py evaluate --model models/model_20260111_085247  # Specific model
```

### Real-Time
```powershell
python app.py run                # Live gesture recognition
```

### Legacy (Real-Time Data Collection)
```powershell
python app.py collect --gesture hello           # Live capture gestures
python app.py collect-sentence --sentence "hi"  # Capture full sentence
python app.py convert                           # Legacy conversion
```

### Other
```powershell
python app.py gpu-check          # Check GPU availability
```

---

## Model Architecture

**Components:**
- Bidirectional LSTM (128→64 units) - captures temporal patterns
- Attention mechanism - focuses on important frames
- Dropout (0.3-0.5) - regularization
- Batch normalization - stable training
- Learning rate schedule - exponential decay

**Training Settings:**
- Loss: Sparse categorical crossentropy
- Optimizer: Adam with exponential decay (0.0005 initial)
- Batch size: 50
- Validation split: 20%
- Early stopping: 10 epochs patience

**Input Data:**
- 30 frames per sequence
- 126 features per frame (2 hands × 21 landmarks × 3D coordinates)
- Normalized hand landmarks (MediaPipe raw coordinates)

---

## Expected Results

After proper training and evaluation you should see:
- ✓ Training accuracy: 85%+
- ✓ Test accuracy: 75%+
- ✓ Per-gesture confusion matrix showing clear distinctions
- ✓ Real-time translation with smooth gesture recognition

---

## Troubleshooting

**No training data after convert-train:**
- Ensure images exist in `data/handsigns/<gesture>/`
- Check image formats (jpg, png, jpeg supported)
- Review conversion log: `data/conversion_log.csv`

**Low evaluation accuracy:**
- Collect more training images (50+ per gesture recommended)
- Check image quality and lighting
- Ensure consistent gesture performance
- Review per-class metrics for weak classes

**No hands detected in images:**
- Check image lighting and hand visibility
- Lower detection confidence in `extract_landmarks_from_image_file()`
- Verify MediaPipe version: `pip list | grep mediapipe`

**Memory errors during training:**
- Reduce batch size in `model_train.py`
- Enable GPU memory growth: Already configured
- Split training into smaller batches

**Real-time translator not working:**
- Check webcam permissions
- Try different camera with `cv2.VideoCapture(1)` 
- Verify model exists in `models/` directory

---

## Workspace Hygiene

Keep the repo lean before adding new data:

```powershell
# Dry run (shows what would be removed)
.venv\Scripts\python tools/clean_workspace.py

# Apply cleanup (caches, pyc, debug detection); keeps data/models
.venv\Scripts\python tools/clean_workspace.py --apply

# Include heavier cleanup (removes model logs and eval PNGs; keeps CSV metrics)
.venv\Scripts\python tools/clean_workspace.py --apply --include-heavy
```

What it leaves untouched by default: datasets, models, logs/metrics, notebooks. Use the heavy flag only if you want to prune TensorBoard logs and plots.

---

## Setup (Windows PowerShell)

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Check GPU (optional)
python app.py gpu-check

# Run workflow
python app.py convert-all
python app.py train
python app.py evaluate
python app.py run
```

---

## File Structure

```
project/
├── app.py                          # Main entry point
├── requirements.txt
├── README.md
│
├── src/
│   ├── data_collection.py          # Image to sequence conversion
│   ├── dataset_manager.py          # NEW: Train/test pipeline manager
│   ├── preprocess.py               # Data loading & preprocessing
│   ├── model_architecture.py       # Model definition
│   ├── model_train.py              # Training loop
│   ├── real_time_translator.py     # Real-time inference
│   ├── evaluate_model.py           # NEW: Test evaluation
│   ├── gpu_check.py
│   └── verify_dataset.py
│
├── data/
│   ├── handsigns/                  # Training images
│   │   ├── bad/ ├── hello/
│   │   └── [gestures]
│   │
│   ├── dataset/                    # Training sequences (auto)
│   │   ├── bad/ ├── hello/
│   │   └── [gestures]
│   │
│   ├── fortesting/                 # Test images
│   │   ├── bad/ ├── hello/
│   │   └── [gestures]
│   │
│   ├── test_dataset/               # Test sequences (auto)
│   │   ├── bad/ ├── hello/
│   │   └── [gestures]
│   │
│   ├── evaluation_results/         # Evaluation outputs (auto)
│   │   ├── metrics.csv
│   │   ├── confusion_matrix.csv
│   │   └── *.png
│   │
│   ├── sentences/                  # Sentence-level data (optional)
│   └── conversion_log.csv
│
├── models/                         # Trained models
│   └── model_<timestamp>/
│       ├── model.keras
│       ├── label_encoder.joblib
│       ├── classes.txt
│       └── training_history.csv
│
└── notebooks/
    └── Workflow.ipynb
```

---

## License

MIT
