# ISL to English Translator

Converts Indian Sign Language (ISL) gestures to English text using deep learning.

## Workflow

### Option 1: Use Existing Hand-Sign Images

If you have images in `data/handsigns/<gesture>/`:

```powershell
# Convert images to training sequences
python app.py convert

# Train the model
python app.py train

# Run real-time translator
python app.py run
```

### Option 2: Collect New Gesture Data

```powershell
# Collect data for a gesture (continuous SPACE-controlled capture)
python app.py collect --gesture <gesture_name>

# Train the model
python app.py train

# Run real-time translator
python app.py run
```

## Data Collection (New Method)

**Continuous Frame Capture:**
- Press **SPACE** to start recording
- Make your gesture continuously - frames captured every ~16ms
- Press **SPACE** again to stop and save the sequence
- Minimum 5 frames required per sequence
- Repeat 30 times (or as many as you want)

This is much faster than the old fixed 30-frame method!

## Real-Time Translation

```powershell
python app.py run
```

**Continuous Gesture Recognition:**
- Make gestures in front of camera
- Gestures are **automatically detected** and added to sentence at bottom
- No waiting - seamless continuous translation!
- **SPACE** → save and finalize current sentence
- **C** → clear current sentence (without saving)
- **Q** → quit

**Visual Indicators:**
- Green dot = hands detected
- Red dot = no hands
- Progress bar shows gesture confidence
- Current sentence displayed at bottom with dark background

**How it works:**
1. System continuously monitors hand gestures
2. When a gesture is held for 5 frames with high confidence, it's added
3. Cooldown prevents duplicate additions of same gesture
4. Build your sentence by making multiple gestures
5. Press SPACE when complete to save

## Technical Details

**Model Architecture:**
- Bidirectional LSTM with attention mechanism
- 128→64 LSTM units, L2 regularization
- Learns normalized hand coordinates (wrist-relative, unit variance)
- 50 epochs, early stopping, learning rate 0.0005

**Data Format:**
- 30 frames per sequence (variable length supported)
- 126 features per frame (2 hands × 21 landmarks × 3D coordinates)
- Normalized relative to wrist position

---

## Setup (Windows PowerShell)

1. Create virtual environment
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Convert existing images (if you have them)
```powershell
python app.py convert
```

3. Train the model
```powershell
python app.py train
```

4. Run translator
```powershell
python app.py run
```

## Commands

- `python app.py collect --gesture <name>` - Collect new gesture data (SPACE start/stop)
- `python app.py convert` - Convert images in `data/handsigns/` to sequences
- `python app.py train` - Train model on all gestures in `data/dataset/`
- `python app.py run` - Real-time gesture translation
- `python app.py gpu-check` - Check GPU availability

## Troubleshooting

- **No hands detected:** Check lighting, ensure hands are visible, green dot shows detection
- **Low accuracy:** Collect more sequences (50+ per gesture recommended)
- **MediaPipe issues:** Ensure `mediapipe==0.10.20`, `protobuf==4.25.3`
- **TensorFlow errors:** Verify `tensorflow==2.17.0`

License: MIT
