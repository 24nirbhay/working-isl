# Root Cause Analysis: Model Not Detecting Gestures

## What We Found

### 1. Model Loads Successfully ✓
- Model file: `models/model_20260112_130741/model.keras`
- Label encoder: `models/model_20260112_130741/label_encoder.joblib`
- Custom layer (ReduceSumLayer): Properly registered ✓
- Model predictions work: ✓

### 2. **CRITICAL: Model Accuracy is Only 33%** ✗
- Tested on training data: 3 out of 9 samples correct
- Validation accuracy (during training): STUCK AT 33.33% (random guessing for 3 classes!)
- Training accuracy: Fluctuating 21-46%, never improving

### 3. **ROOT CAUSE: Dataset Too Small**
- Gesture '1': 14 sequences
- Gesture '2': 13 sequences  
- Gesture 'beautiful': 14 sequences
- **Total: Only 41 samples**

**Problem:** Neural networks need 50-100+ samples per class to learn temporal patterns.  
With only 14 samples per gesture, the model can't distinguish between them.

### 4. Why Model Didn't Learn

The training history shows:
```
Epoch  | Train Acc | Val Acc | Loss  | Notes
-------|-----------|---------|-------|-------
1-50   | 21-46%    | 33.33%  | ~3.7  | Random guessing - no learning!
```

- Validation accuracy **stayed at 33.33%** (1/3 random guessing) for ALL 50 epochs
- This means the model never learned ANY pattern
- Loss barely decreased

## The Fix

### Step 1: Collect More Training Data
```powershell
# Collect 50 samples per gesture (minimum for reliable learning)
python app.py collect --gesture 1 --samples 50
python app.py collect --gesture 2 --samples 50
python app.py collect --gesture beautiful --samples 50
```

Collection tips:
- Vary lighting, angles, clothing
- Perform gestures at different speeds  
- Record multiple variations
- Ensure clear hand visibility

### Step 2: Convert Videos to Sequences
```powershell
python app.py convert-all
```

This creates sliding windows (10-frame windows, 50% overlap):
- Each video generates ~16 training samples (overlapping windows)
- 50 videos × 16 windows = 800 samples per gesture
- Much better for model learning!

### Step 3: Retrain Model
```powershell
python app.py train
```

Expected results:
- Training accuracy: 70-80%
- Validation accuracy: Should increase each epoch
- Loss: Should decrease consistently

### Step 4: Verify Learning
```powershell
python test_predictions.py
```

Expected output:
- Accuracy: 70%+ (not 33%)
- Model predicting correct gestures

### Step 5: Test Real-Time Recognition
```powershell
python app.py run
```

Now perform gestures:
- Should see predictions in console: `Prediction: 1 (conf=0.75)`
- Video window shows gesture name with high confidence
- When you press SPACE, captures sentence

## Why This Matters

**Sliding Windows Strategy:**
- One 30-second video at 30fps = 900 frames
- Extract 14-frame windows (captures gesture motion)
- 50% overlap (stride=7) = generates ~125 windows per video
- 50 videos × 125 windows = **6,250 training samples**
- Model can now learn robust temporal patterns!

## Checking Progress

After collection, before training:
```powershell
python app.py dataset-stats
```

Should show:
```
Data Statistics:
  Gesture '1': 100+ sequences
  Gesture '2': 100+ sequences
  Gesture 'beautiful': 100+ sequences
```

## Common Issues

### Still getting 33% accuracy after retraining?
- **Cause:** Landmarks not being extracted (hands not detected)
- **Fix:** Check `python test_load.py` output
- **Debug:** Run with debug=True: `python app.py collect --gesture test --debug`

### Predictions low confidence (<0.6)?
- **Cause:** Data still too small or inconsistent
- **Fix:** Collect even more data, aim for 100+ sequences per gesture
- **Alternative:** Reduce threshold in `src/real_time_translator.py` line 50

### Model training fails?
- **Cause:** Label encoder mismatch or model corruption
- **Fix:** Delete old model, retrain fresh: `del models/model_*; python app.py train`

## Expected Timeline

- Data collection: 20-30 minutes
- Video conversion: 5 minutes  
- Model training: 10-15 minutes
- **Total: 35-60 minutes**

After this, model should accurately recognize your gestures in real-time!
