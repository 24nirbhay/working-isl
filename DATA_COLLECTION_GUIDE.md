# DATA COLLECTION & RETRAINING GUIDE

## Current Problem
- Model accuracy: 33% (random guessing)
- Validation accuracy: 33.33% (constant for all 50 epochs!)
- Root cause: Only ~40 total samples (14-14-13 per gesture) - TOO LITTLE DATA

## Solution: Collect More Data

### Quick Recolection Protocol
1. **Setup**:
   ```
   python app.py collect --gesture 1 --samples 50
   python app.py collect --gesture 2 --samples 50  
   python app.py collect --gesture beautiful --samples 50
   ```
   Target: 50 samples per gesture (total 150 samples)

2. **Collection Tips**:
   - Wear different colored clothes each session
   - Change lighting conditions
   - Perform gestures at different speeds
   - Record from different angles (move camera)
   - Perform variations of each gesture
   - Ensure hands clearly visible in frame

3. **After Collection**:
   ```
   python app.py convert-all
   python app.py train
   ```

## Expected Results After Retraining
- With 150+ samples: Expected 70-80% training accuracy
- With 200+ samples: Expected 85-90% training accuracy
- Validation accuracy should increase significantly

## Monitoring Training
- Watch console output for epoch accuracy
- Loss should decrease noticeably (not flat like before)
- Validation accuracy should improve
- If accuracy plateaus, model has learned all it can from available data

## If Still Not Working
1. Check landmark extraction: `python diagnose_model.py`
2. Verify data was collected: `python app.py dataset-stats`
3. Check real-time detection: `python app.py run` and watch console for predictions
