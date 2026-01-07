import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import glob

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = "data/dataset"
SENTENCE_PATH = "data/sentences"


def _extract_two_hand_landmarks_from_results(results):
    """Return a flattened vector of length 126 (2 hands x 21 landmarks x 3 coords)
    Order: [Left(21*3), Right(21*3)]. If a hand is missing, its block is zeros.
    Uses raw coordinates (no normalization) to match real-time translator.
    """
    # initialize zeros for left and right
    single_hand_len = 21 * 3
    left = [0.0] * single_hand_len
    right = [0.0] * single_hand_len

    if not results or not results.multi_hand_landmarks:
        return left + right

    # results.multi_handedness aligns with multi_hand_landmarks
    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, getattr(results, 'multi_handedness', [])):
        label = None
        try:
            label = handedness.classification[0].label  # 'Left' or 'Right'
        except Exception:
            label = None
        
        landmarks = []
        for lm in hand_landmarks.landmark:
            # Use raw coordinates (matching real-time translator)
            landmarks.extend([lm.x, lm.y, lm.z])

        if label == 'Left':
            left = landmarks
        elif label == 'Right':
            right = landmarks
        else:
            # If handedness not available, fill whichever is empty first
            if sum(left) == 0:
                left = landmarks
            else:
                right = landmarks

    return left + right


def collect_data(gesture_name, num_samples=30, sequence_length=30):
    """Collect sequences from webcam using SPACE to start/stop recording.
    Continuously captures frames while recording. Press SPACE to start, SPACE again to save.
    Press Q to quit.
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Set to 60fps
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.3)
    os.makedirs(os.path.join(DATA_PATH, gesture_name), exist_ok=True)

    collected_count = 0
    recording = False
    sequence = []
    
    print(f"\n{'='*60}")
    print(f"Collecting data for gesture: {gesture_name}")
    print(f"Target: {num_samples} sequences (continuous seamless capture)")
    print(f"{'='*60}")
    print("\nControls:")
    print("  SPACE - Start recording / Stop and save sequence")
    print("  Q     - Quit collection")
    print(f"\nCollected: 0/{num_samples}")
    print("Ready... Press SPACE to start recording first sequence")
    print(f"{'='*60}\n")

    while collected_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        h, w = frame.shape[:2]

        # Draw hand landmarks
        hands_detected = False
        hand_confidence = 0.0
        if results and results.multi_hand_landmarks:
            hands_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get hand detection confidence
            if results.multi_handedness:
                for handedness in results.multi_handedness:
                    try:
                        score = handedness.classification[0].score
                        hand_confidence = max(hand_confidence, score)
                    except Exception:
                        pass

        # Show hand detection indicator (top-left corner) - smaller
        if hands_detected:
            cv2.circle(frame, (20, 20), 8, (0, 255, 0), -1)  # Green dot
        else:
            cv2.circle(frame, (20, 20), 8, (0, 0, 255), -1)  # Red dot
        
        # Show instructions at top-left (5px font)
        instructions = "SPACE - Start/Stop | Q - Quit"
        cv2.putText(frame, instructions, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Show hand detection confidence at top-right (8px font)
        confidence_text = f"Hand: {hand_confidence:.2f}"
        cv2.putText(frame, confidence_text, (w - 120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show status next to dot
        if recording:
            status_text = f"RECORDING... {len(sequence)} frames"
            status_color = (0, 0, 255)  # Red
        else:
            status_text = f"Ready ({collected_count}/{num_samples} collected)"
            status_color = (0, 255, 0)  # Green
        
        cv2.putText(frame, status_text, (40, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # If recording, continuously collect landmarks
        if recording:
            landmarks = _extract_two_hand_landmarks_from_results(results)
            sequence.append(landmarks)
            # Debug: check if we're getting real data
            if len(sequence) % 30 == 0 and sum(landmarks) != 0:
                print(f"  Frame {len(sequence)}: capturing hand data...")

        cv2.imshow("Collecting Data", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nCollection cancelled by user")
            break
        elif key == ord(' '):  # Spacebar
            if not recording:
                # Start recording
                recording = True
                sequence = []
                print(f"\n● Recording sequence {collected_count + 1}... (Press SPACE to stop)")
            else:
                # Stop recording and save
                if len(sequence) >= 5:  # Minimum 5 frames
                    file_path = os.path.join(DATA_PATH, gesture_name, f"sequence_{collected_count}.csv")
                    with open(file_path, mode='w', newline='', encoding='utf-8') as f:
                        csv.writer(f).writerows(sequence)
                    
                    collected_count += 1
                    print(f"✓ Saved sequence {collected_count}/{num_samples} ({len(sequence)} frames)")
                    
                    if collected_count < num_samples:
                        print(f"Press SPACE to record next sequence")
                else:
                    print(f"✗ Sequence too short ({len(sequence)} frames). Need at least 5 frames.")
                
                # Reset for next sequence
                sequence = []
                recording = False

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print(f"\n{'='*60}")
    print(f"Collection complete: {collected_count}/{num_samples} sequences saved")
    print(f"Location: {os.path.join(DATA_PATH, gesture_name)}")
    print(f"{'='*60}\n")


def collect_sentence(sentence_label):
    """Capture a full sentence-level gesture stream.

    SPACE toggles start/stop capture. While capturing we record every frame's
    landmarks (126 features). On stop a single CSV sequence is written under
    data/sentences/<sentence_label>/sequence_<n>.csv containing all frames.
    Also append/update sentences.csv with mapping: sentence_label,frames,count.
    """
    os.makedirs(SENTENCE_PATH, exist_ok=True)
    sentence_dir = os.path.join(SENTENCE_PATH, sentence_label)
    os.makedirs(sentence_dir, exist_ok=True)

    # Determine next sequence index
    existing = [f for f in os.listdir(sentence_dir) if f.startswith("sequence_") and f.endswith(".csv")]
    next_idx = 0
    if existing:
        try:
            nums = [int(f.replace("sequence_", "").replace(".csv", "")) for f in existing]
            next_idx = max(nums) + 1
        except Exception:
            pass

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    capturing = False
    frames = []
    print(f"\n{'='*60}")
    print(f"Sentence capture for label: '{sentence_label}'")
    print("Press SPACE to start, SPACE again to stop & save, Q to quit.")
    print(f"Output directory: {sentence_dir}")
    print(f"{'='*60}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        h, w = frame.shape[:2]

        # Draw landmarks
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Status dot
        cv2.circle(frame, (20,20), 8, (0,0,255) if not capturing else (0,255,0), -1)

        # Instructions & state
        instr = "SPACE start/stop | Q quit"
        cv2.putText(frame, instr, (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        state = f"Capturing frames: {len(frames)}" if capturing else "Idle"
        cv2.putText(frame, state, (40,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Record if capturing
        if capturing:
            frames.append(_extract_two_hand_landmarks_from_results(results))

        cv2.imshow("Sentence Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if not capturing:
                capturing = True
                frames = []
                print("● Started sentence capture. Press SPACE to finish.")
            else:
                capturing = False
                if len(frames) < 5:
                    print(f"✗ Capture too short ({len(frames)} frames). Discarded.")
                    frames = []
                    continue
                # Save sequence
                out_file = os.path.join(sentence_dir, f"sequence_{next_idx}.csv")
                with open(out_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(frames)
                print(f"✓ Saved sentence sequence '{out_file}' ({len(frames)} frames)")
                # Update sentences.csv summary
                summary_path = os.path.join(SENTENCE_PATH, 'sentences.csv')
                header_needed = not os.path.exists(summary_path)
                with open(summary_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if header_needed:
                        writer.writerow(["sentence_label","sequence_index","frames"])
                    writer.writerow([sentence_label, next_idx, len(frames)])
                next_idx += 1
                frames = []
                print("Press SPACE to record another sentence stream or Q to quit.")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Sentence capture ended.")


def convert_images_to_sequences(images_root="data/handsigns", output_root=DATA_PATH, frames_per_sequence=30):
    """Convert existing hand-sign images into sequences for training.
    
    Takes images from images_root/<gesture>/*.{jpg,png} and creates sequences by:
    1. Processing each image to extract landmarks
    2. Grouping images into sequences of specified length
    3. Saving as CSV files in output_root/<gesture>/sequence_*.csv
    
    Args:
        images_root: Path to directory containing gesture subdirectories with images
        output_root: Path to save generated sequence CSVs
        frames_per_sequence: Number of images to group into each sequence
    """
    if not os.path.exists(images_root):
        print(f"Error: Images directory not found: {images_root}")
        return
    
    os.makedirs(output_root, exist_ok=True)
    total_sequences = 0
    total_failed = 0
    
    print(f"\n{'='*60}")
    print(f"Converting images to sequences")
    print(f"Source: {images_root}")
    print(f"Output: {output_root}")
    print(f"Frames per sequence: {frames_per_sequence}")
    print(f"{'='*60}\n")
    
    for gesture_name in sorted(os.listdir(images_root)):
        gesture_dir = os.path.join(images_root, gesture_name)
        if not os.path.isdir(gesture_dir):
            continue
        
        # Get all image files
        image_files = sorted(
            glob.glob(os.path.join(gesture_dir, "*.jpg")) + 
            glob.glob(os.path.join(gesture_dir, "*.png"))
        )
        
        if not image_files:
            print(f"No images found for gesture '{gesture_name}'")
            continue
        
        print(f"\nProcessing gesture '{gesture_name}': {len(image_files)} images")
        
        # Create output directory
        output_dir = os.path.join(output_root, gesture_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Process images and extract landmarks
        landmarks_list = []
        for img_path in image_files:
            landmarks = extract_landmarks_from_image_file(img_path, debug=False)
            if landmarks is not None and sum(abs(x) for x in landmarks) > 0:  # Valid landmarks
                landmarks_list.append(landmarks)
            else:
                total_failed += 1
        
        if not landmarks_list:
            print(f"  ✗ No valid landmarks extracted for gesture '{gesture_name}'")
            continue
        
        print(f"  ✓ Extracted landmarks from {len(landmarks_list)} images")
        
        # Group into sequences
        num_sequences = len(landmarks_list) // frames_per_sequence
        if num_sequences == 0:
            # If we don't have enough for a full sequence, create one by repeating
            sequence = landmarks_list * (frames_per_sequence // len(landmarks_list) + 1)
            sequence = sequence[:frames_per_sequence]
            
            file_path = os.path.join(output_dir, "sequence_0.csv")
            with open(file_path, mode='w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerows(sequence)
            total_sequences += 1
            print(f"  ✓ Created 1 sequence (repeated {len(landmarks_list)} landmarks to fill)")
        else:
            # Create multiple sequences
            for seq_idx in range(num_sequences):
                start_idx = seq_idx * frames_per_sequence
                end_idx = start_idx + frames_per_sequence
                sequence = landmarks_list[start_idx:end_idx]
                
                file_path = os.path.join(output_dir, f"sequence_{seq_idx}.csv")
                with open(file_path, mode='w', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerows(sequence)
                total_sequences += 1
            
            remaining = len(landmarks_list) % frames_per_sequence
            print(f"  ✓ Created {num_sequences} sequences ({remaining} images unused)")
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Total sequences created: {total_sequences}")
    print(f"Failed image extractions: {total_failed}")
    print(f"{'='*60}\n")


def extract_landmarks_from_image_file(image_path, debug=False):
    """Process a single image file and return a 126-length landmark list."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None
    
    # Resize if image is too large (helps with detection)
    max_dim = 1280
    h, w = img.shape[:2]
    if h > max_dim or w > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5  # Lower this if hands aren't being detected
    )
    
    results = hands.process(img_rgb)
    landmarks = _extract_two_hand_landmarks_from_results(results)
    
    if debug:
        debug_img = img.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(debug_img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            print(f"Detected {len(results.multi_hand_landmarks)} hands in {image_path}")
        else:
            print(f"No hands detected in {image_path}")
        
        # Save debug image
        debug_dir = os.path.join("data", "debug_detection")
        os.makedirs(debug_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        debug_path = os.path.join(debug_dir, f"{base_name}_debug.jpg")
        cv2.imwrite(debug_path, debug_img)
    
    hands.close()
    return landmarks


def split_large_sequence(gesture_name, sequence_file, target_length=30, overlap=10):
    """Split a large sequence file into multiple smaller sequences.
    
    Args:
        gesture_name: Name of the gesture
        sequence_file: Path to the large sequence CSV file
        target_length: Target length for each split sequence (default: 30)
        overlap: Number of overlapping frames between sequences (default: 10)
    """
    import pandas as pd
    
    gesture_dir = os.path.join(DATA_PATH, gesture_name)
    
    # Read the large sequence
    df = pd.read_csv(sequence_file, header=None)
    total_frames = len(df)
    
    print(f"\nSplitting {sequence_file}")
    print(f"Total frames: {total_frames}")
    
    # Calculate number of sequences we can create
    stride = target_length - overlap
    num_sequences = max(1, (total_frames - target_length) // stride + 1)
    
    print(f"Will create {num_sequences} sequences of ~{target_length} frames each")
    
    # Get existing sequence numbers
    existing_files = glob.glob(os.path.join(gesture_dir, "sequence_*.csv"))
    if existing_files:
        existing_nums = [int(os.path.basename(f).replace("sequence_", "").replace(".csv", "")) 
                        for f in existing_files]
        start_num = max(existing_nums) + 1
    else:
        start_num = 0
    
    # Split into sequences
    sequences_created = 0
    for i in range(num_sequences):
        start_idx = i * stride
        end_idx = min(start_idx + target_length, total_frames)
        
        if end_idx - start_idx < 10:  # Skip very short sequences
            continue
        
        sequence_data = df.iloc[start_idx:end_idx]
        
        output_file = os.path.join(gesture_dir, f"sequence_{start_num + sequences_created}.csv")
        sequence_data.to_csv(output_file, header=False, index=False)
        sequences_created += 1
        print(f"  Created sequence_{start_num + sequences_created - 1}.csv ({len(sequence_data)} frames)")
    
    print(f"\n✓ Created {sequences_created} sequences from {total_frames} frames")
    return sequences_created


def process_image_dataset(src_root, dst_root=DATA_PATH):
    """Walk src_root/<label>/*.(jpg|png) and write per-image CSVs to dst_root/<label>/sequence_*.csv
    This converts image datasets into the same CSV sequence format (single-row sequences) used by the trainer.
    """
    os.makedirs(dst_root, exist_ok=True)
    total_processed = 0
    total_failed = 0
    
    for label_dir in os.listdir(src_root):
        src_label_path = os.path.join(src_root, label_dir)
        if not os.path.isdir(src_label_path):
            continue
        dst_label_path = os.path.join(dst_root, label_dir)
        os.makedirs(dst_label_path, exist_ok=True)

        images = glob.glob(os.path.join(src_label_path, "*.jpg")) + glob.glob(os.path.join(src_label_path, "*.png"))
        print(f"\nProcessing {len(images)} images for label '{label_dir}'...")
        
        for idx, img_path in enumerate(images):
            print(f"  Processing {os.path.basename(img_path)}...", end="")
            landmarks = extract_landmarks_from_image_file(img_path, debug=True)
            if landmarks is None:
                print(" Failed!")
                total_failed += 1
                continue
            print(" Done")
            file_path = os.path.join(dst_label_path, f"sequence_{idx}.csv")
            with open(file_path, mode='w', newline='') as f:
                csv.writer(f).writerow(landmarks)
