import cv2
import mediapipe as mp
import csv
import os
import glob

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = "data/dataset"


def _extract_two_hand_landmarks_from_results(results):
    """Return a flattened vector of length 126 (2 hands x 21 landmarks x 3 coords)
    Order: [Left(21*3), Right(21*3)]. If a hand is missing, its block is zeros.
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
    Each sequence is saved as a CSV with sequence_length rows of 126 floats.
    Press SPACE to start recording, SPACE again to save and move to next sequence.
    Press Q to quit.
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Set to 60fps
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    os.makedirs(os.path.join(DATA_PATH, gesture_name), exist_ok=True)

    collected_count = 0
    recording = False
    sequence = []
    
    print(f"\n{'='*60}")
    print(f"Collecting data for gesture: {gesture_name}")
    print(f"Target: {num_samples} sequences of {sequence_length} frames each")
    print(f"{'='*60}")
    print("\nControls:")
    print("  SPACE - Start/Stop recording a sequence")
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

        # Draw hand landmarks
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show recording status
        if recording:
            status_text = f"RECORDING... {len(sequence)}/{sequence_length}"
            status_color = (0, 0, 255)  # Red
            cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)  # Red dot
        else:
            status_text = f"Ready - Press SPACE to record ({collected_count}/{num_samples})"
            status_color = (0, 255, 0)  # Green
            cv2.circle(frame, (30, 30), 15, (0, 255, 0), -1)  # Green dot
        
        cv2.putText(frame, status_text, (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, "Q - Quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # If recording, collect landmarks
        if recording:
            landmarks = _extract_two_hand_landmarks_from_results(results)
            sequence.append(landmarks)
            
            # Auto-save when sequence is full
            if len(sequence) >= sequence_length:
                file_path = os.path.join(DATA_PATH, gesture_name, f"sequence_{collected_count}.csv")
                with open(file_path, mode='w', newline='') as f:
                    csv.writer(f).writerows(sequence)
                
                collected_count += 1
                print(f"✓ Saved sequence {collected_count}/{num_samples}")
                
                # Reset for next sequence
                sequence = []
                recording = False
                
                if collected_count < num_samples:
                    print(f"Press SPACE to record next sequence ({collected_count}/{num_samples})")

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
                print(f"\nRecording sequence {collected_count + 1}...")
            else:
                # Stop recording early (user wants to restart)
                print(f"Recording stopped (only {len(sequence)} frames). Press SPACE to start again.")
                sequence = []
                recording = False

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print(f"\n{'='*60}")
    print(f"Collection complete: {collected_count}/{num_samples} sequences saved")
    print(f"Location: {os.path.join(DATA_PATH, gesture_name)}")
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
