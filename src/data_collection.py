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


def collect_data(gesture_name, num_samples=10, sequence_length=30):
    """Collect sequences from webcam. Each sequence row is 126 floats (left+right).
    Saves CSVs under data/dataset/<gesture_name>/sequence_*.csv with sequence_length rows each.
    """
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=2)
    os.makedirs(os.path.join(DATA_PATH, gesture_name), exist_ok=True)

    for i in range(num_samples):
        sequence = []
        while len(sequence) < sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            landmarks = _extract_two_hand_landmarks_from_results(results)
            sequence.append(landmarks)

            # draw whichever hands are present for feedback
            if results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Collecting Data", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        file_path = os.path.join(DATA_PATH, gesture_name, f"sequence_{i}.csv")
        with open(file_path, mode='w', newline='') as f:
            csv.writer(f).writerows(sequence)


def extract_landmarks_from_image_file(image_path):
    """Process a single image file and return a 126-length landmark list."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
    results = hands.process(img_rgb)
    landmarks = _extract_two_hand_landmarks_from_results(results)
    hands.close()
    return landmarks


def process_image_dataset(src_root, dst_root=DATA_PATH):
    """Walk src_root/<label>/*.(jpg|png) and write per-image CSVs to dst_root/<label>/sequence_*.csv
    This converts image datasets into the same CSV sequence format (single-row sequences) used by the trainer.
    """
    os.makedirs(dst_root, exist_ok=True)
    for label_dir in os.listdir(src_root):
        src_label_path = os.path.join(src_root, label_dir)
        if not os.path.isdir(src_label_path):
            continue
        dst_label_path = os.path.join(dst_root, label_dir)
        os.makedirs(dst_label_path, exist_ok=True)

        images = glob.glob(os.path.join(src_label_path, "*.jpg")) + glob.glob(os.path.join(src_label_path, "*.png"))
        for idx, img_path in enumerate(images):
            landmarks = extract_landmarks_from_image_file(img_path)
            if landmarks is None:
                continue
            file_path = os.path.join(dst_label_path, f"sequence_{idx}.csv")
            with open(file_path, mode='w', newline='') as f:
                csv.writer(f).writerow(landmarks)
