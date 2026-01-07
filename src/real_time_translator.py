import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from src.model_architecture import ReduceSumLayer
import glob
import os
import joblib

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class RealTimeTranslator:
    def __init__(self, model_path, max_sequence_length=30):
        """Initialize the real-time translator.
        
        Args:
            model_path: Path to the trained model directory
            max_sequence_length: Maximum sequence length for padding
        """
        self.max_sequence_length = max_sequence_length
        self.current_sequence = []
        self.sentence = []
        self.current_confidence = 0.0  # Track current prediction confidence
        self.capturing = False  # Sentence-level capture state
        
        # Load model with custom objects
        model_file = os.path.join(model_path, 'model.keras')
        self.model = keras.models.load_model(model_file, custom_objects={'ReduceSumLayer': ReduceSumLayer})
        
        # Load label encoder
        label_encoder_file = os.path.join(model_path, 'label_encoder.joblib')
        self.label_encoder = joblib.load(label_encoder_file)
        
        # Detection parameters
        self.threshold = 0.6  # Confidence threshold
        self.min_consecutive_frames = 3  # Frames needed for detection (reduced for responsiveness)
        self.consecutive_count = 0
        self.last_prediction = None
        self.gesture_cooldown = 0  # Cooldown to prevent duplicate detections
        self.cooldown_frames = 15  # Number of frames to wait
        
        print(f"Model loaded from {model_path}")
        print(f"Gestures: {list(self.label_encoder.classes_)}")
        print("\nControls:")
        print("  SPACE - Start/Stop sentence capture")
        print("  C     - Clear current sentence (while capturing)")
        print("  Q     - Quit")
        print("\nPress SPACE to begin capturing a sentence of gestures.")
        self._ensure_translation_file()

    def _ensure_translation_file(self):
        if not os.path.exists("translations.txt"):
            with open("translations.txt", "w", encoding="utf-8") as f:
                pass

    def _update_translation_file(self, finalize=False):
        """Update the translations.txt file with current sentence.
        While capturing, we overwrite the last line with the in-progress sentence.
        On finalize we append a newline (persist sentence)."""
        sentence_text = " ".join(self.sentence)
        if not os.path.exists("translations.txt"):
            self._ensure_translation_file()
        with open("translations.txt", "r", encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f.readlines()]
        if finalize:
            # Append finalized sentence
            if sentence_text:
                lines.append(sentence_text)
        else:
            # Overwrite last line with in-progress sentence
            if lines and (self.capturing or sentence_text):
                lines[-1] = sentence_text
            else:
                # Start a new in-progress line
                lines.append(sentence_text)
        # Write all lines with newline termination for clarity in viewers
        with open("translations.txt", "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
    
    def _extract_hand_landmarks(self, results):
        """Extract hand landmarks from MediaPipe results.
        Returns a flattened vector of length 126 (2 hands x 21 landmarks x 3 coords).
        Uses raw coordinates (matching data collection).
        """
        single_hand_len = 21 * 3
        left = [0.0] * single_hand_len
        right = [0.0] * single_hand_len

        if not results or not results.multi_hand_landmarks:
            return left + right

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, getattr(results, 'multi_handedness', [])):
            label = None
            try:
                label = handedness.classification[0].label
            except Exception:
                label = None
            
            landmarks = []
            for lm in hand_landmarks.landmark:
                # Use raw coordinates (matching data collection)
                landmarks.extend([lm.x, lm.y, lm.z])

            if label == 'Left':
                left = landmarks
            elif label == 'Right':
                right = landmarks
            else:
                if sum(left) == 0:
                    left = landmarks
                else:
                    right = landmarks

        return left + right
    
    def _predict_gesture(self, sequence):
        """Predict gesture from sequence of landmarks."""
        if len(sequence) == 0:
            return None, 0.0
        
        # Pad sequence to max length
        padded = np.zeros((self.max_sequence_length, 126))
        seq_len = min(len(sequence), self.max_sequence_length)
        padded[:seq_len] = sequence[-seq_len:]
        
        # Predict
        input_data = np.expand_dims(padded, axis=0)
        predictions = self.model.predict(input_data, verbose=0)
        
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        
        if confidence >= self.threshold:
            gesture = self.label_encoder.inverse_transform([predicted_idx])[0]
            return gesture, confidence
        
        return None, confidence
    
    def process_frame(self, frame, results):
        """Process a single frame and update detection state.
        
        Args:
            frame: The current video frame
            results: MediaPipe hand detection results
            
        Returns:
            frame: The annotated frame
        """
        # Extract landmarks
        landmarks = self._extract_hand_landmarks(results)
        
        # Draw hand landmarks with MediaPipe style (connections and all)
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Only collect and predict gestures if currently capturing
        if self.capturing and sum(landmarks) != 0:
            self.current_sequence.append(landmarks)
            if len(self.current_sequence) > self.max_sequence_length:
                self.current_sequence.pop(0)
        
        # Decrement cooldown
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
        
        # Continuous prediction if we have enough frames, no cooldown, and capturing
        if self.capturing and len(self.current_sequence) >= 5 and self.gesture_cooldown == 0:
            gesture, confidence = self._predict_gesture(self.current_sequence)
            self.current_confidence = confidence  # Store for UI display
            # Debug print for predicted gesture (helps troubleshoot missing updates)
            if confidence >= self.threshold:
                try:
                    print(f"Prediction: {gesture} (conf={confidence:.2f})")
                except Exception:
                    print(f"Prediction: idx? (conf={confidence:.2f})")
            
            if gesture:
                if gesture == self.last_prediction:
                    self.consecutive_count += 1
                else:
                    self.consecutive_count = 1
                    self.last_prediction = gesture
                
                # Add to sentence if we have enough consecutive detections
                if self.consecutive_count >= self.min_consecutive_frames:
                    self.sentence.append(gesture)
                    self.current_sequence = []
                    self.consecutive_count = 0
                    self.last_prediction = None
                    self.gesture_cooldown = self.cooldown_frames  # Start cooldown
                    print(f"  Detected: {gesture} ({confidence:.2f})")
                    # Update translation file live
                    self._update_translation_file(finalize=False)
            else:
                self.consecutive_count = 0
                self.last_prediction = None
        else:
            if len(self.current_sequence) < 5:
                self.current_confidence = 0.0
        
        # Draw UI
        self._draw_ui(frame, results)
        
        return frame
    
    def _draw_ui(self, frame, results):
        """Draw UI elements on the frame."""
        h, w, _ = frame.shape
        
        # Top-left controls (smaller font - 5px)
        status = "CAPTURING" if self.capturing else "IDLE"
        controls_text = f"SPACE start/stop | C clear | Q quit ({status})"
        cv2.putText(frame, controls_text, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Top-right: Hand detection confidence (8px font)
        hand_confidence = 0.0
        if results and results.multi_hand_landmarks and results.multi_handedness:
            # Get the highest confidence from detected hands
            for handedness in results.multi_handedness:
                try:
                    score = handedness.classification[0].score
                    hand_confidence = max(hand_confidence, score)
                except Exception:
                    pass
        
        confidence_text = f"Hand: {hand_confidence:.2f}"
        cv2.putText(frame, confidence_text, (w - 120, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show prediction confidence if available
        if self.current_confidence > 0.0:
            pred_text = f"Pred: {self.current_confidence:.2f}"
            cv2.putText(frame, pred_text, (w - 120, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Current sentence (bottom) - display directly without "Sentence:" prefix
        sentence_text = " ".join(self.sentence) if self.sentence else ""
        if sentence_text:
            cv2.putText(frame, sentence_text, (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def finalize_sentence(self):
        """Finalize current sentence (stop capturing if active)."""
        if self.capturing:
            self.capturing = False
        if self.sentence:
            sentence_text = " ".join(self.sentence)
            print(f"\n✓ Finalized: {sentence_text}\n")
            self._update_translation_file(finalize=True)
        self.sentence = []
        self.current_sequence = []
        self.current_confidence = 0.0
    
    def clear_sentence(self):
        """Clear the current sentence without saving."""
        self.sentence = []
        self.current_sequence = []
        self.consecutive_count = 0
        self.last_prediction = None
        self._update_translation_file(finalize=False)
        print("\n✗ Sentence cleared\n")


def main():
    """Main function to run the real-time translator."""
    # Find the most recent model
    model_dirs = glob.glob("models/model_*")
    if not model_dirs:
        print("No trained model found. Please train a model first using: python app.py train")
        return
    
    latest_model = max(model_dirs, key=os.path.getmtime)
    print(f"Using model: {latest_model}")
    
    translator = RealTimeTranslator(latest_model)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Process frame
            frame = translator.process_frame(frame, results)
            
            # Display
            cv2.imshow('ISL to English Translator', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space toggles capture / finalize sentence
                if not translator.capturing:
                    translator.capturing = True
                    translator.sentence = []
                    translator.current_sequence = []
                    # Reset detection state
                    translator.consecutive_count = 0
                    translator.last_prediction = None
                    translator.gesture_cooldown = 0
                    translator.current_confidence = 0.0
                    translator._update_translation_file(finalize=False)
                    print("\n● Started sentence capture. Perform gestures... Press SPACE to stop.\n")
                else:
                    translator.finalize_sentence()
            elif key == ord('c'):  # C to clear
                translator.clear_sentence()
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
