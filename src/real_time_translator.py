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
        
        # Load model with custom objects
        model_file = os.path.join(model_path, 'model.keras')
        self.model = keras.models.load_model(model_file, custom_objects={'ReduceSumLayer': ReduceSumLayer})
        
        # Load label encoder
        label_encoder_file = os.path.join(model_path, 'label_encoder.joblib')
        self.label_encoder = joblib.load(label_encoder_file)
        
        # Detection parameters
        self.threshold = 0.6  # Confidence threshold
        self.min_consecutive_frames = 5  # Frames needed for detection
        self.consecutive_count = 0
        self.last_prediction = None
        self.gesture_cooldown = 0  # Cooldown to prevent duplicate detections
        self.cooldown_frames = 15  # Number of frames to wait
        
        print(f"Model loaded from {model_path}")
        print(f"Gestures: {list(self.label_encoder.classes_)}")
        print("\nControls:")
        print("  SPACE - Finalize and save sentence")
        print("  C - Clear current sentence")
        print("  Q - Quit")
        print("\nContinuous gesture detection active...")
    
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
        
        # Add to sequence if hands detected
        if sum(landmarks) != 0:
            self.current_sequence.append(landmarks)
            # Keep only recent frames for continuous detection
            if len(self.current_sequence) > self.max_sequence_length:
                self.current_sequence.pop(0)
        
        # Decrement cooldown
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
        
        # Continuous prediction if we have enough frames and no cooldown
        if len(self.current_sequence) >= 5 and self.gesture_cooldown == 0:
            gesture, confidence = self._predict_gesture(self.current_sequence)
            self.current_confidence = confidence  # Store for UI display
            
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
            else:
                self.consecutive_count = 0
                self.last_prediction = None
        else:
            # Reset confidence when not actively predicting
            if len(self.current_sequence) < 5:
                self.current_confidence = 0.0
        
        # Draw UI
        self._draw_ui(frame, results)
        
        return frame
    
    def _draw_ui(self, frame, results):
        """Draw UI elements on the frame."""
        h, w, _ = frame.shape
        
        # Top-left controls (smaller font - 5px)
        controls_text = "SPACE: Save | C: Clear | Q: Quit"
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
        """Save and clear the current sentence."""
        if self.sentence:
            sentence_text = " ".join(self.sentence)
            print(f"\n✓ Saved: {sentence_text}\n")
            
            # Save to file
            with open("translations.txt", "a", encoding="utf-8") as f:
                f.write(sentence_text + "\n")
            
            self.sentence = []
            self.current_sequence = []
    
    def clear_sentence(self):
        """Clear the current sentence without saving."""
        self.sentence = []
        self.current_sequence = []
        self.consecutive_count = 0
        self.last_prediction = None
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
            elif key == ord(' '):  # Space to finalize
                translator.finalize_sentence()
            elif key == ord('c'):  # C to clear
                translator.clear_sentence()
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
