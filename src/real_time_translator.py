import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from joblib import load
from collections import deque
import time
import os

class SignLanguageTranslator:
    def __init__(self, model_dir="models", gesture_threshold=0.4, pause_threshold=2.0):
        """Initialize the translator with models and parameters.
        
        Args:
            model_dir: Directory containing the model files
            gesture_threshold: Confidence threshold for gesture recognition
            pause_threshold: Time in seconds to wait before considering a sentence complete
        """
        # Load the latest model if model_dir is a directory
        if os.path.isdir(model_dir):
            model_dirs = [d for d in os.listdir(model_dir) if d.startswith('model_')]
            if model_dirs:
                latest_model = max(model_dirs)
                model_dir = os.path.join(model_dir, latest_model)
        
        self.model = tf.keras.models.load_model(os.path.join(model_dir, "model.keras"))
        self.le = load(os.path.join(model_dir, "label_encoder.joblib"))
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Parameters
        self.sequence_length = 30
        self.gesture_threshold = gesture_threshold
        self.pause_threshold = pause_threshold
        
        # State variables
        self.sequence = deque(maxlen=self.sequence_length)
        self.current_sentence = []
        self.last_gesture_time = time.time()
        self.last_prediction = None
        self.consecutive_frames = 0
        
        # Sentence finalization
        self.sentence_finalized = False
        self.finalized_time = 0
        self.finalized_text = ""
        self.output_file = "../isl_to_english.txt"
        
    def _extract_hand_landmarks(self, results):
        """Extract hand landmarks in the correct format."""
        single_hand_len = 21 * 3
        left = [0.0] * single_hand_len
        right = [0.0] * single_hand_len

        if not results.multi_hand_landmarks:
            return left + right

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                           results.multi_handedness):
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if handedness.classification[0].label == 'Left':
                left = landmarks
            else:
                right = landmarks

        return left + right
    
    def _predict_gesture(self):
        """Predict the current gesture from the sequence."""
        if len(self.sequence) < self.sequence_length:
            return None, 0.0
            
        X = np.array([list(self.sequence)], dtype='float32')
        preds = self.model.predict(X, verbose=0)[0]
        idx = np.argmax(preds)
        confidence = preds[idx]
        predicted_class = self.le.inverse_transform([idx])[0]
        
        # Debug: Show all predictions above 0.2
        if confidence > 0.2:
            print(f"\rDetected: {predicted_class} ({confidence*100:.1f}%) | Seq: {len(self.sequence)}/{self.sequence_length}", end='', flush=True)
        
        if confidence >= self.gesture_threshold:
            return predicted_class, confidence
        return None, confidence
    
    def process_frame(self, frame):
        """Process a single frame and update the state."""
        # Flip frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Extract landmarks and add to sequence continuously
        landmarks = self._extract_hand_landmarks(results)
        self.sequence.append(landmarks)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # Predict gesture continuously
        gesture, confidence = self._predict_gesture()
        
        # Update state continuously
        current_time = time.time()
        if gesture:
            if gesture != self.last_prediction:
                self.consecutive_frames = 1
            else:
                self.consecutive_frames += 1
                
            if self.consecutive_frames >= 5:  # Require 5 consecutive same predictions
                if not self.current_sentence or self.current_sentence[-1] != gesture:
                    self.current_sentence.append(gesture)
                    self.last_gesture_time = current_time
                
            self.last_prediction = gesture
        
        # Check if finalized sentence should be cleared (after 3 seconds)
        if self.sentence_finalized and (current_time - self.finalized_time) >= 3.0:
            self.sentence_finalized = False
            self.finalized_text = ""
            
        # Draw UI
        self._draw_ui(frame, gesture, confidence, current_time)
        
        return frame, current_time
    
    def _draw_ui(self, frame, gesture, confidence, current_time):
        """Draw the UI elements on the frame."""
        # Show hand detection status as green dot
        hands_detected = len(self.sequence) > 0 and np.sum(self.sequence[-1]) > 0
        dot_color = (0, 255, 0) if hands_detected else (0, 0, 255)
        cv2.circle(frame, (frame.shape[1] - 30, 30), 15, dot_color, -1)
        
        # Draw current sentence at bottom
        if self.current_sentence:
            sentence = " ".join(self.current_sentence)
            cv2.putText(frame, sentence, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Draw finalized sentence (green, will disappear after 3 seconds)
        if self.sentence_finalized:
            time_remaining = 3.0 - (current_time - self.finalized_time)
            cv2.putText(frame, f"SAVED: {self.finalized_text}", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Clearing in {time_remaining:.1f}s", (10, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def get_current_sentence(self):
        """Get the current sentence and reset if completed."""
        if not self.current_sentence:
            return ""
        
        sentence = " ".join(self.current_sentence)
        self.current_sentence = []
        self.last_prediction = None
        self.consecutive_frames = 0
        return sentence
    
    def finalize_sentence(self):
        """Finalize and save the current sentence."""
        if not self.current_sentence:
            return None
        
        sentence = self.get_current_sentence()
        if sentence:
            self.finalized_text = sentence
            self.sentence_finalized = True
            self.finalized_time = time.time()
            self.save_to_output_file(sentence)
            return sentence
        return None
    
    def save_to_output_file(self, sentence):
        """Save the translated sentence to output file for English-to-Konkani model."""
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(sentence + '\n')
            print(f"Saved to {self.output_file}: {sentence}")
        except Exception as e:
            print(f"Error saving to file: {e}")
    
    def __del__(self):
        """Clean up resources."""
        self.hands.close()

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    translator = SignLanguageTranslator()
    
    # Clear output file at start
    try:
        with open(translator.output_file, 'w', encoding='utf-8') as f:
            f.write('')
        print(f"Output file cleared: {translator.output_file}")
    except Exception as e:
        print(f"Warning: Could not clear output file: {e}")
    
    print("\n" + "="*60)
    print("ISL to English Translator")
    print("="*60)
    print(f"Model loaded: {len(translator.le.classes_)} gestures")
    print(f"Available gestures: {', '.join(sorted(translator.le.classes_))}")
    print(f"Gesture threshold: {translator.gesture_threshold} (lowered for easier detection)")
    print("="*60)
    print("\nControls:")
    print("  SPACE - Save current sentence and reset")
    print("  q - Quit")
    print("\nInstructions:")
    print("  1. Make gestures - they appear in real-time")
    print("  2. Hold gesture steady for best detection")
    print("  3. Watch the buffer fill up (30 frames needed)")
    print("  4. Press SPACE when you want to save the sentence")
    print(f"\nOutput saved to: {translator.output_file}")
    print("="*60 + "\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Process frame
        frame, current_time = translator.process_frame(frame)
        
        # Show frame
        cv2.imshow('ISL to English Translator', frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            # Finalize and save current sentence
            sentence = translator.finalize_sentence()
            if sentence:
                print(f"\n{'='*60}")
                print(f"SAVED: {sentence}")
                print(f"{'='*60}\n")
            else:
                print("\nNo sentence to save (make some gestures first)\n")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()