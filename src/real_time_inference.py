import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from joblib import load
from collections import deque


def _extract_two_hand_landmarks_from_results(results):
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


def run_inference(sequence_length=30, threshold=0.6):
    # Load sequence model and label encoder
    model = tf.keras.models.load_model("models/isl_seq2seq_model.keras")
    le = load("models/tokenizer.pkl")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    seq = deque(maxlen=sequence_length)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        features = _extract_two_hand_landmarks_from_results(results)
        seq.append(features)

        # draw hands
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if len(seq) == sequence_length:
            X = np.array([list(seq)], dtype='float32')  # shape (1, seq_len, 126)
            preds = model.predict(X)
            prob = np.max(preds[0])
            idx = np.argmax(preds[0])
            if prob >= threshold:
                predicted_class = le.inverse_transform([idx])[0]
                text = f"{predicted_class} ({prob*100:.1f}%)"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("ISL to English Translator - Two Hands", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_inference()
