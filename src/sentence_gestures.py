"""
Sentence-Level Gesture Processing
Processes entire videos as sentences containing multiple consecutive gestures.
Automatically segments sequences based on landmark state transitions.
"""

import os
import csv
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class GestureSegment:
    """Represents a detected gesture within a video."""
    start_frame: int
    end_frame: int
    gesture_label: str
    confidence: float
    landmarks: List[np.ndarray]  # Frames in this segment
    

class GestureSegmenter:
    """Detects gesture boundaries from hand position/pose transitions."""
    
    def __init__(self, 
                 position_change_threshold: float = 0.25,
                 window_size: int = 5,
                 min_gesture_frames: int = 8):
        """
        Args:
            position_change_threshold: Hand position change to detect boundary (0-1)
            window_size: Frames to compute position change over
            min_gesture_frames: Minimum frames per gesture
        """
        self.position_change_threshold = position_change_threshold
        self.window_size = window_size
        self.min_gesture_frames = min_gesture_frames
        
    def compute_hand_position_change(self, landmarks_window: List[np.ndarray]) -> float:
        """
        Compute hand position change from first to last frame in window.
        
        Hand position = centroid of all landmarks (wrist + fingers).
        This captures the overall hand displacement, which is the pivot point
        for gesture boundaries.
        
        Args:
            landmarks_window: List of landmark arrays (frames)
            
        Returns:
            Position change magnitude (0-1 scale, normalized)
        """
        if len(landmarks_window) < 2:
            return 0.0
        
        first_frame = landmarks_window[0]  # Shape: (42, 3) for 2 hands
        last_frame = landmarks_window[-1]
        
        # Compute hand centroids (average position of all landmarks)
        first_centroid = np.mean(first_frame, axis=0)
        last_centroid = np.mean(last_frame, axis=0)
        
        # Euclidean distance between centroids
        position_change = np.linalg.norm(last_centroid - first_centroid)
        
        # Normalize: typical hand movement in a gesture is ~0.2-0.3 of frame
        # (frame normalized to 0-1), so cap at 0.3
        normalized_change = min(position_change / 0.3, 1.0)
        
        return normalized_change
    
    def compute_pose_variance(self, landmarks_window: List[np.ndarray]) -> float:
        """
        Compute how much the hand pose varies within the window.
        
        Low pose variance = stable gesture hold
        High pose variance = transition between gestures
        
        Args:
            landmarks_window: List of landmark arrays
            
        Returns:
            Pose variance (0-1 scale)
        """
        if len(landmarks_window) < 2:
            return 0.0
        
        # Stack all frames and compute frame-to-frame distances
        all_frames = np.array(landmarks_window)  # Shape: (n_frames, 42, 3)
        
        # Compute centroid for each frame
        centroids = np.mean(all_frames, axis=1)  # Shape: (n_frames, 3)
        
        # Variance of centroids across window
        variance = np.var(centroids, axis=0).mean()
        
        # Normalize
        normalized_variance = min(variance / 0.02, 1.0)
        
        return normalized_variance
    
    def detect_gesture_boundaries(self, 
                                  all_landmarks: List[np.ndarray]) -> List[Tuple[int, int]]:
        """
        Detect gesture boundaries from hand position transitions.
        
        Key insight: When hand position changes significantly, a new gesture begins.
        This is the pivot point for gesture boundaries.
        
        Algorithm:
        1. For each frame window, compute hand position change
        2. High position change = transition between gestures
        3. Mark boundaries where position change crosses threshold
        
        Args:
            all_landmarks: List of landmark arrays (one per frame)
            
        Returns:
            List of (start_frame, end_frame) tuples for each gesture segment
        """
        if len(all_landmarks) < self.min_gesture_frames:
            return [(0, len(all_landmarks))]
        
        # Compute position change for sliding windows
        position_changes = []
        for i in range(len(all_landmarks)):
            # Get window centered on frame i
            start_idx = max(0, i - self.window_size // 2)
            end_idx = min(len(all_landmarks), i + self.window_size // 2 + 1)
            window = all_landmarks[start_idx:end_idx]
            
            pos_change = self.compute_hand_position_change(window)
            position_changes.append(pos_change)
        
        position_changes = np.array(position_changes)
        
        # Find transitions (high position change) = gesture boundaries
        boundaries = []
        in_gesture = False
        gesture_start = 0
        last_boundary_frame = 0
        
        for frame_idx in range(len(position_changes)):
            pos_change = position_changes[frame_idx]
            
            # Detect high position change (pivot point for gesture change)
            if pos_change > self.position_change_threshold:
                # Transition detected
                if in_gesture and (frame_idx - gesture_start) >= self.min_gesture_frames:
                    # End current gesture
                    gesture_end = frame_idx
                    if gesture_end > gesture_start and (gesture_end - last_boundary_frame) >= 2:
                        boundaries.append((gesture_start, gesture_end))
                        last_boundary_frame = gesture_end
                
                # Start new gesture after transition
                gesture_start = frame_idx
                in_gesture = False
            else:
                # Stable hand position = continue current gesture
                in_gesture = True
        
        # Add final gesture
        if in_gesture and (len(all_landmarks) - gesture_start) >= self.min_gesture_frames:
            boundaries.append((gesture_start, len(all_landmarks)))
        elif not boundaries:
            # No clear boundaries detected, use entire sequence
            boundaries.append((0, len(all_landmarks)))
        
        return boundaries
    
    def infer_gesture_labels(self, 
                            sentence: str,
                            num_segments: int) -> List[str]:
        """
        Infer gesture labels from sentence.
        
        Maps detected segments to words in the sentence.
        Handles cases where segments ≠ words by grouping or splitting gracefully.
        
        Args:
            sentence: Full sentence (e.g., "hi how are you")
            num_segments: Number of detected segments
            
        Returns:
            List of gesture labels (one per segment)
        """
        words = sentence.strip().lower().split()
        
        if len(words) == 0:
            return [f"gesture_{i}" for i in range(num_segments)]
        
        if num_segments == len(words):
            return words
        
        if num_segments < len(words):
            # More words than segments: group words
            labels = []
            words_per_segment = len(words) / num_segments
            for seg_idx in range(num_segments):
                start_word = int(seg_idx * words_per_segment)
                end_word = int((seg_idx + 1) * words_per_segment)
                end_word = min(end_word, len(words))
                
                if start_word < len(words):
                    label = "_".join(words[start_word:end_word])
                    labels.append(label if label else f"gesture_{seg_idx}")
                else:
                    labels.append(f"gesture_{seg_idx}")
            return labels
        else:
            # More segments than words: distribute segments
            labels = []
            segments_per_word = num_segments / len(words)
            seg_idx = 0
            for word_idx, word in enumerate(words):
                word_segments = int((word_idx + 1) * segments_per_word) - seg_idx
                for _ in range(word_segments):
                    if word_segments > 1:
                        labels.append(f"{word}_{_ + 1}")
                    else:
                        labels.append(word)
                seg_idx += word_segments
            return labels[:num_segments]


def extract_sentence_gestures(video_path: str,
                             extract_landmarks_func,
                             segmenter: GestureSegmenter = None,
                             debug: bool = False) -> List[GestureSegment]:
    """
    Extract gesture segments from a video containing a sentence.
    
    Args:
        video_path: Path to video file
        extract_landmarks_func: Function to extract landmarks (returns all frames)
        segmenter: GestureSegmenter instance (creates default if None)
        debug: If True, save debug info
        
    Returns:
        List of GestureSegment objects detected in video
    """
    if segmenter is None:
        segmenter = GestureSegmenter()
    
    # Extract all landmarks from video
    all_landmarks, frame_count, valid_count = extract_landmarks_func(
        video_path, 
        skip_empty_frames=False,  # Keep frames for continuity
        debug=debug
    )
    
    if not all_landmarks or len(all_landmarks) == 0:
        print(f"⊘ No landmarks extracted from {video_path}")
        return []
    
    # Detect gesture boundaries based on hand position changes
    boundaries = segmenter.detect_gesture_boundaries(all_landmarks)
    
    # Extract sentence from filename
    video_filename = os.path.basename(video_path)
    sentence = os.path.splitext(video_filename)[0]
    
    # Infer gesture labels from sentence
    gesture_labels = segmenter.infer_gesture_labels(sentence, len(boundaries))
    
    # Create gesture segments
    segments = []
    for seg_idx, ((start, end), label) in enumerate(zip(boundaries, gesture_labels)):
        segment_landmarks = all_landmarks[start:end]
        
        # Compute confidence based on hand position stability within segment
        if len(segment_landmarks) > 1:
            # Confidence = inverse of position variance (stable = high confidence)
            variance = segmenter.compute_pose_variance(segment_landmarks)
            # Higher variance (lots of movement) = lower confidence
            confidence = 1.0 - min(variance, 1.0)
        else:
            confidence = 0.5
        
        segment = GestureSegment(
            start_frame=start,
            end_frame=end,
            gesture_label=label,
            confidence=confidence,
            landmarks=segment_landmarks
        )
        segments.append(segment)
    
    return segments


def save_sentence_gestures(segments: List[GestureSegment],
                          output_dir: str) -> Dict[str, List[str]]:
    """
    Save gesture segments as sequences.
    
    Creates one CSV per gesture within the sentence, grouped by gesture label.
    
    Args:
        segments: List of detected gesture segments
        output_dir: Output directory
        
    Returns:
        Dict mapping gesture_label → list of saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    gesture_files = {}
    gesture_counts = {}
    
    for seg_idx, segment in enumerate(segments):
        label = segment.gesture_label
        
        # Track count per gesture (multiple instances in one sentence)
        if label not in gesture_counts:
            gesture_counts[label] = 0
        gesture_counts[label] += 1
        
        # Create gesture folder
        gesture_dir = os.path.join(output_dir, label)
        os.makedirs(gesture_dir, exist_ok=True)
        
        # Save as sequence
        sequence_id = gesture_counts[label] - 1
        seq_file = os.path.join(gesture_dir, f"sequence_{sequence_id}.csv")
        
        with open(seq_file, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(segment.landmarks)
        
        if label not in gesture_files:
            gesture_files[label] = []
        gesture_files[label].append(seq_file)
    
    return gesture_files


def process_sentence_video(video_path: str,
                          extract_landmarks_func,
                          output_root: str = "data/dataset",
                          segmenter: GestureSegmenter = None,
                          debug: bool = False) -> Dict:
    """
    Complete pipeline: video → segments → saved sequences.
    
    Args:
        video_path: Path to video
        extract_landmarks_func: Landmark extraction function
        output_root: Output directory root
        segmenter: GestureSegmenter (creates default if None)
        debug: Debug mode
        
    Returns:
        Summary dict with processing results
    """
    video_filename = os.path.basename(video_path)
    sentence = os.path.splitext(video_filename)[0]
    
    print(f"\n🎬 {video_filename}")
    print(f"   Sentence: '{sentence}'")
    
    try:
        # Extract and segment
        segments = extract_sentence_gestures(
            video_path, 
            extract_landmarks_func, 
            segmenter, 
            debug
        )
        
        if not segments:
            print(f"   ✗ No gestures detected")
            return {'success': False, 'reason': 'No gestures detected'}
        
        # Save segments
        gesture_files = save_sentence_gestures(segments, output_root)
        
        # Print summary
        print(f"   ✓ Detected {len(segments)} gestures:")
        for seg_idx, seg in enumerate(segments):
            frames = seg.end_frame - seg.start_frame
            print(f"     [{seg_idx+1}] {seg.gesture_label:15} "
                  f"(frames {seg.start_frame}-{seg.end_frame}, "
                  f"confidence: {seg.confidence:.2f})")
        
        print(f"   ✓ Saved {sum(len(files) for files in gesture_files.values())} sequences")
        
        return {
            'success': True,
            'sentence': sentence,
            'segments': len(segments),
            'gestures': {seg.gesture_label: len(gesture_files.get(seg.gesture_label, [])) 
                        for seg in segments},
            'total_sequences': sum(len(files) for files in gesture_files.values())
        }
    
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return {'success': False, 'reason': str(e)}
