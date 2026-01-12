"""
Dataset Management Pipeline
Handles train/test data separation and conversion workflows.
"""

import os
import shutil
import csv
import glob
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from .data_collection import extract_landmarks_from_image_file, extract_landmarks_from_video_file, build_sliding_windows, _extract_two_hand_landmarks_from_results
    from .sentence_gestures import GestureSegmenter, extract_sentence_gestures, process_sentence_video
except ImportError:
    from data_collection import extract_landmarks_from_image_file, extract_landmarks_from_video_file, build_sliding_windows, _extract_two_hand_landmarks_from_results
    from sentence_gestures import GestureSegmenter, extract_sentence_gestures, process_sentence_video

# Paths
HANDSIGNS_PATH = "data/handsigns"
DATASET_PATH = "data/dataset"
FORTESTING_PATH = "fortesting"
TEST_DATASET_PATH = "data/test_dataset"
CONVERSION_LOG_PATH = "data/conversion_log.csv"


def extract_gesture_class(video_filename):
    """Extract gesture class from video filename.
    
    Handles patterns like:
    - hello.mp4 → 'hello'
    - hello(1).mp4 → 'hello'
    -hello how are you.mp4 → 'hello' , 'how are you'
    - hello_1.mp4 → 'hello'
    - hello_v1.mp4 → 'hello'
    - sign_name(123).mp4 → 'sign_name'
    - sign_name_v2.mp4 → 'sign_name'
    
    Returns:
        Base gesture name without version/numbering
    """
    import re
    
    # Remove extension
    name = os.path.splitext(video_filename)[0]
    
    # Remove trailing version patterns: (1), (2), _1, _v1, _v2, etc.
    # First handle (N) pattern
    name = re.sub(r'\(\d+\)$', '', name)            # Remove (123) at end
    # Then handle _N and _vN pattern
    name = re.sub(r'_v\d+$', '', name)              # Remove _v123 at end
    name = re.sub(r'_\d+$', '', name)               # Remove _123 at end
    
    return name.strip()


def create_training_dataset_from_videos(video_root="data/handsigns", output_root="data/dataset", window_size=10, stride=5, debug=False):
    """Video-first temporal learning pipeline with sliding windows.
    
    Process videos and extract multiple overlapping temporal windows per video.
    Multiple videos with same gesture name (e.g., hello(1).mp4, hello(2).mp4)
    are consolidated into a single gesture folder.
    
    Consolidation examples:
        hello(1).mp4, hello(2).mp4, hello_1.mp4 → all sequences saved to output_root/hello/
        gesture_v1.mp4, gesture_v2.mp4 → all sequences saved to output_root/gesture/
    
    Args:
        video_root: Root directory containing videos (flat or nested)
        output_root: Output directory for sequences (one folder per unique gesture)
        window_size: Frames per sequence (10 = ~0.33s at 30fps, typical gesture)
        stride: Step between windows (5 = 50% overlap, generates more training data)
        debug: If True, save debug frames
    
    Returns:
        Total sequences created
    """
    if not os.path.exists(video_root):
        print(f"Error: {video_root} not found")
        return 0
    
    os.makedirs(output_root, exist_ok=True)
    total_sequences = 0
    conversion_records = []
    
    print(f"\n{'='*70}")
    print(f"VIDEO-FIRST TRAINING DATA CONVERSION (Consolidated Gestures)")
    print(f"Input:  {video_root}")
    print(f"Output: {output_root}")
    print(f"Window: {window_size} frames, Stride: {stride} frames")
    print(f"Strategy: hello(1), hello(2) → consolidated into single gesture folder")
    print(f"{'='*70}\n")
    
    # Find all video files recursively
    video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv')
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_root, '**', ext), recursive=True))
    
    if not video_files:
        print(f"⊘ No video files found in {video_root}")
        return 0
    
    print(f"Found {len(video_files)} video file(s)\n")
    
    gesture_sequence_counts = {}  # Track sequences per gesture for consolidation
    
    for video_path in sorted(video_files):
        # Extract base gesture class from video filename (removes version numbers)
        video_filename = os.path.basename(video_path)
        gesture_class = extract_gesture_class(video_filename)
        
        print(f"🎬 {video_filename}")
        print(f"   → Class: '{gesture_class}'")
        
        try:
            # Extract all landmarks from video
            all_landmarks, frame_count, valid_count = extract_landmarks_from_video_file(
                video_path, skip_empty_frames=True, debug=debug
            )
            
            if not all_landmarks:
                print(f"   ✗ No valid landmarks extracted")
                continue
            
            # Build overlapping windows
            windows = build_sliding_windows(all_landmarks, window_size=window_size, stride=stride)
            
            if not windows:
                print(f"   ⚠ Could not create windows from {len(all_landmarks)} frames")
                continue
            
            # Create class directory (consolidated by gesture name)
            gesture_dir = os.path.join(output_root, gesture_class)
            os.makedirs(gesture_dir, exist_ok=True)
            
            # Get next sequence ID for this gesture (to avoid overwrites)
            if gesture_class not in gesture_sequence_counts:
                # Count existing sequences for this gesture
                existing_sequences = glob.glob(os.path.join(gesture_dir, "sequence_*.csv"))
                gesture_sequence_counts[gesture_class] = len(existing_sequences)
            
            # Save each window as a separate sequence
            start_seq_id = gesture_sequence_counts[gesture_class]
            for window_idx, window in enumerate(windows):
                seq_id = start_seq_id + window_idx
                seq_file = os.path.join(gesture_dir, f"sequence_{seq_id}.csv")
                with open(seq_file, 'w', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerows(window)
                
                total_sequences += 1
                conversion_records.append({
                    'gesture': gesture_class,
                    'source_video': video_filename,
                    'window_id': window_idx,
                    'sequence_id': seq_id,
                    'frames': len(window),
                    'total_video_frames': frame_count
                })
            
            gesture_sequence_counts[gesture_class] += len(windows)
            print(f"   ✓ Created {len(windows)} sequences")
            print(f"     Frames extracted: {len(all_landmarks)} ({valid_count}/{frame_count} detected)")
            print()
        
        except Exception as e:
            print(f"   ✗ Error: {e}")
            print()
            continue
    
    # Log conversion
    if conversion_records:
        with open(CONVERSION_LOG_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['gesture', 'source_video', 'window_id', 'sequence_id', 'frames', 'total_video_frames'])
            writer.writeheader()
            writer.writerows(conversion_records)
    
    print(f"{'='*70}")
    print(f"✓ Conversion complete!")
    print(f"  Total sequences: {total_sequences}")
    print(f"  Gestures (consolidated): {len(gesture_sequence_counts)}")
    for gesture, count in sorted(gesture_sequence_counts.items()):
        print(f"    - {gesture}: {count} sequences")
    print(f"  Output: {output_root}")
    print(f"  Log: {CONVERSION_LOG_PATH}")
    print(f"{'='*70}\n")
    print(f"✓ Conversion complete!")
    print(f"  Total sequences created: {total_sequences}")
    print(f"  Output: {output_root}")
    print(f"  Log: {CONVERSION_LOG_PATH}")
    print(f"{'='*70}\n")
    
    return total_sequences


def create_training_dataset_from_sentences(video_root="data/handsigns", 
                                          output_root="data/dataset",
                                          position_change_threshold=0.25,
                                          window_size=5,
                                          min_gesture_frames=8,
                                          debug=False):
    """Sentence-level gesture detection pipeline.
    
    Process videos as sentences containing multiple consecutive gestures.
    Detects gesture boundaries from HAND POSITION CHANGES (pivot points).
    
    Key: When hand position changes significantly, a new gesture begins.
    
    Example:
        video: "hello how are you.mp4"
        ↓
        Extract frames and hand landmarks
        ↓
        Detect hand position transitions
        ↓
        Output:
          data/dataset/
            ├── hello/
            │   └── sequence_0.csv
            ├── how/
            │   └── sequence_0.csv
            ├── are/
            │   └── sequence_0.csv
            └── you/
                └── sequence_0.csv
    
    Args:
        video_root: Input videos (each filename = sentence)
        output_root: Output directory (gestures extracted by label)
        position_change_threshold: Hand position change to detect boundary (0-1, default 0.25)
        window_size: Frames to compute position change over (default 5)
        min_gesture_frames: Minimum frames per gesture (default 8)
        debug: Save debug info
        
    Returns:
        Total sequences created
    """
    if not os.path.exists(video_root):
        print(f"Error: {video_root} not found")
        return 0
    
    os.makedirs(output_root, exist_ok=True)
    total_sequences = 0
    segmenter = GestureSegmenter(
        position_change_threshold=position_change_threshold,
        window_size=window_size,
        min_gesture_frames=min_gesture_frames
    )
    
    print(f"\n{'='*70}")
    print(f"SENTENCE-LEVEL GESTURE CONVERSION")
    print(f"Mode: Hand position transition detection (pivot point = gesture boundary)")
    print(f"Position change threshold: {position_change_threshold}")
    print(f"Window size: {window_size} frames, Min gesture: {min_gesture_frames} frames")
    print(f"Input:  {video_root}")
    print(f"Output: {output_root}")
    print(f"{'='*70}\n")
    
    # Find all video files recursively
    video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv')
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_root, '**', ext), recursive=True))
    
    if not video_files:
        print(f"⊘ No video files found in {video_root}")
        return 0
    
    print(f"Found {len(video_files)} video file(s)\n")
    
    for video_path in sorted(video_files):
        try:
            result = process_sentence_video(
                video_path,
                extract_landmarks_from_video_file,
                output_root,
                segmenter,
                debug
            )
            
            if result['success']:
                total_sequences += result.get('total_sequences', 0)
            print()
        
        except Exception as e:
            print(f"   ✗ Error: {e}\n")
            continue
    
    print(f"{'='*70}")
    print(f"✓ Sentence-level conversion complete!")
    print(f"  Total sequences extracted: {total_sequences}")
    print(f"  Output: {output_root}")
    print(f"{'='*70}\n")
    
    return total_sequences


def create_training_dataset_from_handsigns(frames_per_sequence=10, stride=5, debug=False):
    """Backward compatible wrapper. Calls video-first pipeline with sliding windows.
    
    Args:
        frames_per_sequence: Window size (default 10 frames ≈ 0.33s at 30fps)
        stride: Step between windows (default 5 = 50% overlap)
        debug: Save debug frames
    """
    return create_training_dataset_from_videos(
        video_root=HANDSIGNS_PATH,
        output_root=DATASET_PATH,
        window_size=frames_per_sequence,
        stride=stride,
        debug=debug
    )


def create_test_dataset_from_videos(video_root="fortesting", output_root="data/test_dataset", window_size=10, stride=5, debug=False):
    """Video-first temporal learning for test data with sliding windows.
    
    Consolidates multiple video versions into single gesture folders.
    
    Args:
        video_root: Root directory containing test videos
        output_root: Output directory for test sequences
        window_size: Frames per sequence window
        stride: Step size between windows
        debug: If True, save debug frames
    
    Returns:
        Total sequences created
    """
    if not os.path.exists(video_root):
        print(f"Error: {video_root} not found")
        return 0
    
    os.makedirs(output_root, exist_ok=True)
    total_sequences = 0
    test_records = []
    gesture_sequence_counts = {}
    
    print(f"\n{'='*70}")
    print(f"VIDEO-FIRST TEST DATA CONVERSION (Consolidated Gestures)")
    print(f"Input:  {video_root}")
    print(f"Output: {output_root}")
    print(f"Window: {window_size} frames, Stride: {stride} frames")
    print(f"{'='*70}\n")
    
    # Find all video files recursively
    video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv')
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_root, '**', ext), recursive=True))
    
    if not video_files:
        print(f"⊘ No video files found in {video_root}")
        return 0
    
    print(f"Found {len(video_files)} video file(s)\n")
    
    for video_path in sorted(video_files):
        # Extract base gesture class from video filename (removes version numbers)
        video_filename = os.path.basename(video_path)
        gesture_class = extract_gesture_class(video_filename)
        
        print(f"🎬 {video_filename}")
        print(f"   → Class: '{gesture_class}'")
        
        try:
            # Extract all landmarks from video
            all_landmarks, frame_count, valid_count = extract_landmarks_from_video_file(
                video_path, skip_empty_frames=True, debug=debug
            )
            
            if not all_landmarks:
                print(f"   ✗ No valid landmarks extracted")
                continue
            
            # Build overlapping windows
            windows = build_sliding_windows(all_landmarks, window_size=window_size, stride=stride)
            
            if not windows:
                print(f"   ⚠ Could not create windows from {len(all_landmarks)} frames")
                continue
            
            # Create class directory (consolidated by gesture name)
            gesture_dir = os.path.join(output_root, gesture_class)
            os.makedirs(gesture_dir, exist_ok=True)
            
            # Get next sequence ID for this gesture
            if gesture_class not in gesture_sequence_counts:
                existing_sequences = glob.glob(os.path.join(gesture_dir, "sequence_*.csv"))
                gesture_sequence_counts[gesture_class] = len(existing_sequences)
            
            # Save each window as a separate sequence
            start_seq_id = gesture_sequence_counts[gesture_class]
            for window_idx, window in enumerate(windows):
                seq_id = start_seq_id + window_idx
                seq_file = os.path.join(gesture_dir, f"sequence_{seq_id}.csv")
                with open(seq_file, 'w', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerows(window)
                
                total_sequences += 1
                test_records.append({
                    'gesture': gesture_class,
                    'source_video': video_filename,
                    'window_id': window_idx,
                    'sequence_id': seq_id,
                    'frames': len(window),
                    'total_video_frames': frame_count
                })
            
            gesture_sequence_counts[gesture_class] += len(windows)
            print(f"   ✓ Created {len(windows)} sequences")
            print(f"     Frames extracted: {len(all_landmarks)} ({valid_count}/{frame_count} detected)")
            print()
        
        except Exception as e:
            print(f"   ✗ Error: {e}")
            print()
            continue
    
    # Log conversion
    test_log_path = "data/test_conversion_log.csv"
    if test_records:
        with open(test_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['gesture', 'source_video', 'window_id', 'sequence_id', 'frames', 'total_video_frames'])
            writer.writeheader()
            writer.writerows(test_records)
    
    print(f"{'='*70}")
    print(f"✓ Test conversion complete!")
    print(f"  Total sequences: {total_sequences}")
    print(f"  Gestures (consolidated): {len(gesture_sequence_counts)}")
    for gesture, count in sorted(gesture_sequence_counts.items()):
        print(f"    - {gesture}: {count} sequences")
    print(f"  Output: {output_root}")
    print(f"  Log: {test_log_path}")
    print(f"{'='*70}\n")
    
    return total_sequences


def create_test_dataset_from_fortesting(frames_per_sequence=10, stride=5, debug=False):
    """Backward compatible wrapper. Calls video-first pipeline with sliding windows.
    
    Args:
        frames_per_sequence: Window size (default 10 frames ≈ 0.33s at 30fps)
        stride: Step between windows (default 5 = 50% overlap)
        debug: Save debug frames
    """
    return create_test_dataset_from_videos(
        video_root=FORTESTING_PATH,
        output_root=TEST_DATASET_PATH,
        window_size=frames_per_sequence,
        stride=stride,
        debug=debug
    )


def get_dataset_stats() -> Dict:
    """Get statistics about current training dataset."""
    stats = {
        'training': {'gestures': {}, 'total_sequences': 0},
        'testing': {'gestures': {}, 'total_sequences': 0}
    }
    
    # Training stats
    if os.path.exists(DATASET_PATH):
        for gesture in os.listdir(DATASET_PATH):
            gesture_dir = os.path.join(DATASET_PATH, gesture)
            if os.path.isdir(gesture_dir):
                sequences = len(glob.glob(os.path.join(gesture_dir, "sequence_*.csv")))
                if sequences > 0:
                    stats['training']['gestures'][gesture] = sequences
                    stats['training']['total_sequences'] += sequences
    
    # Testing stats
    if os.path.exists(TEST_DATASET_PATH):
        for gesture in os.listdir(TEST_DATASET_PATH):
            gesture_dir = os.path.join(TEST_DATASET_PATH, gesture)
            if os.path.isdir(gesture_dir):
                sequences = len(glob.glob(os.path.join(gesture_dir, "sequence_*.csv")))
                if sequences > 0:
                    stats['testing']['gestures'][gesture] = sequences
                    stats['testing']['total_sequences'] += sequences
    
    return stats


def print_dataset_summary():
    """Print a summary of training and testing datasets."""
    stats = get_dataset_stats()
    
    print(f"\n{'='*70}")
    print(f"DATASET SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"TRAINING DATA (from {HANDSIGNS_PATH}):")
    print(f"  Gestures: {len(stats['training']['gestures'])}")
    for gesture, count in sorted(stats['training']['gestures'].items()):
        print(f"    - {gesture}: {count} sequences")
    print(f"  Total sequences: {stats['training']['total_sequences']}\n")
    
    print(f"TEST DATA (from {FORTESTING_PATH}):")
    print(f"  Gestures: {len(stats['testing']['gestures'])}")
    for gesture, count in sorted(stats['testing']['gestures'].items()):
        print(f"    - {gesture}: {count} sequences")
    print(f"  Total sequences: {stats['testing']['total_sequences']}\n")
    
    if stats['training']['total_sequences'] == 0:
        print("⚠ No training data found! Run conversion first.")
    if stats['testing']['total_sequences'] == 0:
        print("⚠ No test data found! Run test conversion first.")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "convert-train":
            create_training_dataset_from_handsigns(debug=False)
            print_dataset_summary()
        elif command == "convert-test":
            create_test_dataset_from_fortesting(debug=False)
            print_dataset_summary()
        elif command == "convert-all":
            create_training_dataset_from_handsigns(debug=False)
            create_test_dataset_from_fortesting(debug=False)
            print_dataset_summary()
        elif command == "stats":
            print_dataset_summary()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python dataset_manager.py [convert-train|convert-test|convert-all|stats]")
    else:
        print("Dataset Manager - Manage train/test data pipeline")
        print("\nUsage: python dataset_manager.py [command]")
        print("\nCommands:")
        print("  convert-train  - Convert handsigns/ to training sequences")
        print("  convert-test   - Convert fortesting/ to test sequences")
        print("  convert-all    - Convert both training and test data")
        print("  stats          - Show dataset statistics")
