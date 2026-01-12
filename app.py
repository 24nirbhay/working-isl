import argparse
import os
from src.data_collection import collect_data, convert_images_to_sequences, collect_sentence
from src.model_train import train_model
from src.real_time_translator import main as run_translator
from src.dataset_manager import (
    create_training_dataset_from_handsigns,
    create_test_dataset_from_fortesting,
    create_training_dataset_from_sentences,
    print_dataset_summary
)
from src.evaluate_model import evaluate_model, print_evaluation_report, save_evaluation_report, find_latest_model

def main():
    parser = argparse.ArgumentParser(description="ISL to English Translator")
    parser.add_argument("command", choices=[
        "collect", "collect-sentence", "convert", "train", "run", "gpu-check",
        "convert-train", "convert-test", "convert-all", "convert-sentences",
        "dataset-stats", "evaluate"
    ], help="Command to execute")
    parser.add_argument("--gesture", help="Gesture name for data collection")
    parser.add_argument("--sentence", help="Sentence label for sentence-level capture")
    parser.add_argument("--use-sentences", action="store_true", help="Include data/sentences in training dataset if available")
    parser.add_argument("--sentences-only", action="store_true", help="Train using only data/sentences")
    parser.add_argument("--model", help="Path to trained model for evaluation")
    args = parser.parse_args()

    # Create necessary directories if they don't exist
    os.makedirs("data/dataset", exist_ok=True)
    os.makedirs("data/test_dataset", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if args.command == "collect":
        if not args.gesture:
            print("Please provide a gesture name using --gesture")
            return
        collect_data(args.gesture)
    elif args.command == "collect-sentence":
        if not args.sentence:
            print("Please provide a sentence label using --sentence")
            return
        collect_sentence(args.sentence)
    elif args.command == "convert":
        # Legacy: Convert images in data/handsigns into landmark CSVs under data/dataset
        print("Converting images from data/handsigns/ to sequences...")
        convert_images_to_sequences(images_root="data/handsigns", output_root="data/dataset", frames_per_sequence=30)
        print("Dataset conversion complete. Ready for training!")
    elif args.command == "convert-train":
        # Convert handsigns/ to training dataset
        create_training_dataset_from_handsigns(frames_per_sequence=30, debug=False)
        print_dataset_summary()
    elif args.command == "convert-test":
        # Convert fortesting/ to test dataset
        create_test_dataset_from_fortesting(frames_per_sequence=30, debug=False)
        print_dataset_summary()
    elif args.command == "convert-all":
        # Convert both training and test data
        print("Converting all data...")
        create_training_dataset_from_handsigns(frames_per_sequence=30, debug=False)
        create_test_dataset_from_fortesting(frames_per_sequence=30, debug=False)
        print_dataset_summary()
    elif args.command == "convert-sentences":
        # Convert sentence-level videos with hand position transition detection
        print("Converting sentence-level videos with hand position transition detection...")
        create_training_dataset_from_sentences(
            video_root="data/handsigns",
            output_root="data/dataset",
            position_change_threshold=0.25,
            window_size=5,
            min_gesture_frames=8,
            debug=False
        )
        print_dataset_summary()
    elif args.command == "dataset-stats":
        # Show dataset statistics
        print_dataset_summary()
    elif args.command == "train":
        roots = None
        if args.sentences_only:
            roots = ["data/sentences"]
        elif args.use_sentences:
            roots = ["data/dataset", "data/sentences"]
        train_model(roots=roots)
    elif args.command == "evaluate":
        # Evaluate model on test dataset
        if args.model:
            model_path = args.model
        else:
            model_path = find_latest_model()
        
        if model_path is None:
            print("No model found. Please train a model first.")
            return
        
        print(f"Evaluating model: {model_path}")
        results = evaluate_model(model_path)
        if results:
            print_evaluation_report(results)
            save_evaluation_report(results)
        else:
            print("Evaluation failed")
    elif args.command == "run":
        run_translator()
    elif args.command == "gpu-check":
        from src.gpu_check import check_gpu
        check_gpu()

if __name__ == "__main__":
    main()
