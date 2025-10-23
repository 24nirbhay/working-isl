import argparse
import os
from src.data_collection import collect_data
from src.model_train import train_model
from src.real_time_inference import run_inference

def main():
    parser = argparse.ArgumentParser(description="ISL to English Translator")
    parser.add_argument("command", choices=["collect", "train", "run"], help="Command to execute")
    parser.add_argument("--gesture", help="Gesture name for data collection")
    args = parser.parse_args()

    # Create necessary directories if they don't exist
    os.makedirs("data/dataset", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if args.command == "collect":
        if not args.gesture:
            print("Please provide a gesture name using --gesture")
            return
        collect_data(args.gesture)
    elif args.command == "train":
        train_model()
    elif args.command == "run":
        run_inference()

if __name__ == "__main__":
    main()
