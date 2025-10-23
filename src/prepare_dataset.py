from data_collection import process_image_dataset
import os

def main():
    # Convert images in handsigns/ to landmark CSVs in dataset/
    src_dir = os.path.join("data", "handsigns")
    dst_dir = os.path.join("data", "dataset")
    
    print(f"Processing images from {src_dir} into landmark CSVs in {dst_dir}...")
    process_image_dataset(src_dir, dst_dir)
    print("Done processing images.")
    
    # Count processed files
    total_csvs = 0
    for label in os.listdir(dst_dir):
        label_path = os.path.join(dst_dir, label)
        if os.path.isdir(label_path):
            csvs = [f for f in os.listdir(label_path) if f.endswith('.csv')]
            print(f"Label '{label}': {len(csvs)} sequences")
            total_csvs += len(csvs)
    print(f"\nTotal sequences available for training: {total_csvs}")

if __name__ == '__main__':
    main()