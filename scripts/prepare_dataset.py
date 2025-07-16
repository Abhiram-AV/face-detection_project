import os
import random
import json
import sys

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import PROCESSED_IMAGES_DIR from config
from src.config import DATA_DIR, TRAIN_TEST_SPLIT_RATIO, DATA_SPLIT_FILE, PROCESSED_IMAGES_DIR
from src.utils import ensure_dir

def main():
    print("Preparing dataset split...")
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found at '{DATA_DIR}'")
        print("Please create it and place your dataset inside.")
        return

    # Get all subdirectories, then filter out the PROCESSED_IMAGES_DIR
    # We use os.path.basename to get just the folder name from PROCESSED_IMAGES_DIR path
    processed_dir_name = os.path.basename(PROCESSED_IMAGES_DIR)

    identities = [
        name for name in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, name)) and name != processed_dir_name
    ]
    
    if not identities:
        print(f"Error: No identity sub-folders found in '{DATA_DIR}' (after excluding '{processed_dir_name}').")
        return

    random.shuffle(identities)
    split_index = int(len(identities) * TRAIN_TEST_SPLIT_RATIO)
    
    train_identities = identities[:split_index]
    eval_identities = identities[split_index:]

    dataset_split = {
        'train': train_identities,
        'eval': eval_identities
    }

    # Ensure the directory for the split file exists
    ensure_dir(os.path.dirname(DATA_SPLIT_FILE))

    with open(DATA_SPLIT_FILE, 'w') as f:
        json.dump(dataset_split, f, indent=4)

    print(f"Dataset split complete.")
    print(f"Total identities: {len(identities)}")
    print(f"Training identities ({len(train_identities)}): {train_identities[:5]}...")
    print(f"Evaluation identities ({len(eval_identities)}): {eval_identities[:5]}...")
    print(f"Split information saved to '{DATA_SPLIT_FILE}'")

if __name__ == "__main__":
    main()