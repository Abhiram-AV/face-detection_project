import os
import cv2
import json
import sys
from tqdm import tqdm

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import DATA_DIR, PROCESSED_IMAGES_DIR, DATA_SPLIT_FILE
from src.utils import ensure_dir


def process_cropped_faces():
    print("--- Starting Copying of Cropped Face Images ---")

    try:
        with open(DATA_SPLIT_FILE, 'r') as f:
            dataset_split = json.load(f)
        all_identities = list(set(dataset_split.get('train', []) + dataset_split.get('eval', [])))
        if not all_identities:
            print("No identities found in dataset_split.json. Processing all folders in data directory.")
            all_identities = [name for name in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, name))]
    except Exception as e:
        print(f"Error loading dataset_split.json: {e}")
        return

    print(f"Copying from: {DATA_DIR}")
    print(f"Saving to: {PROCESSED_IMAGES_DIR}")

    copied_count = 0
    skipped_count = 0

    for identity in tqdm(all_identities, desc="Processing identities"):
        src_identity_dir = os.path.join(DATA_DIR, identity)
        dst_identity_dir = os.path.join(PROCESSED_IMAGES_DIR, identity)
        ensure_dir(dst_identity_dir)

        quality_folders = [d for d in os.listdir(src_identity_dir) if os.path.isdir(os.path.join(src_identity_dir, d))]
        if not quality_folders:
            quality_folders = [""]

        for quality in quality_folders:
            src_quality_dir = os.path.join(src_identity_dir, quality) if quality else src_identity_dir
            dst_quality_dir = os.path.join(dst_identity_dir, quality) if quality else dst_identity_dir
            ensure_dir(dst_quality_dir)

            image_files = [f for f in os.listdir(src_quality_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for file in image_files:
                src_file = os.path.join(src_quality_dir, file)
                dst_file = os.path.join(dst_quality_dir, file)

                img = cv2.imread(src_file)
                if img is None or img.shape[0] == 0 or img.shape[1] == 0:
                    print(f"⚠️ Skipped unreadable or empty: {src_file}")
                    skipped_count += 1
                    continue

                # Optionally resize (if needed to IMG_SIZE) — skipped if already 112x112
                cv2.imwrite(dst_file, img)
                copied_count += 1

    print(f"\n Finished copying {copied_count} images.")
    print(f" Skipped {skipped_count} unreadable or corrupt images.")


if __name__ == "__main__":
    process_cropped_faces()
