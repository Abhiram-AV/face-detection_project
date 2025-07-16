import os
import json
import sys
import numpy as np

# Add project root to sys.path for importing src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import PROCESSED_IMAGES_DIR, DATA_SPLIT_FILE, OUTPUT_DIR
from src.model import FaceRecognitionModel
from src.data_loader import get_identity_filepaths
from src.evaluator import run_evaluation, calculate_accuracy
from src.utils import plot_metrics, ensure_dir


def main():
    print("--- Running Baseline Evaluation ---")

    # 1. Load Model
    model = FaceRecognitionModel()

    # 2. Load Evaluation Data
    with open(DATA_SPLIT_FILE, 'r') as f:
        eval_identities = json.load(f).get('eval', [])

    if not eval_identities:
        print(" No identities found in evaluation split.")
        return

    filepaths, labels, _ = get_identity_filepaths(PROCESSED_IMAGES_DIR, eval_identities)

    if not filepaths:
        print("No image files found for evaluation.")
        return

    # 3. Get Embeddings
    embeddings, valid_indices = model.get_embeddings(filepaths)

    if len(valid_indices) == 0:
        print("No valid faces detected during embedding extraction.")
        return

    valid_labels = [labels[i] for i in valid_indices]

    # 4. Run Evaluation
    scores, ground_truth = run_evaluation(embeddings, valid_labels)

    if len(scores) == 0:
        print(" Evaluation could not be completed. Not enough valid pairs.")
        return

    # 5. Save Results and Plots
    output_path = os.path.join(OUTPUT_DIR, "baseline")
    ensure_dir(output_path)

    plot_metrics(ground_truth, scores, output_path, prefix="Baseline")

    np.savez(os.path.join(output_path, "baseline_results.npz"), scores=scores, ground_truth=ground_truth)

    print(f"Baseline evaluation complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
