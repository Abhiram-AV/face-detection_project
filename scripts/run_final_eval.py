import os
import json
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import DATA_DIR, DATA_SPLIT_FILE, OUTPUT_DIR, DEVICE
from src.model import get_model_for_finetuning
from src.data_loader import get_dataloader
from src.evaluator import run_evaluation
from src.utils import plot_metrics, plot_comparison_roc
from tqdm import tqdm

def get_finetuned_embeddings(model, dataloader):
    """Extracts embeddings using the fine-tuned model."""
    model.eval()
    all_embeddings = []
    all_labels = []

    feature_extractor = model[0]  # Extract the backbone part

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting Fine-Tuned Embeddings"):
            images = images.to(DEVICE)
            embeddings = feature_extractor(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_embeddings), np.concatenate(all_labels)

def main():
    print("--- Running Final Evaluation on Fine-Tuned Model ---")

    checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoints", "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Fine-tuned model not found at {checkpoint_path}")
        print("Please run the fine-tuning script first.")
        return

    with open(DATA_SPLIT_FILE, 'r') as f:
        train_identities = json.load(f)['train']
    num_classes = len(train_identities)

    model = get_model_for_finetuning(num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)

    eval_loader, _, _ = get_dataloader(is_train=False)

    embeddings, labels = get_finetuned_embeddings(model, eval_loader)

    scores, ground_truth = run_evaluation(embeddings, labels)

    if len(scores) == 0:
        print("Evaluation could not be completed.")
        return

    output_path = os.path.join(OUTPUT_DIR, "fine_tuned")
    os.makedirs(output_path, exist_ok=True)
    plot_metrics(ground_truth, scores, output_path, prefix="Fine-Tuned")

    #  Save finetuned results
    result_file = os.path.join(output_path, "finetuned_results.npz")
    np.savez_compressed(result_file, scores=scores, ground_truth=ground_truth)
    print(f"Saved fine-tuned results to: {result_file}")

    # Comparison ROC Curve
    baseline_results_path = os.path.join(OUTPUT_DIR, "baseline", "baseline_results.npz")
    if os.path.exists(baseline_results_path):
        baseline_data = np.load(baseline_results_path)
        baseline_results = (baseline_data['ground_truth'], baseline_data['scores'])
        finetuned_results = (ground_truth, scores)

        plot_comparison_roc(baseline_results, finetuned_results, output_path)
        print("Generated comparison ROC curve.")

    print(f"Final evaluation complete. Results saved to '{output_path}'")

if __name__ == "__main__":
    main()
