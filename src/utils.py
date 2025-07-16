import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np

def ensure_dir(directory):
    """Ensures that a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_metrics(ground_truth, scores, output_path, prefix=""):
    """
    Plots and saves ROC Curve, Precision-Recall Curve, and Similarity Distributions.
    """
    ensure_dir(output_path)

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(ground_truth, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{prefix} Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f"{prefix.lower().replace(' ', '_')}_roc_curve.png"))
    plt.close()

    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(ground_truth, scores)
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{prefix} Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f"{prefix.lower().replace(' ', '_')}_pr_curve.png"))
    plt.close()

    # 3. Similarity Distributions
    same_scores = scores[ground_truth == 1]
    diff_scores = scores[ground_truth == 0]
    plt.figure(figsize=(10, 6))
    plt.hist(same_scores, bins=50, alpha=0.7, label='Same Identity', color='green', density=True)
    plt.hist(diff_scores, bins=50, alpha=0.7, label='Different Identity', color='red', density=True)
    plt.title(f'{prefix} Similarity Score Distributions')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f"{prefix.lower().replace(' ', '_')}_similarity_dist.png"))
    plt.close()

def plot_comparison_roc(baseline_results, finetuned_results, output_path):
    """Plots a comparison of two ROC curves."""
    plt.figure(figsize=(8, 8))

    # Plot baseline
    b_gt, b_scores = baseline_results
    fpr, tpr, _ = roc_curve(b_gt, b_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Baseline ROC (AUC = {roc_auc:.3f})')

    # Plot fine-tuned
    ft_gt, ft_scores = finetuned_results
    fpr, tpr, _ = roc_curve(ft_gt, ft_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Fine-Tuned ROC (AUC = {roc_auc:.3f})')

    # Formatting
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison: Baseline vs. Fine-Tuned')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_path, "comparison_roc_curve.png"))
    plt.close()