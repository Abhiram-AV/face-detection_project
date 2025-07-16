import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def calculate_accuracy(ground_truth, scores, threshold):
    """Calculates accuracy given a similarity threshold."""
    predictions = scores >= threshold
    correct = np.sum(predictions == ground_truth)
    return correct / len(ground_truth)

def run_evaluation(embeddings, labels):
    """
    Efficiently computes pairwise similarity and ground truth labels.

    Returns:
        scores (np.array): Array of cosine similarity scores for each pair.
        ground_truth (np.array): Array of 1 if same person, else 0.
    """
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    n = len(embeddings)

    if n < 2:
        return np.array([]), np.array([])

    print("Calculating pairwise similarity...")

    # Compute full cosine similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # Get upper triangle indices (excluding diagonal)
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # Extract similarity scores for unique pairs
    scores = sim_matrix[i_idx, j_idx]

    # Determine if each pair has the same label
    ground_truth = (labels[i_idx] == labels[j_idx]).astype(int)

    return scores, ground_truth
