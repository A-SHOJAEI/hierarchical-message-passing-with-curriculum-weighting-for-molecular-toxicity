"""Evaluation metrics for molecular toxicity prediction."""

import logging
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics.

    Args:
        y_true: Ground truth labels [n_samples]
        y_pred: Predicted labels [n_samples]
        y_prob: Predicted probabilities [n_samples]

    Returns:
        Dictionary of metric names and values
    """
    metrics = {}

    # Classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

    # ROC-AUC and PR-AUC
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        metrics['pr_auc'] = average_precision_score(y_true, y_prob)
    except ValueError as e:
        logger.warning(f"Could not compute AUC metrics: {e}")
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)

        # Specificity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return metrics


def compute_complexity_stratified_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    complexities: np.ndarray,
    num_bins: int = 3,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics stratified by molecular complexity.

    Args:
        y_true: Ground truth labels [n_samples]
        y_pred: Predicted labels [n_samples]
        y_prob: Predicted probabilities [n_samples]
        complexities: Complexity scores [n_samples]
        num_bins: Number of complexity bins

    Returns:
        Dictionary mapping complexity bins to their metrics
    """
    stratified_metrics = {}

    # Define complexity bins
    percentiles = np.linspace(0, 100, num_bins + 1)
    thresholds = np.percentile(complexities, percentiles)

    bin_names = ['simple', 'medium', 'complex']
    if num_bins > 3:
        bin_names = [f'bin_{i}' for i in range(num_bins)]

    for i in range(num_bins):
        bin_name = bin_names[i] if i < len(bin_names) else f'bin_{i}'

        # Select samples in this complexity range
        mask = (complexities >= thresholds[i]) & (complexities < thresholds[i + 1])
        if i == num_bins - 1:
            mask = complexities >= thresholds[i]

        if mask.sum() == 0:
            continue

        bin_y_true = y_true[mask]
        bin_y_pred = y_pred[mask]
        bin_y_prob = y_prob[mask]

        # Compute metrics for this bin
        try:
            bin_metrics = compute_metrics(bin_y_true, bin_y_pred, bin_y_prob)
            bin_metrics['num_samples'] = int(mask.sum())
            bin_metrics['complexity_range'] = (float(thresholds[i]), float(thresholds[i + 1]))

            stratified_metrics[bin_name] = bin_metrics
        except Exception as e:
            logger.warning(f"Error computing metrics for {bin_name}: {e}")

    return stratified_metrics


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on a dataset.

    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to use

    Returns:
        Tuple of (metrics, y_true, y_pred, y_prob, complexities)
    """
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    all_complexities = []

    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            batch['x'] = batch['x'].to(device)
            batch['edge_index'] = batch['edge_index'].to(device)
            if 'edge_attr' in batch and batch['edge_attr'] is not None:
                batch['edge_attr'] = batch['edge_attr'].to(device)
            batch['batch'] = batch['batch'].to(device)
            batch['y'] = batch['y'].to(device)

            # Forward pass
            outputs = model(batch)
            probs = torch.sigmoid(outputs).squeeze()

            # Collect predictions
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch['y'].cpu().numpy())
            all_complexities.append(batch['complexity'].cpu().numpy())

    # Concatenate all batches
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels).astype(int)
    y_pred = (y_prob > 0.5).astype(int)
    complexities = np.concatenate(all_complexities)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)

    return metrics, y_true, y_pred, y_prob, complexities
