"""Results analysis and visualization."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: List[Dict[str, float]],
    val_metrics: List[Dict[str, float]],
    save_path: Optional[str] = None,
) -> None:
    """Plot training and validation curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_metrics: List of training metrics per epoch
        val_metrics: List of validation metrics per epoch
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curves
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
    axes[0].plot(epochs, val_losses, label='Val Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ROC-AUC curves
    train_aucs = [m.get('roc_auc', 0) for m in train_metrics]
    val_aucs = [m.get('roc_auc', 0) for m in val_metrics]
    axes[1].plot(epochs, train_aucs, label='Train ROC-AUC', marker='o', markersize=3)
    axes[1].plot(epochs, val_aucs, label='Val ROC-AUC', marker='s', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('ROC-AUC')
    axes[1].set_title('Training and Validation ROC-AUC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")

    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Plot confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Non-Toxic', 'Toxic'],
        yticklabels=['Non-Toxic', 'Toxic'],
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")

    plt.close()


def plot_complexity_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    complexities: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Plot analysis of performance vs molecular complexity.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        complexities: Complexity scores
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Complexity distribution
    axes[0, 0].hist(complexities, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Molecular Complexity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Molecular Complexity')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Accuracy vs complexity
    num_bins = 10
    bin_edges = np.linspace(complexities.min(), complexities.max(), num_bins + 1)
    bin_accuracies = []
    bin_centers = []

    for i in range(num_bins):
        mask = (complexities >= bin_edges[i]) & (complexities < bin_edges[i + 1])
        if i == num_bins - 1:
            mask = complexities >= bin_edges[i]

        if mask.sum() > 0:
            bin_acc = (y_true[mask] == y_pred[mask]).mean()
            bin_accuracies.append(bin_acc)
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)

    axes[0, 1].plot(bin_centers, bin_accuracies, marker='o', linestyle='-', linewidth=2)
    axes[0, 1].set_xlabel('Molecular Complexity')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy vs Molecular Complexity')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Error analysis by complexity
    errors = (y_true != y_pred).astype(int)
    axes[1, 0].scatter(complexities, errors, alpha=0.3, s=10)
    axes[1, 0].set_xlabel('Molecular Complexity')
    axes[1, 0].set_ylabel('Error (0=Correct, 1=Wrong)')
    axes[1, 0].set_title('Error Distribution by Complexity')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Confidence vs complexity
    confidence = np.abs(y_prob - 0.5) * 2  # Scale to [0, 1]
    axes[1, 1].scatter(complexities, confidence, alpha=0.3, s=10, c=errors, cmap='RdYlGn_r')
    axes[1, 1].set_xlabel('Molecular Complexity')
    axes[1, 1].set_ylabel('Prediction Confidence')
    axes[1, 1].set_title('Prediction Confidence vs Complexity')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved complexity analysis to {save_path}")

    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Plot ROC curve.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        save_path: Path to save the plot
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curve to {save_path}")

    plt.close()
