#!/usr/bin/env python
"""Evaluation script for hierarchical molecular GNN."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import torch

from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.utils.config import (
    load_config,
    get_device,
    setup_logging,
)
from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.data.loader import (
    get_data_loaders,
)
from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.models.model import (
    HierarchicalMolecularGNN,
)
from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.evaluation.metrics import (
    evaluate_model,
    compute_complexity_stratified_metrics,
)
from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.evaluation.analysis import (
    plot_confusion_matrix,
    plot_complexity_analysis,
    plot_roc_curve,
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Hierarchical Molecular GNN')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (optional, loaded from checkpoint if not provided)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Which split to evaluate'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    return parser.parse_args()


def evaluate(
    checkpoint_path: str,
    config_path: str = None,
    output_dir: str = 'results',
    split: str = 'test',
    log_level: str = 'INFO',
):
    """Evaluate the model.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to configuration file
        output_dir: Output directory for results
        split: Which split to evaluate
        log_level: Logging level
    """
    # Setup logging
    setup_logging(log_level)
    logger.info("=" * 80)
    logger.info("Model Evaluation")
    logger.info("=" * 80)

    try:
        # Load checkpoint
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load configuration
        if config_path:
            config = load_config(config_path)
        else:
            config = checkpoint.get('config')
            if config is None:
                raise ValueError("Config not found in checkpoint and no config path provided")

        logger.info("Configuration loaded")

        # Get device
        device = get_device(config.get('device', 'cuda'))

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        logger.info("Loading data...")
        train_loader, val_loader, test_loader = get_data_loaders(config)

        # Select split
        if split == 'train':
            data_loader = train_loader
        elif split == 'val':
            data_loader = val_loader
        else:
            data_loader = test_loader

        logger.info(f"Evaluating on {split} split with {len(data_loader)} batches")

        # Create and load model
        logger.info("Loading model...")
        model = HierarchicalMolecularGNN(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        # Evaluate
        logger.info("Evaluating model...")
        metrics, y_true, y_pred, y_prob, complexities = evaluate_model(model, data_loader, device)

        # Print overall metrics
        logger.info("\n" + "=" * 80)
        logger.info("Overall Metrics")
        logger.info("=" * 80)
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"{metric_name:20s}: {value:.4f}")
            else:
                logger.info(f"{metric_name:20s}: {value}")

        # Compute complexity-stratified metrics
        logger.info("\n" + "=" * 80)
        logger.info("Complexity-Stratified Metrics")
        logger.info("=" * 80)
        stratified_metrics = compute_complexity_stratified_metrics(
            y_true, y_pred, y_prob, complexities, num_bins=3
        )

        for bin_name, bin_metrics in stratified_metrics.items():
            logger.info(f"\n{bin_name.upper()}:")
            logger.info(f"  Samples: {bin_metrics['num_samples']}")
            logger.info(f"  Complexity Range: {bin_metrics['complexity_range']}")
            logger.info(f"  ROC-AUC: {bin_metrics['roc_auc']:.4f}")
            logger.info(f"  Accuracy: {bin_metrics['accuracy']:.4f}")
            logger.info(f"  F1-Score: {bin_metrics['f1']:.4f}")

        # Save metrics to JSON
        results = {
            'overall_metrics': metrics,
            'stratified_metrics': stratified_metrics,
            'checkpoint': str(checkpoint_path),
            'split': split,
        }

        with open(output_dir / f'evaluation_{split}.json', 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nSaved metrics to {output_dir / f'evaluation_{split}.json'}")

        # Save predictions to CSV
        predictions_df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'complexity': complexities,
            'correct': (y_true == y_pred).astype(int),
        })
        predictions_df.to_csv(output_dir / f'predictions_{split}.csv', index=False)
        logger.info(f"Saved predictions to {output_dir / f'predictions_{split}.csv'}")

        # Generate visualizations
        logger.info("\nGenerating visualizations...")

        plot_confusion_matrix(
            y_true, y_pred,
            save_path=output_dir / f'confusion_matrix_{split}.png'
        )

        plot_complexity_analysis(
            y_true, y_pred, y_prob, complexities,
            save_path=output_dir / f'complexity_analysis_{split}.png'
        )

        plot_roc_curve(
            y_true, y_prob,
            save_path=output_dir / f'roc_curve_{split}.png'
        )

        logger.info(f"Visualizations saved to {output_dir}")

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("Evaluation Summary")
        logger.info("=" * 80)
        logger.info(f"Split: {split}")
        logger.info(f"Samples: {len(y_true)}")
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1-Score: {metrics['f1']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")

        logger.info("\nEvaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        args.checkpoint,
        args.config,
        args.output,
        args.split,
        args.log_level,
    )
