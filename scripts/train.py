#!/usr/bin/env python
"""Training script for hierarchical molecular GNN."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.utils.config import (
    load_config,
    save_config,
    set_seed,
    get_device,
    setup_logging,
)
from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.data.loader import (
    get_data_loaders,
)
from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.models.model import (
    HierarchicalMolecularGNN,
)
from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.training.trainer import (
    Trainer,
)
from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.evaluation.analysis import (
    plot_training_curves,
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Hierarchical Molecular GNN')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    return parser.parse_args()


def train_model(config_path: str, checkpoint_path: str = None, log_level: str = 'INFO'):
    """Train the model.

    Args:
        config_path: Path to configuration file
        checkpoint_path: Path to checkpoint to resume from
        log_level: Logging level
    """
    # Setup logging
    setup_logging(log_level)
    logger.info("=" * 80)
    logger.info("Hierarchical Message Passing with Curriculum Weighting")
    logger.info("=" * 80)

    try:
        # Load configuration
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")

        # Set random seed
        seed = config.get('seed', 42)
        set_seed(seed)

        # Get device
        device_name = config.get('device', 'cuda')
        device = get_device(device_name)

        # Create results directory
        results_dir = Path(config.get('logging', {}).get('results_dir', 'results'))
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save config to results
        save_config(config, results_dir / 'config.yaml')

        # Load data
        logger.info("Loading data...")
        train_loader, val_loader, test_loader = get_data_loaders(
            config,
            num_workers=config.get('data', {}).get('num_workers', 4),
        )
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

        # Create model
        logger.info("Creating model...")
        model = HierarchicalMolecularGNN(config)

        # Create trainer
        trainer = Trainer(model, config, device)

        # Load checkpoint if provided
        if checkpoint_path:
            trainer.load_checkpoint(Path(checkpoint_path))

        # MLflow tracking (optional)
        mlflow_enabled = config.get('logging', {}).get('mlflow_tracking', False)
        if mlflow_enabled:
            try:
                import mlflow

                experiment_name = config.get('logging', {}).get('mlflow_experiment', 'hierarchical_mp_toxicity')
                mlflow.set_experiment(experiment_name)

                with mlflow.start_run():
                    # Log parameters
                    mlflow.log_params({
                        'hidden_dim': config.get('model', {}).get('hidden_dim'),
                        'num_layers': config.get('model', {}).get('num_layers'),
                        'batch_size': config.get('training', {}).get('batch_size'),
                        'learning_rate': config.get('training', {}).get('learning_rate'),
                        'curriculum_enabled': config.get('curriculum', {}).get('enable'),
                        'hierarchical_enabled': config.get('hierarchical', {}).get('enable_group_level'),
                    })

                    # Train
                    trainer.fit(train_loader, val_loader)

                    # Log metrics
                    mlflow.log_metrics({
                        'best_val_auc': trainer.best_val_auc,
                        'best_val_loss': trainer.best_val_loss,
                    })

                    # Log model
                    mlflow.pytorch.log_model(model, "model")

            except Exception as e:
                logger.warning(f"MLflow tracking failed: {e}")
                logger.info("Continuing training without MLflow...")
                trainer.fit(train_loader, val_loader)
        else:
            # Train without MLflow
            trainer.fit(train_loader, val_loader)

        # Plot training curves
        plot_training_curves(
            trainer.train_losses,
            trainer.val_losses,
            trainer.train_metrics,
            trainer.val_metrics,
            save_path=results_dir / 'training_curves.png',
        )

        # Save training history
        history = {
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'train_metrics': trainer.train_metrics,
            'val_metrics': trainer.val_metrics,
        }

        with open(results_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        logger.info("Training completed successfully!")
        logger.info(f"Best model saved to: {trainer.checkpoint_dir / 'best_model.pt'}")
        logger.info(f"Results saved to: {results_dir}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    args = parse_args()
    train_model(args.config, args.checkpoint, args.log_level)
