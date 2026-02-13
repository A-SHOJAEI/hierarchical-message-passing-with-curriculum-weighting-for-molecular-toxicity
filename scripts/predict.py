#!/usr/bin/env python
"""Prediction script for hierarchical molecular GNN."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Union

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
from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.models.model import (
    HierarchicalMolecularGNN,
)
from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.data.preprocessing import (
    mol_to_graph,
    detect_functional_groups,
    compute_scaffold_complexity,
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict with Hierarchical Molecular GNN')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--smiles',
        type=str,
        default=None,
        help='Single SMILES string to predict'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to CSV file with SMILES column'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Output file for predictions'
    )
    parser.add_argument(
        '--smiles-column',
        type=str,
        default='smiles',
        help='Name of SMILES column in input CSV'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    return parser.parse_args()


def predict_smiles(
    model: torch.nn.Module,
    smiles_list: List[str],
    config: dict,
    device: torch.device,
) -> List[dict]:
    """Predict toxicity for a list of SMILES.

    Args:
        model: Trained model
        smiles_list: List of SMILES strings
        config: Configuration dictionary
        device: Device to use

    Returns:
        List of prediction dictionaries
    """
    from rdkit import Chem

    model.eval()
    predictions = []

    num_groups = config.get('model', {}).get('num_groups', 16)

    with torch.no_grad():
        for smiles in smiles_list:
            try:
                # Convert to graph
                graph = mol_to_graph(smiles)

                # Detect functional groups
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mol = Chem.AddHs(mol)
                    groups, group_features = detect_functional_groups(mol, num_groups)
                else:
                    groups = [[]]
                    group_features = torch.zeros(1, num_groups)

                # Compute complexity
                complexity = compute_scaffold_complexity(smiles)

                # Create batch
                batch = {
                    'x': graph.x.to(device),
                    'edge_index': graph.edge_index.to(device),
                    'edge_attr': graph.edge_attr.to(device),
                    'batch': torch.zeros(graph.x.size(0), dtype=torch.long).to(device),
                    'groups': [groups],
                    'group_features': [group_features],
                }

                # Predict
                output = model(batch)
                prob = torch.sigmoid(output).item()
                pred = int(prob > 0.5)

                predictions.append({
                    'smiles': smiles,
                    'prediction': pred,
                    'probability': prob,
                    'complexity': complexity,
                    'status': 'success',
                })

            except Exception as e:
                logger.warning(f"Error predicting for {smiles}: {e}")
                predictions.append({
                    'smiles': smiles,
                    'prediction': None,
                    'probability': None,
                    'complexity': None,
                    'status': f'error: {str(e)}',
                })

    return predictions


def predict(
    checkpoint_path: str,
    smiles: str = None,
    input_file: str = None,
    output_file: str = 'predictions.csv',
    smiles_column: str = 'smiles',
    log_level: str = 'INFO',
):
    """Run predictions.

    Args:
        checkpoint_path: Path to model checkpoint
        smiles: Single SMILES string to predict
        input_file: Path to CSV file with SMILES
        output_file: Output file for predictions
        smiles_column: Name of SMILES column in input CSV
        log_level: Logging level
    """
    # Setup logging
    setup_logging(log_level)
    logger.info("=" * 80)
    logger.info("Model Prediction")
    logger.info("=" * 80)

    try:
        # Validate inputs
        if smiles is None and input_file is None:
            raise ValueError("Either --smiles or --input must be provided")

        # Load checkpoint
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load configuration
        config = checkpoint.get('config')
        if config is None:
            raise ValueError("Config not found in checkpoint")

        # Get device
        device = get_device(config.get('device', 'cuda'))

        # Create and load model
        logger.info("Loading model...")
        model = HierarchicalMolecularGNN(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        # Get SMILES list
        if smiles is not None:
            smiles_list = [smiles]
        else:
            logger.info(f"Loading SMILES from {input_file}")
            df = pd.read_csv(input_file)
            if smiles_column not in df.columns:
                raise ValueError(f"Column '{smiles_column}' not found in {input_file}")
            smiles_list = df[smiles_column].tolist()

        logger.info(f"Predicting for {len(smiles_list)} molecules...")

        # Predict
        predictions = predict_smiles(model, smiles_list, config, device)

        # Save predictions
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(output_file, index=False)

        logger.info(f"\nPredictions saved to {output_file}")

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("Prediction Summary")
        logger.info("=" * 80)
        logger.info(f"Total molecules: {len(predictions)}")

        successful = sum(1 for p in predictions if p['status'] == 'success')
        logger.info(f"Successful predictions: {successful}")

        if successful > 0:
            toxic_count = sum(1 for p in predictions if p['prediction'] == 1 and p['status'] == 'success')
            logger.info(f"Predicted toxic: {toxic_count} ({100*toxic_count/successful:.1f}%)")
            logger.info(f"Predicted non-toxic: {successful - toxic_count} ({100*(successful-toxic_count)/successful:.1f}%)")

            avg_prob = sum(p['probability'] for p in predictions if p['status'] == 'success') / successful
            logger.info(f"Average probability: {avg_prob:.4f}")

        # Print sample predictions
        if smiles is None and len(predictions) > 5:
            logger.info("\nSample predictions (first 5):")
            for i, pred in enumerate(predictions[:5], 1):
                if pred['status'] == 'success':
                    logger.info(
                        f"{i}. {pred['smiles'][:50]}... "
                        f"-> {'Toxic' if pred['prediction'] == 1 else 'Non-toxic'} "
                        f"(prob: {pred['probability']:.4f}, complexity: {pred['complexity']:.4f})"
                    )
                else:
                    logger.info(f"{i}. {pred['smiles'][:50]}... -> {pred['status']}")
        elif smiles:
            # Single prediction
            pred = predictions[0]
            if pred['status'] == 'success':
                logger.info("\nPrediction:")
                logger.info(f"  SMILES: {pred['smiles']}")
                logger.info(f"  Prediction: {'Toxic' if pred['prediction'] == 1 else 'Non-toxic'}")
                logger.info(f"  Probability: {pred['probability']:.4f}")
                logger.info(f"  Complexity: {pred['complexity']:.4f}")
            else:
                logger.info(f"\nPrediction failed: {pred['status']}")

        logger.info("\nPrediction completed successfully!")

    except Exception as e:
        logger.error(f"Prediction failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    args = parse_args()
    predict(
        args.checkpoint,
        args.smiles,
        args.input,
        args.output,
        args.smiles_column,
        args.log_level,
    )
