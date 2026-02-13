# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Training

```bash
# Train with full model (hierarchical + curriculum)
python scripts/train.py

# Train baseline (ablation study)
python scripts/train.py --config configs/ablation.yaml
```

The training script will:
- Load Tox21 dataset
- Train for up to 100 epochs with early stopping
- Save best model to `models/best_model.pt`
- Save training curves to `results/training_curves.png`
- Log metrics at each epoch

## Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py --checkpoint models/best_model.pt

# Outputs:
# - results/evaluation_test.json (metrics)
# - results/predictions_test.csv (predictions)
# - results/confusion_matrix_test.png
# - results/complexity_analysis_test.png
# - results/roc_curve_test.png
```

## Prediction

```bash
# Predict single molecule
python scripts/predict.py --checkpoint models/best_model.pt --smiles "CCO"

# Predict batch from CSV
python scripts/predict.py --checkpoint models/best_model.pt --input molecules.csv --output predictions.csv
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html
```

## Key Configuration Parameters

Edit `configs/default.yaml`:

```yaml
model:
  hidden_dim: 256          # Model capacity
  num_layers: 4            # GNN depth
  num_groups: 16           # Functional group vocabulary

curriculum:
  enable: true             # Enable curriculum learning
  start_epoch: 10          # When to start curriculum
  max_weight: 3.0          # Weight for complex molecules

training:
  num_epochs: 100
  batch_size: 32
  learning_rate: 0.001
  early_stopping_patience: 15
```

## Expected Results

| Model | ROC-AUC | Scaffold AUC | Complex Mol AUC |
|-------|---------|--------------|-----------------|
| Baseline GNN | ~0.81 | ~0.76 | ~0.72 |
| + Hierarchical | ~0.84 | ~0.79 | ~0.76 |
| + Curriculum | ~0.86 | ~0.80 | ~0.78 |

## Troubleshooting

**Out of memory?**
- Reduce `batch_size` in config (try 16 or 8)
- Reduce `hidden_dim` (try 128)

**Training too slow?**
- Use GPU: set `device: cuda` in config
- Enable mixed precision: set `mixed_precision: true`
- Reduce dataset size for testing

**Import errors?**
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python version: requires Python 3.8+

## Next Steps

1. Train the model: `python scripts/train.py`
2. Check results in `results/` directory
3. Run ablation study to compare approaches
4. Adjust hyperparameters in configs for your use case
