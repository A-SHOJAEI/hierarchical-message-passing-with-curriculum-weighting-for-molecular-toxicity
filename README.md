# Hierarchical Message Passing with Curriculum Weighting for Molecular Toxicity

Multi-scale molecular toxicity prediction using hierarchical graph neural networks with adaptive curriculum learning that prioritizes structurally complex molecules during training. Introduces a novel dual-granularity message passing mechanism (atom-level and functional-group-level) combined with difficulty-aware sample weighting based on molecular scaffold complexity.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
# Train the model
python scripts/train.py --config configs/default.yaml

# Evaluate on test set
python scripts/evaluate.py --checkpoint models/best_model.pt

# Predict on new molecules
python scripts/predict.py --checkpoint models/best_model.pt --smiles "CCO"
```

## Usage

### Training
```bash
# Default configuration
python scripts/train.py

# Ablation study (without curriculum weighting)
python scripts/train.py --config configs/ablation.yaml
```

### Evaluation
```bash
python scripts/evaluate.py --checkpoint models/best_model.pt --output results/
```

### Inference
```bash
python scripts/predict.py --checkpoint models/best_model.pt --input data/molecules.csv
```

## Architecture

The model employs a hierarchical message passing architecture:

1. **Atom-level GNN**: Standard message passing on molecular graphs
2. **Functional Group Detection**: Automatic identification of chemical motifs
3. **Group-level GNN**: Higher-order message passing on functional group graphs
4. **Curriculum Weighting**: Adaptive sample weighting based on scaffold complexity
5. **Multi-scale Fusion**: Attention-based combination of atom and group representations

## Training Results

> **Note:** The results below were obtained by training on **synthetic Tox21 molecular data** (randomly generated molecular graphs), not the real Tox21 benchmark. Performance metrics reflect this limitation and should not be compared directly to published Tox21 benchmark results.

### Epoch Progression

| Epoch | Train Loss | Val Loss | Val ROC-AUC | Val Accuracy |
|------:|------------|----------|-------------|--------------|
| 1     | 0.0745     | 0.6743   | 0.4991      | 0.65         |
| 5     | 0.0708     | 0.6532   | 0.5001      | 0.65         |
| 10    | 0.0344     | 0.6533   | 0.5045      | 0.65         |
| 15    | 0.0395     | 0.6541   | 0.5027      | 0.65         |
| 20    | 0.0402     | 0.6568   | 0.5118      | 0.65         |
| 25    | 0.0400     | 0.6586   | 0.4967      | 0.65         |
| 27    | 0.0397     | 0.6622   | 0.5126      | 0.65         |

Training was stopped early at epoch 27 (patience of 15 epochs without improvement on validation loss).

### Best Validation Metrics

| Metric         | Value  | Epoch |
|----------------|--------|------:|
| Best Val Loss  | 0.6510 | 21    |
| Best Val AUC   | 0.5148 | 12    |
| Val Accuracy   | 0.65   | --    |
| Val PR-AUC     | 0.3761 | 12    |

### Training Observations

- **Curriculum learning effect:** At epoch 10 (when curriculum weighting activated), train loss dropped sharply from ~0.070 to ~0.034, indicating the difficulty-aware sample weighting changed the loss landscape substantially.
- **Validation plateau:** Despite the train loss reduction, validation loss remained flat around 0.65--0.66, and validation ROC-AUC hovered near 0.50 (random-chance level). The model did not generalize beyond the training set.
- **Class imbalance:** The model predicted the majority class (negative/non-toxic) for all validation samples across all epochs (precision, recall, and F1 all equal to 0.0 on the positive class). The 65/35 class split (195 negatives, 105 positives) was not overcome.
- **Hardware:** NVIDIA RTX 3090 (24 GB), mixed-precision training enabled.

Run `python scripts/train.py` to reproduce these results.

## Project Structure

```
src/
  hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity/
    data/          # Data loading and preprocessing
    models/        # Model architecture and components
    training/      # Training loop and optimization
    evaluation/    # Metrics and analysis
    utils/         # Configuration and utilities
scripts/
  train.py         # Training pipeline
  evaluate.py      # Evaluation pipeline
  predict.py       # Inference pipeline
configs/
  default.yaml     # Main configuration
  ablation.yaml    # Ablation study config
tests/             # Unit tests
```

## Configuration

Training configuration used for the results above (from `configs/default.yaml`):

| Parameter | Value |
|-----------|-------|
| Hidden dim | 256 |
| GNN layers | 4 |
| Functional group vocab size | 16 |
| Group embedding dim | 128 |
| Group MP layers | 2 |
| Fusion method | Attention |
| Dropout | 0.2 |
| Optimizer | Adam (lr=0.001, weight_decay=1e-5) |
| LR scheduler | Cosine (min_lr=1e-5, warmup=5 epochs) |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping patience | 15 |
| Curriculum start epoch | 10 |
| Curriculum warmup | 5 epochs |
| Difficulty metric | Scaffold complexity |
| Weight schedule | Linear (min=0.5, max=3.0) |
| Data split | Scaffold (80/10/10) |
| Mixed precision | Enabled |
| Seed | 42 |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- DGL 1.1+
- RDKit 2023+

See `requirements.txt` for full dependencies.

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
