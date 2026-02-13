# Setup Guide

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run training**:
```bash
python scripts/train.py --config configs/default.yaml
```

3. **Run evaluation**:
```bash
python scripts/evaluate.py --checkpoint models/best_model.pt
```

4. **Run prediction**:
```bash
python scripts/predict.py --checkpoint models/best_model.pt --smiles "CCO"
```

## Ablation Study

Compare the full model with the baseline (without hierarchical message passing and curriculum weighting):

```bash
# Train baseline
python scripts/train.py --config configs/ablation.yaml

# Evaluate baseline
python scripts/evaluate.py --checkpoint models/best_model.pt --output results/ablation
```

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=html
```

## Project Structure

```
hierarchical-message-passing-with-curriculum-weighting-for-molecular-toxicity/
├── src/                           # Source code
│   └── hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity/
│       ├── data/                  # Data loading and preprocessing
│       ├── models/                # Model architecture
│       ├── training/              # Training loop
│       ├── evaluation/            # Metrics and analysis
│       └── utils/                 # Utilities
├── scripts/                       # Executable scripts
│   ├── train.py                   # Training pipeline
│   ├── evaluate.py                # Evaluation pipeline
│   └── predict.py                 # Inference pipeline
├── configs/                       # YAML configurations
│   ├── default.yaml              # Full model config
│   └── ablation.yaml             # Baseline config
├── tests/                         # Unit tests
├── models/                        # Saved models
├── results/                       # Evaluation results
└── data/                          # Data directory
```

## Key Features

1. **Hierarchical Message Passing**: Dual-granularity GNN operating at both atom and functional-group levels
2. **Curriculum Learning**: Adaptive sample weighting based on molecular scaffold complexity
3. **Custom Components**:
   - `CurriculumWeightScheduler`: Progressive difficulty-based weighting
   - `FocalLoss`: Custom loss for handling class imbalance
   - `AttentionPooling`: Attention-based graph pooling
   - `MultiScaleFusion`: Fusion of atom and group representations

4. **Advanced Training**:
   - Cosine learning rate scheduling
   - Gradient clipping
   - Early stopping
   - Mixed precision training support
   - MLflow tracking (optional)

## Configuration

Edit `configs/default.yaml` to adjust:
- Model architecture (hidden_dim, num_layers, etc.)
- Training hyperparameters (learning_rate, batch_size, etc.)
- Curriculum learning settings
- Data splitting strategy

## Hardware Requirements

- Recommended: GPU with 8GB+ VRAM
- Minimum: CPU with 16GB RAM
- Training time: ~30-60 minutes on GPU for full dataset
