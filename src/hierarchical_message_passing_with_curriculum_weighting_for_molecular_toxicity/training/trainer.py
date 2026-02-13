"""Training loop with curriculum learning and advanced features."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from tqdm import tqdm

from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.models.components import (
    CurriculumWeightScheduler,
    FocalLoss,
)
from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.evaluation.metrics import (
    evaluate_model,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer with curriculum learning and advanced training techniques.

    Args:
        model: Model to train
        config: Configuration dictionary
        device: Device to use for training
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Extract training config
        train_config = config.get('training', {})
        self.num_epochs = train_config.get('num_epochs', 100)
        self.gradient_clip = train_config.get('gradient_clip', 1.0)
        self.early_stopping_patience = train_config.get('early_stopping_patience', 15)

        # Optimizer
        opt_config = config.get('optimizer', {})
        opt_type = opt_config.get('type', 'adam').lower()
        lr = config.get('training', {}).get('learning_rate', 0.001)
        weight_decay = config.get('training', {}).get('weight_decay', 1e-5)

        if opt_type == 'adamw':
            self.optimizer = AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=opt_config.get('betas', [0.9, 0.999]),
                eps=opt_config.get('eps', 1e-8),
            )
        else:
            self.optimizer = Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=opt_config.get('betas', [0.9, 0.999]),
                eps=opt_config.get('eps', 1e-8),
            )

        # Learning rate scheduler
        lr_config = config.get('lr_scheduler', {})
        scheduler_type = lr_config.get('type', 'cosine').lower()

        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=lr_config.get('min_lr', 1e-5),
            )
        elif scheduler_type == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=lr_config.get('step_size', 30),
                gamma=lr_config.get('gamma', 0.1),
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=lr_config.get('factor', 0.5),
                patience=lr_config.get('patience', 5),
                min_lr=lr_config.get('min_lr', 1e-5),
            )
        else:
            self.scheduler = None

        # Loss function
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)

        # Curriculum learning
        curriculum_config = config.get('curriculum', {})
        self.use_curriculum = curriculum_config.get('enable', True)

        if self.use_curriculum:
            self.curriculum_scheduler = CurriculumWeightScheduler(
                start_epoch=curriculum_config.get('start_epoch', 10),
                warmup_epochs=curriculum_config.get('warmup_epochs', 5),
                max_weight=curriculum_config.get('max_weight', 3.0),
                min_weight=curriculum_config.get('min_weight', 0.5),
                schedule=curriculum_config.get('weight_schedule', 'linear'),
            )
        else:
            self.curriculum_scheduler = None

        # Mixed precision training
        self.use_amp = config.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Logging
        log_config = config.get('logging', {})
        self.log_interval = log_config.get('log_interval', 10)
        self.checkpoint_dir = Path(log_config.get('checkpoint_dir', 'models'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_metrics: List[Dict[str, float]] = []
        self.val_metrics: List[Dict[str, float]] = []

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_auc = 0.0
        self.patience_counter = 0

        logger.info(f"Initialized Trainer with {opt_type} optimizer and {scheduler_type} scheduler")

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Tuple of (average loss, metrics)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_probs = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{self.num_epochs}')

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch['x'] = batch['x'].to(self.device)
            batch['edge_index'] = batch['edge_index'].to(self.device)
            if 'edge_attr' in batch and batch['edge_attr'] is not None:
                batch['edge_attr'] = batch['edge_attr'].to(self.device)
            batch['batch'] = batch['batch'].to(self.device)
            batch['y'] = batch['y'].to(self.device)
            batch['complexity'] = batch['complexity'].to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch)
                    outputs = outputs.squeeze(-1)  # Only squeeze last dimension

                    # Compute curriculum weights
                    if self.use_curriculum and self.curriculum_scheduler is not None:
                        weights = self.curriculum_scheduler.compute_weights(batch['complexity'])
                    else:
                        weights = None

                    # Compute loss
                    loss = self.criterion(outputs, batch['y'], weights)
            else:
                outputs = self.model(batch)
                outputs = outputs.squeeze(-1)  # Only squeeze last dimension

                # Compute curriculum weights
                if self.use_curriculum and self.curriculum_scheduler is not None:
                    weights = self.curriculum_scheduler.compute_weights(batch['complexity'])
                else:
                    weights = None

                # Compute loss
                loss = self.criterion(outputs, batch['y'], weights)

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Collect predictions
            probs = torch.sigmoid(outputs).detach()
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch['y'].cpu().numpy())

            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({'loss': loss.item()})

        # Compute metrics
        avg_loss = total_loss / num_batches
        y_prob = np.concatenate(all_probs)
        y_true = np.concatenate(all_labels).astype(int)
        y_pred = (y_prob > 0.5).astype(int)

        from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.evaluation.metrics import compute_metrics
        metrics = compute_metrics(y_true, y_pred, y_prob)

        return avg_loss, metrics

    def validate(
        self,
        val_loader: torch.utils.data.DataLoader,
    ) -> Tuple[float, Dict[str, float]]:
        """Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (average loss, metrics)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch['x'] = batch['x'].to(self.device)
                batch['edge_index'] = batch['edge_index'].to(self.device)
                if 'edge_attr' in batch and batch['edge_attr'] is not None:
                    batch['edge_attr'] = batch['edge_attr'].to(self.device)
                batch['batch'] = batch['batch'].to(self.device)
                batch['y'] = batch['y'].to(self.device)

                # Forward pass
                outputs = self.model(batch)
                outputs = outputs.squeeze(-1)  # Only squeeze last dimension

                # Compute loss (no curriculum weighting for validation)
                loss = F.binary_cross_entropy_with_logits(outputs, batch['y'])

                total_loss += loss.item()
                num_batches += 1

                # Collect predictions
                probs = torch.sigmoid(outputs)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(batch['y'].cpu().numpy())

        # Compute metrics
        avg_loss = total_loss / num_batches
        y_prob = np.concatenate(all_probs)
        y_true = np.concatenate(all_labels).astype(int)
        y_pred = (y_prob > 0.5).astype(int)

        from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.evaluation.metrics import compute_metrics
        metrics = compute_metrics(y_true, y_pred, y_prob)

        return avg_loss, metrics

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ) -> None:
        """Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        logger.info("Starting training...")

        for epoch in range(1, self.num_epochs + 1):
            # Update curriculum scheduler
            if self.curriculum_scheduler is not None:
                self.curriculum_scheduler.step(epoch)

            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metrics)

            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)

            # Log
            logger.info(
                f"Epoch {epoch}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train AUC: {train_metrics.get('roc_auc', 0):.4f} - "
                f"Val Loss: {val_loss:.4f}, Val AUC: {val_metrics.get('roc_auc', 0):.4f}"
            )

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Save best model
            val_auc = val_metrics.get('roc_auc', 0)
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_val_loss = val_loss
                self.patience_counter = 0

                checkpoint_path = self.checkpoint_dir / 'best_model.pt'
                self.save_checkpoint(checkpoint_path, epoch, val_loss, val_auc)
                logger.info(f"Saved best model with Val AUC: {val_auc:.4f}")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

            # Save periodic checkpoint
            if epoch % 10 == 0:
                checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
                self.save_checkpoint(checkpoint_path, epoch, val_loss, val_auc)

        logger.info("Training completed!")
        logger.info(f"Best Val AUC: {self.best_val_auc:.4f}, Best Val Loss: {self.best_val_loss:.4f}")

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_loss: float,
        val_auc: float,
    ) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            val_loss: Validation loss
            val_auc: Validation AUC
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_auc': val_auc,
            'config': self.config,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Loaded checkpoint from {path}")
