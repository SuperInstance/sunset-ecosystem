# Distillation Full Experiment

PyTorch-based knowledge distillation on CIFAR-10.

## Quick Start

```bash
pip install torch torchvision pyyaml
python experiments/distillation_full.py --config experiments/distillation_config.yaml
```

## Expected Results

- Teacher (ResNet-50): ~93-94% accuracy on CIFAR-10
- Student (ResNet-18) with hard targets: ~88-90%
- Student with soft targets (T=4): ~90-91%
- Student with feature matching: ~91-92%
- Full training: 50-100 epochs recommended

## Architecture

```
CIFAR-10 (32x32 RGB)
    │
    ▼
┌─────────────────────┐
│   Teacher (frozen)  │  ResNet-50 → logits + features
│   Accuracy: ~93%    │
└─────────────────────┘
    │
    ├── logits (soft targets, T=4)
    ├── intermediate features (for MSE loss)
    │
    ▼
┌─────────────────────┐
│   Student (train)   │  ResNet-18 → logits
│   Target: ~91%      │
└─────────────────────┘
    │
    ├── CrossEntropy vs hard labels (λ₁ = 0.5)
    ├── KL Divergence vs soft targets (λ₂ = 0.4)
    └── MSE vs teacher features  (λ₃ = 0.1)
```

## Hardware Requirements

- CPU: works (slow, ~30 min/epoch)
- GPU: 1x NVIDIA GPU with 4GB+ VRAM (recommended, ~3 min/epoch)
- RAM: 4GB minimum
- Disk: ~500MB for CIFAR-10 download

## Files

- `experiments/distillation_full.py` — main training script
- `experiments/distillation_config.yaml` — hyperparameters
- `logs/distillation.csv` — per-epoch loss/accuracy log
- `checkpoints/student_best.pt` — best model by val accuracy

## Citation

Hinton et al. (2015). "Distilling the Knowledge in a Neural Network."
Romero et al. (2015). "FitNets: Hints for Thin Deep Nets."
