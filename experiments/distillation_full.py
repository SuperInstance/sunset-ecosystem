#!/usr/bin/env python3
"""distillation_full.py — Production-grade knowledge distillation on CIFAR-10.

Three loss variants in one script:
  1. Hard targets (CE vs ground-truth labels)
  2. Soft targets (KL vs teacher logits, temperature T)
  3. Feature matching (MSE on intermediate activations)

Usage:
    python distillation_full.py --config distillation_config.yaml
    python distillation_full.py --smoke-test   # verify imports + 1 batch
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

import yaml

# ── Ensure project root on path ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════
#  Models
# ═══════════════════════════════════════════════════════════════════

class FeatureExtractor(nn.Module):
    """Wraps a torchvision model so we can grab intermediate layers."""

    def __init__(self, arch: str, num_classes: int = 10):
        super().__init__()
        if arch == "resnet50":
            base = torchvision.models.resnet50(weights=None, num_classes=num_classes)
            self._feat_channels = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
        elif arch == "resnet18":
            base = torchvision.models.resnet18(weights=None, num_classes=num_classes)
            self._feat_channels = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}
        else:
            raise ValueError(f"Unknown arch: {arch}")
        self.base = base
        self._features: dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self):
        def hook(name):
            def fn(_, __, out):
                self._features[name] = out
            return fn
        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            if hasattr(self.base, layer_name):
                getattr(self.base, layer_name).register_forward_hook(hook(layer_name))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._features.clear()
        logits = self.base(x)
        return logits, dict(self._features)

    @property
    def feat_channels(self) -> dict[str, int]:
        return self._feat_channels


# ═══════════════════════════════════════════════════════════════════
#  Distillation Loss
# ═══════════════════════════════════════════════════════════════════

class ChannelAdapter(nn.Module):
    """1x1 conv to match teacher channels → student channels for feature distillation."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DistillationLoss(nn.Module):
    """Combines hard CE + soft KL + feature MSE."""

    def __init__(
        self,
        teacher_channels: dict[str, int],
        student_channels: dict[str, int],
        temperature: float = 4.0,
        alpha_hard: float = 0.5,
        alpha_soft: float = 0.4,
        alpha_feat: float = 0.1,
        feat_layers: list[list[str]] | None = None,
    ):
        super().__init__()
        self.T = temperature
        self.alpha_hard = alpha_hard
        self.alpha_soft = alpha_soft
        self.alpha_feat = alpha_feat
        self.feat_layers = feat_layers or [["layer2", "layer2"], ["layer3", "layer3"]]
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        # Learnable adapters: teacher feature channels → student feature channels
        self.adapters = nn.ModuleDict()
        for t_layer, s_layer in self.feat_layers:
            if t_layer in teacher_channels and s_layer in student_channels:
                t_ch = teacher_channels[t_layer]
                s_ch = student_channels[s_layer]
                if t_ch != s_ch:
                    self.adapters[t_layer] = ChannelAdapter(t_ch, s_ch)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_feats: dict[str, torch.Tensor],
        teacher_feats: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # 1) Hard target
        loss_hard = self.ce(student_logits, labels)

        # 2) Soft target (KL div)
        p_teacher = F.softmax(teacher_logits / self.T, dim=1).detach()
        p_student = F.log_softmax(student_logits / self.T, dim=1)
        loss_soft = F.kl_div(p_student, p_teacher, reduction="batchmean") * (self.T ** 2)

        # 3) Feature matching
        loss_feat = 0.0
        matched = 0
        for t_layer, s_layer in self.feat_layers:
            if t_layer in teacher_feats and s_layer in student_feats:
                t_feat = teacher_feats[t_layer].detach()
                s_feat = student_feats[s_layer]
                # Match channels via adapter if needed
                if t_layer in self.adapters:
                    t_feat = self.adapters[t_layer](t_feat)
                # Match spatial dims
                if t_feat.shape[-2:] != s_feat.shape[-2:]:
                    t_feat = F.adaptive_avg_pool2d(t_feat, s_feat.shape[-2:])
                loss_feat += self.mse(s_feat, t_feat)
                matched += 1
        if matched:
            loss_feat = loss_feat / matched

        total = (
            self.alpha_hard * loss_hard
            + self.alpha_soft * loss_soft
            + self.alpha_feat * loss_feat
        )
        metrics = {
            "hard": loss_hard.item(),
            "soft": loss_soft.item(),
            "feat": loss_feat.item() if isinstance(loss_feat, torch.Tensor) else loss_feat,
            "total": total.item(),
        }
        return total, metrics


# ═══════════════════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════════════════

def get_loaders(cfg: dict, data_root: str | None = None):
    root = data_root or cfg["dataset"]["root"]
    batch = cfg["dataset"]["batch_size"]
    workers = cfg["dataset"]["num_workers"]

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_val = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
    val_ds = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform_val)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════
#  Training / Validation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: FeatureExtractor, loader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train_epoch(
    teacher: FeatureExtractor,
    student: FeatureExtractor,
    loader: DataLoader,
    criterion: DistillationLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    print_every: int,
    epoch: int,
):
    student.train()
    teacher.eval()

    running_metrics: dict[str, float] = {}
    num_batches = 0

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                t_logits, t_feats = teacher(images)
                s_logits, s_feats = student(images)
                loss, metrics = criterion(s_logits, t_logits, labels, s_feats, t_feats)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            t_logits, t_feats = teacher(images)
            s_logits, s_feats = student(images)
            loss, metrics = criterion(s_logits, t_logits, labels, s_feats, t_feats)
            loss.backward()
            optimizer.step()

        # Accumulate metrics
        for k, v in metrics.items():
            running_metrics[k] = running_metrics.get(k, 0.0) + v
        num_batches += 1

        if (i + 1) % print_every == 0:
            avg = {k: v / num_batches for k, v in running_metrics.items()}
            print(f"  [Epoch {epoch} Batch {i+1}] loss={avg['total']:.4f} (hard={avg['hard']:.3f} soft={avg['soft']:.3f} feat={avg['feat']:.3f})")

    return {k: v / num_batches for k, v in running_metrics.items()}


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def build_optimizer(model: nn.Module, cfg: dict):
    lr = cfg["training"]["lr"]
    wd = cfg["training"]["weight_decay"]
    mom = cfg["training"]["momentum"]
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd, nesterov=True)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, steps_per_epoch: int):
    epochs = cfg["training"]["epochs"]
    warmup = cfg["training"]["warmup_epochs"]
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    import math
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run(cfg: dict, smoke_test: bool = False):
    device_str = cfg["hardware"]["device"]
    device = torch.device("cuda" if torch.cuda.is_available() and device_str in ("auto", "cuda") else "cpu")
    use_amp = cfg["hardware"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print(f"Device: {device} | AMP: {use_amp}")

    # Models
    teacher = FeatureExtractor(cfg["models"]["teacher"]).to(device)
    student = FeatureExtractor(cfg["models"]["student"]).to(device)

    # Optional: load pretrained teacher weights here
    if cfg["models"]["pretrained_teacher"]:
        pt_path = Path(cfg["models"]["pretrained_teacher"])
        if pt_path.exists():
            teacher.load_state_dict(torch.load(pt_path, map_location=device))
            print(f"Loaded teacher from {pt_path}")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Data
    train_loader, val_loader = get_loaders(cfg)

    if smoke_test:
        print("🔥 SMOKE TEST — running 1 batch only")
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            t_logits, t_feats = teacher(images)
            s_logits, s_feats = student(images)
        print(f"  Teacher logits shape: {t_logits.shape}  features: {[v.shape for v in t_feats.values()]}")
        print(f"  Student logits shape: {s_logits.shape}  features: {[v.shape for v in s_feats.values()]}")
        # Quick forward through loss
        criterion = DistillationLoss(
            teacher_channels=teacher.feat_channels,
            student_channels=student.feat_channels,
            **cfg["distillation"],
        )
        loss, metrics = criterion(s_logits, t_logits, labels, s_feats, t_feats)
        print(f"  Loss computed: {loss.item():.4f}  metrics: {metrics}")
        print("✅ Smoke test passed")
        return None

    # Loss / optimizer / scheduler
    criterion = DistillationLoss(
        teacher_channels=teacher.feat_channels,
        student_channels=student.feat_channels,
        **cfg["distillation"],
    ).to(device)
    optimizer = build_optimizer(student, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    # Logging
    log_dir = Path(cfg["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / cfg["logging"]["log_file"]
    csv_file = open(log_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["epoch", "time_sec", "train_loss", "train_hard", "train_soft", "train_feat", "val_acc", "best"])

    # Checkpointing
    save_dir = Path(cfg["checkpointing"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    epochs = cfg["training"]["epochs"]
    print_every = cfg["logging"]["print_every"]

    print(f"\nTraining {epochs} epochs — student={cfg['models']['student']} teacher={cfg['models']['teacher']}")
    print(f"Loss weights: hard={cfg['distillation']['alpha_hard']} soft={cfg['distillation']['alpha_soft']} feat={cfg['distillation']['alpha_feat']}")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(teacher, student, train_loader, criterion, optimizer, device, scaler, print_every, epoch)
        val_acc = evaluate(student, val_loader, device)
        elapsed = time.time() - t0

        # Step scheduler every epoch
        scheduler.step()

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_path = save_dir / "student_best.pt"
            torch.save(student.state_dict(), best_path)

        # Logging
        writer.writerow([
            epoch,
            f"{elapsed:.1f}",
            f"{train_metrics['total']:.4f}",
            f"{train_metrics['hard']:.4f}",
            f"{train_metrics['soft']:.4f}",
            f"{train_metrics['feat']:.4f}",
            f"{val_acc:.4f}",
            "*" if is_best else "",
        ])
        csv_file.flush()

        print(f"Epoch {epoch:3d}/{epochs} | {elapsed:.1f}s | loss={train_metrics['total']:.4f} | val_acc={val_acc:.4f} {'*** BEST ***' if is_best else ''}")

        # Periodic save
        if cfg["checkpointing"]["save_every"] and epoch % cfg["checkpointing"]["save_every"] == 0:
            torch.save(student.state_dict(), save_dir / f"student_epoch{epoch}.pt")

    csv_file.close()
    print(f"\n🏁 Finished. Best val accuracy: {best_acc:.4f}")
    print(f"📄 Logs: {log_path}")
    print(f"💾 Best checkpoint: {save_dir / 'student_best.pt'}")
    return best_acc


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Knowledge Distillation")
    parser.add_argument("--config", "-c", default=str(PROJECT_ROOT / "experiments" / "distillation_config.yaml"), help="YAML config path")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-batch sanity check and exit")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    run(cfg, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
