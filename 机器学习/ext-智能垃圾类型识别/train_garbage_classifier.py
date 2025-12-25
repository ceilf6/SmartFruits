import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    arch: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    img_size: int
    num_workers: int
    seed: int
    device: str
    amp: bool
    label_smoothing: float
    patience: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    all_preds = []
    all_targets = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = loss_fn(logits, targets)
        total_loss += loss.item()
        total_acc += accuracy_top1(logits, targets)
        n_batches += 1

        all_preds.append(logits.argmax(dim=1).detach().cpu())
        all_targets.append(targets.detach().cpu())

    if n_batches == 0:
        return {"loss": float("nan"), "acc": float("nan"), "confusion": None}

    preds = torch.cat(all_preds)
    targs = torch.cat(all_targets)
    num_classes = int(max(preds.max(), targs.max()).item()) + 1 if preds.numel() else 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(targs.tolist(), preds.tolist()):
        confusion[t, p] += 1

    return {"loss": total_loss / n_batches, "acc": total_acc / n_batches, "confusion": confusion}


def build_model(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower().strip()
    if arch == "resnet18":
        # 兼容 torchvision 旧版本（0.8.x）: 使用 pretrained=True
        try:
            weights = getattr(models, "ResNet18_Weights").DEFAULT  # torchvision>=0.13
            model = models.resnet18(weights=weights)
        except Exception:
            model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "mobilenet_v3_small":
        if not hasattr(models, "mobilenet_v3_small"):
            raise ValueError("当前 torchvision 版本不支持 mobilenet_v3_small，请升级 torchvision 或使用 --arch resnet18")
        try:
            weights = getattr(models, "MobileNet_V3_Small_Weights").DEFAULT
            model = models.mobilenet_v3_small(weights=weights)
        except Exception:
            model = models.mobilenet_v3_small(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if arch == "efficientnet_b0":
        if not hasattr(models, "efficientnet_b0"):
            raise ValueError("当前 torchvision 版本不支持 efficientnet_b0，请升级 torchvision 或使用 --arch resnet18")
        try:
            weights = getattr(models, "EfficientNet_B0_Weights").DEFAULT
            model = models.efficientnet_b0(weights=weights)
        except Exception:
            model = models.efficientnet_b0(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"不支持的 arch: {arch}（可选: resnet18 / mobilenet_v3_small / efficientnet_b0）")


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_tf, val_tf


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="智能垃圾分类（图像分类）模型训练脚本（迁移学习）")
    parser.add_argument("--data-dir", type=str, required=True, help="数据集目录，结构: data-dir/train/类名/*.jpg 与 data-dir/val/类名/*.jpg")
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).parent / "runs_garbage"), help="输出目录")
    parser.add_argument("--arch", type=str, default="resnet18", help="backbone: resnet18 / mobilenet_v3_small / efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--img-size", type=int, default=224, help="输入尺寸（分类模型通常 224）")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cuda / cpu / cuda:0")
    parser.add_argument("--amp", action="store_true", help="启用混合精度（仅 CUDA 有效）")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="标签平滑系数（0~0.2 常见）")
    parser.add_argument("--patience", type=int, default=7, help="早停耐心值（验证集不提升的轮数）")
    args = parser.parse_args()
    return TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        img_size=args.img_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        amp=bool(args.amp),
        label_smoothing=args.label_smoothing,
        patience=args.patience,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    data_dir = Path(cfg.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"数据目录结构不正确：需要存在 {train_dir} 与 {val_dir}\n"
            f"示例：\n"
            f"  data/train/plastic/*.jpg\n"
            f"  data/train/paper/*.jpg\n"
            f"  data/val/plastic/*.jpg\n"
            f"  data/val/paper/*.jpg"
        )

    out_dir = Path(cfg.out_dir) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    # 保存配置，方便写报告/复现实验
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    device = torch.device(cfg.device)
    train_tf, val_tf = build_transforms(cfg.img_size)
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)

    class_names = train_ds.classes
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError(f"类别数过少（num_classes={num_classes}），请检查 {train_dir} 下的子文件夹（每个子文件夹为一个类别）")

    # 类别映射（报告里经常需要）
    with open(out_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in enumerate(class_names)}, f, ensure_ascii=False, indent=2)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(cfg.arch, num_classes=num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    use_amp = bool(cfg.amp) and device.type == "cuda"
    # 兼容 torch>=1.7: 使用 torch.cuda.amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_acc = -1.0
    best_epoch = -1
    bad_epochs = 0

    log_path = out_dir / "metrics.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr,seconds\n")

    print("====== 智能垃圾分类模型训练 ======")
    print(f"data_dir: {data_dir}")
    print(f"out_dir:  {out_dir}")
    print(f"arch:     {cfg.arch}")
    print(f"classes({num_classes}): {class_names}")
    print(f"device:   {device}  amp={use_amp}")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_acc += accuracy_top1(logits.detach(), targets)
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        train_acc = running_acc / max(n_batches, 1)

        val_metrics = evaluate(model, val_loader, device)
        val_loss = float(val_metrics["loss"])
        val_acc = float(val_metrics["acc"])

        scheduler.step()
        lr = float(optimizer.param_groups[0]["lr"])
        seconds = time.time() - t0

        ckpt_last = out_dir / "checkpoints" / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "classes": class_names,
                "arch": cfg.arch,
                "img_size": cfg.img_size,
                "val_acc": val_acc,
            },
            ckpt_last,
        )

        improved = val_acc > best_val_acc + 1e-6
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            bad_epochs = 0
            ckpt_best = out_dir / "checkpoints" / "best.pt"
            torch.save(torch.load(ckpt_last, map_location="cpu"), ckpt_best)
        else:
            bad_epochs += 1

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{lr:.8f},{seconds:.2f}\n")

        print(
            f"[{epoch:03d}/{cfg.epochs}] "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"lr={lr:.2e} | {seconds:.1f}s "
            f"{'(best)' if improved else ''}"
        )

        if cfg.patience > 0 and bad_epochs >= cfg.patience:
            print(f"验证集连续 {bad_epochs} 轮未提升，触发早停（best_epoch={best_epoch}, best_acc={best_val_acc:.4f}）。")
            break

    # 训练结束后输出混淆矩阵，便于报告分析
    best_path = out_dir / "checkpoints" / "best.pt"
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model_state"])
        val_metrics = evaluate(model, val_loader, device)
        confusion = val_metrics["confusion"]
        if confusion is not None:
            np.save(out_dir / "confusion.npy", confusion.numpy())

    print("====== 训练完成 ======")
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_acc: {best_val_acc:.4f}")
    print(f"checkpoints: {(out_dir / 'checkpoints').resolve()}")
    print(f"logs: {log_path.resolve()}")


if __name__ == "__main__":
    # 允许在某些环境下限制线程数，避免 Mac/Windows 卡顿
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()


