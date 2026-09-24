from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import segmentation_models_pytorch as smp


DEFAULT_TRAIN_CSV = r"C:\SlickTrace\outputs\train.csv"
DEFAULT_VAL_CSV = r"C:\SlickTrace\outputs\val.csv"
DEFAULT_OUTPUT_DIR = r"C:\SlickTrace\models\deeplabv3plus_mobilenetv2"

IMAGE_SIZE = 2048
IN_CHANNELS = 2
NUM_CLASSES = 1

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

BATCH_SIZE = 1
DEFAULT_EPOCHS = 10

SEED = 42


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_image(path):
    with rasterio.open(path) as src:
        image = src.read().astype(np.float32)

    if image.shape[0] != IN_CHANNELS:
        raise ValueError(
            f"Expected {IN_CHANNELS} input bands, got {image.shape[0]}: {path}"
        )

    normalized = np.zeros_like(image, dtype=np.float32)

    for channel in range(image.shape[0]):
        band = image[channel]

        low = np.percentile(band, 1)
        high = np.percentile(band, 99)

        if high <= low:
            normalized[channel] = 0.0
        else:
            normalized[channel] = np.clip(
                (band - low) / (high - low),
                0.0,
                1.0,
            )

    return normalized


def load_mask(path):
    with rasterio.open(path) as src:
        mask = src.read(1).astype(np.float32)

    return (mask > 0).astype(np.float32)


class SlickTraceDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        required_columns = {
            "image_path",
            "mask_path",
            "label",
        }

        missing = required_columns - set(self.df.columns)

        if missing:
            raise ValueError(
                f"Missing columns in {csv_path}: {sorted(missing)}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image = load_image(row["image_path"])
        mask = load_mask(row["mask_path"])

        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return image_tensor, mask_tensor


def freeze_batchnorm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

            for parameter in module.parameters():
                parameter.requires_grad = False


def dice_loss(logits, target):
    probability = torch.sigmoid(logits)

    smooth = 1.0

    intersection = (
        probability * target
    ).sum(dim=(1, 2, 3))

    denominator = (
        probability.sum(dim=(1, 2, 3))
        + target.sum(dim=(1, 2, 3))
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    return 1.0 - dice.mean()


def combined_loss(logits, target, bce_loss):
    loss_bce = bce_loss(logits, target)
    loss_dice = dice_loss(logits, target)

    return loss_bce + loss_dice, loss_bce, loss_dice


def segmentation_metrics(logits, target):
    probability = torch.sigmoid(logits)
    prediction = probability >= 0.5
    target_bool = target >= 0.5

    intersection = (
        prediction & target_bool
    ).sum().item()

    union = (
        prediction | target_bool
    ).sum().item()

    predicted_positive = prediction.sum().item()
    actual_positive = target_bool.sum().item()

    true_positive = intersection

    false_positive = max(
        predicted_positive - true_positive,
        0,
    )

    false_negative = max(
        actual_positive - true_positive,
        0,
    )

    dice_denominator = (
        predicted_positive + actual_positive
    )

    if dice_denominator == 0:
        dice = 1.0
    else:
        dice = (
            2.0 * true_positive
        ) / dice_denominator

    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union

    precision_denominator = (
        true_positive + false_positive
    )

    recall_denominator = (
        true_positive + false_negative
    )

    if precision_denominator == 0:
        precision = 1.0
    else:
        precision = (
            true_positive
            / precision_denominator
        )

    if recall_denominator == 0:
        recall = 1.0
    else:
        recall = (
            true_positive
            / recall_denominator
        )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def create_model():
    model = smp.DeepLabV3Plus(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        activation=None,
    )

    return model


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_dice,
    config,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_dice": best_dice,
        "config": config,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
):
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    start_epoch = checkpoint["epoch"] + 1

    best_dice = checkpoint.get(
        "best_dice",
        0.0,
    )

    return start_epoch, best_dice


def train_one_epoch(
    model,
    loader,
    optimizer,
    bce_loss,
    device,
    scaler,
    amp_enabled,
    epoch,
    total_epochs,
):
    model.train()
    freeze_batchnorm(model)

    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0

    start_time = time.perf_counter()

    for step, (images, masks) in enumerate(loader, start=1):
        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=amp_enabled,
        ):
            logits = model(images)

            loss, loss_bce, loss_dice = combined_loss(
                logits,
                masks,
                bce_loss,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_bce += loss_bce.item()
        total_dice_loss += loss_dice.item()

        if step == 1 or step % 25 == 0 or step == len(loader):
            elapsed = time.perf_counter() - start_time

            print(
                f"Epoch {epoch}/{total_epochs} "
                f"Step {step}/{len(loader)} "
                f"Loss {loss.item():.4f} "
                f"Time {elapsed:.1f}s"
            )

    count = len(loader)

    return {
        "loss": total_loss / count,
        "bce": total_bce / count,
        "dice_loss": total_dice_loss / count,
    }


@torch.no_grad()
def validate(
    model,
    loader,
    bce_loss,
    device,
    amp_enabled,
):
    model.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0

    total_intersection = 0
    total_union = 0
    total_predicted_positive = 0
    total_actual_positive = 0

    for images, masks in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=amp_enabled,
        ):
            logits = model(images)

            loss, loss_bce, loss_dice = combined_loss(
                logits,
                masks,
                bce_loss,
            )

        total_loss += loss.item()
        total_bce += loss_bce.item()
        total_dice_loss += loss_dice.item()

        probability = torch.sigmoid(logits)
        prediction = probability >= 0.5
        target = masks >= 0.5

        intersection = (
            prediction & target
        ).sum().item()

        union = (
            prediction | target
        ).sum().item()

        predicted_positive = prediction.sum().item()
        actual_positive = target.sum().item()

        total_intersection += intersection
        total_union += union
        total_predicted_positive += predicted_positive
        total_actual_positive += actual_positive

    count = len(loader)

    if (
        total_predicted_positive
        + total_actual_positive
        == 0
    ):
        dice = 1.0
    else:
        dice = (
            2.0 * total_intersection
        ) / (
            total_predicted_positive
            + total_actual_positive
        )

    if total_union == 0:
        iou = 1.0
    else:
        iou = (
            total_intersection
            / total_union
        )

    precision_denominator = (
        total_intersection
        + total_predicted_positive
        - total_intersection
    )

    recall_denominator = (
        total_actual_positive
    )

    if precision_denominator == 0:
        precision = 1.0
    else:
        precision = (
            total_intersection
            / precision_denominator
        )

    if recall_denominator == 0:
        recall = 1.0
    else:
        recall = (
            total_intersection
            / recall_denominator
        )

    return {
        "loss": total_loss / count,
        "bce": total_bce / count,
        "dice_loss": total_dice_loss / count,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-csv",
        default=DEFAULT_TRAIN_CSV,
    )

    parser.add_argument(
        "--val-csv",
        default=DEFAULT_VAL_CSV,
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional limit for a short GPU smoke test.",
    )

    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional limit for a short validation smoke test.",
    )

    args = parser.parse_args()

    set_seed(SEED)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    amp_enabled = device.type == "cuda"

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print("SlickTrace AI training")
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "GPU memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / 1024**3,
                2,
            ),
            "GB",
        )

    print("Image size:", IMAGE_SIZE)
    print("Input channels:", IN_CHANNELS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Epochs:", args.epochs)
    print("AMP:", amp_enabled)
    print("Workers:", args.workers)

    train_dataset = SlickTraceDataset(
        args.train_csv
    )

    val_dataset = SlickTraceDataset(
        args.val_csv
    )

    if args.max_train_samples is not None:
        if args.max_train_samples < 1:
            raise ValueError("--max-train-samples must be >= 1")

        limit = min(
            args.max_train_samples,
            len(train_dataset),
        )

        train_dataset = Subset(
            train_dataset,
            range(limit),
        )

    if args.max_val_samples is not None:
        if args.max_val_samples < 1:
            raise ValueError("--max-val-samples must be >= 1")

        limit = min(
            args.max_val_samples,
            len(val_dataset),
        )

        val_dataset = Subset(
            val_dataset,
            range(limit),
        )

    print(
        "Training samples:",
        len(train_dataset),
    )

    print(
        "Validation samples:",
        len(val_dataset),
    )

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty.")

    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=amp_enabled,
        persistent_workers=args.workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=amp_enabled,
        persistent_workers=args.workers > 0,
    )

    model = create_model().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    bce_loss = nn.BCEWithLogitsLoss()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    checkpoint_path = (
        output_dir / "last_checkpoint.pt"
    )

    best_model_path = (
        output_dir / "best_model.pt"
    )

    history_path = (
        output_dir / "training_history.csv"
    )

    config = {
        "image_size": IMAGE_SIZE,
        "input_channels": IN_CHANNELS,
        "architecture": "DeepLabV3Plus",
        "encoder": "mobilenet_v2",
        "encoder_weights": None,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss": "BCEWithLogitsLoss + DiceLoss",
        "normalization": "per-image per-band percentile 1-99 to 0-1",
        "augmentation": "none",
        "seed": SEED,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "workers": args.workers,
    }

    start_epoch = 1
    best_dice = 0.0

    if args.resume and checkpoint_path.exists():
        print(
            "Resuming from:",
            checkpoint_path,
        )

        start_epoch, best_dice = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )

        print(
            "Resume epoch:",
            start_epoch,
        )

        print(
            "Previous best Dice:",
            best_dice,
        )

    history = []

    if args.resume and history_path.exists():
        try:
            existing_history = pd.read_csv(
                history_path
            )

            history = existing_history.to_dict(
                orient="records"
            )

            print(
                "Loaded history rows:",
                len(history),
            )
        except Exception as exc:
            print(
                "Warning: could not load previous history:",
                exc,
            )

    if start_epoch > args.epochs:
        print(
            f"No training required: checkpoint is already at epoch "
            f"{start_epoch - 1}, requested total epochs={args.epochs}."
        )
        print("Training finished.")
        print("Best validation Dice:", best_dice)
        print("Best model:", best_model_path)
        print("Last checkpoint:", checkpoint_path)
        print("History:", history_path)
        return

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            bce_loss,
            device,
            scaler,
            amp_enabled,
            epoch,
            args.epochs,
        )

        val_metrics = validate(
            model,
            val_loader,
            bce_loss,
            device,
            amp_enabled,
        )

        scheduler.step(
            val_metrics["dice"]
        )

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch} complete | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"Dice={val_metrics['dice']:.4f} | "
            f"IoU={val_metrics['iou']:.4f} | "
            f"Precision={val_metrics['precision']:.4f} | "
            f"Recall={val_metrics['recall']:.4f} | "
            f"LR={current_lr:.6g} | "
            f"Time={epoch_time:.1f}s"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_bce": train_metrics["bce"],
            "train_dice_loss": train_metrics["dice_loss"],
            "val_loss": val_metrics["loss"],
            "val_bce": val_metrics["bce"],
            "val_dice_loss": val_metrics["dice_loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "learning_rate": current_lr,
            "epoch_time_seconds": epoch_time,
        }

        history.append(record)

        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_dice": best_dice,
                    "config": config,
                },
                best_model_path,
            )

            print(
                f"New best model saved. Dice={best_dice:.4f}"
            )

        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_dice,
            config,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (
        output_dir / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    print("Training finished.")
    print("Best validation Dice:", best_dice)
    print("Best model:", best_model_path)
    print("Last checkpoint:", checkpoint_path)
    print("History:", history_path)


if __name__ == "__main__":
    main()