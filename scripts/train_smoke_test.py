from pathlib import Path
import time

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


CSV_PATH = Path(r"C:\SlickTrace\outputs\train.csv")
CHECKPOINT_PATH = Path(r"C:\SlickTrace\models\smoke_test_checkpoint.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(path):
    with rasterio.open(path) as src:
        image = src.read().astype(np.float32)

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


def dice_loss(logits, target):
    probability = torch.sigmoid(logits)

    smooth = 1.0

    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    return 1.0 - dice.mean()


def main():
    print("SlickTrace AI training smoke test")
    print("Device:", DEVICE)

    df = pd.read_csv(CSV_PATH)

    rows = df.sample(
        n=min(10, len(df)),
        random_state=42,
    ).reset_index(drop=True)

    model = smp.DeepLabV3Plus(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=2,
        classes=1,
        activation=None,
    )

    model = model.to(DEVICE)

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    bce = nn.BCEWithLogitsLoss()

    model.apply(
        lambda module: (
            module.eval()
            if isinstance(module, nn.BatchNorm2d)
            else None
        )
    )

    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    print("Training samples:", len(rows))
    print("Resolution: 2048x2048")
    print("Batch size: 1")
    print("Steps: 10")

    model.train()

    model.apply(
        lambda module: (
            module.eval()
            if isinstance(module, nn.BatchNorm2d)
            else None
        )
    )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()

    for step in range(10):
        row = rows.iloc[step % len(rows)]

        image = load_image(row["image_path"])
        mask = load_mask(row["mask_path"])

        image_tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        start = time.perf_counter()

        logits = model(image_tensor)

        loss_bce = bce(logits, mask_tensor)
        loss_dice = dice_loss(logits, mask_tensor)
        loss = loss_bce + loss_dice

        loss.backward()

        optimizer.step()

        elapsed = time.perf_counter() - start

        print(
            f"Step {step + 1}/10 | "
            f"loss={loss.item():.4f} | "
            f"bce={loss_bce.item():.4f} | "
            f"dice={loss_dice.item():.4f} | "
            f"time={elapsed:.2f}s"
        )

    total_time = time.perf_counter() - total_start

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "steps": 10,
        },
        CHECKPOINT_PATH,
    )

    print()
    print("Training smoke test completed.")
    print("Total time:", round(total_time, 2), "seconds")
    print("Checkpoint:", CHECKPOINT_PATH)


if __name__ == "__main__":
    main()