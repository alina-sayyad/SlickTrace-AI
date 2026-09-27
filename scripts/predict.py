import argparse
import os

import numpy as np
import pandas as pd
import rasterio
import torch
import segmentation_models_pytorch as smp


IN_CHANNELS = 2
NUM_CLASSES = 1


def pad_to_multiple_of_16(array):
    if array.ndim == 3:
        _, h, w = array.shape
        ph = (16 - h % 16) % 16
        pw = (16 - w % 16) % 16

        return np.pad(
            array,
            ((0, 0), (0, ph), (0, pw)),
            mode="constant",
            constant_values=0,
        )

    h, w = array.shape
    ph = (16 - h % 16) % 16
    pw = (16 - w % 16) % 16

    return np.pad(
        array,
        ((0, ph), (0, pw)),
        mode="constant",
        constant_values=0,
    )


def load_image(path):
    with rasterio.open(path) as src:
        image = src.read().astype(np.float32)

    if image.shape[0] != IN_CHANNELS:
        raise ValueError(
            f"Expected {IN_CHANNELS} channels, got {image.shape[0]}: {path}"
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

    return pad_to_multiple_of_16(normalized)


def load_mask(path):
    with rasterio.open(path) as src:
        mask = src.read(1).astype(np.float32)

    return pad_to_multiple_of_16(
        (mask > 0).astype(np.float32)
    )


def build_model():
    model = smp.DeepLabV3Plus(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=2,
        classes=1,
    )

    return model


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)

    return checkpoint


def calculate_metrics(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(
        prediction,
        target,
    ).sum()

    union = np.logical_or(
        prediction,
        target,
    ).sum()

    predicted_positive = prediction.sum()
    actual_positive = target.sum()

    dice_denominator = (
        predicted_positive + actual_positive
    )

    if dice_denominator == 0:
        dice = 1.0
    else:
        dice = (
            2.0 * intersection
        ) / dice_denominator

    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union

    if predicted_positive == 0:
        precision = (
            1.0
            if actual_positive == 0
            else 0.0
        )
    else:
        precision = (
            intersection
            / predicted_positive
        )

    if actual_positive == 0:
        recall = (
            1.0
            if predicted_positive == 0
            else 0.0
        )
    else:
        recall = (
            intersection
            / actual_positive
        )

    return dice, iou, precision, recall


def parse_thresholds(text):
    thresholds = []

    for value in text.split(","):
        value = value.strip()

        if not value:
            continue

        threshold = float(value)

        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(
                f"Threshold must be between 0 and 1: {threshold}"
            )

        thresholds.append(threshold)

    thresholds = sorted(set(thresholds))

    if not thresholds:
        raise ValueError(
            "No valid thresholds were provided."
        )

    return thresholds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--threshold-sweep",
        action="store_true",
        help="Evaluate multiple thresholds without saving every threshold mask.",
    )

    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80",
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("SLICKTRACE AI SEGMENTATION INFERENCE")
    print("=" * 70)

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    df = pd.read_csv(args.csv)

    if args.max_samples is not None:
        df = df.iloc[
            :args.max_samples
        ].copy()

    print(
        "Samples:",
        len(df),
    )

    print(
        "Checkpoint:",
        args.checkpoint,
    )

    print(
        "Threshold:",
        args.threshold,
    )

    model = build_model().to(device)

    load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    model.eval()

    if args.threshold_sweep:

        thresholds = parse_thresholds(
            args.thresholds
        )

        print()
        print(
            "THRESHOLD SWEEP ENABLED"
        )

        print(
            "Thresholds:",
            ", ".join(
                f"{t:.2f}"
                for t in thresholds
            ),
        )

        metric_storage = {}

        for threshold in thresholds:
            metric_storage[threshold] = {
                "dice": [],
                "iou": [],
                "precision": [],
                "recall": [],
            }

        with torch.no_grad():

            for count, (_, row) in enumerate(
                df.iterrows(),
                start=1,
            ):

                image = load_image(
                    row["image_path"]
                )

                image_tensor = (
                    torch.from_numpy(image)
                    .unsqueeze(0)
                    .to(device)
                )

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=(
                        device.type == "cuda"
                    ),
                ):
                    logits = model(
                        image_tensor
                    )

                probability = (
                    torch.sigmoid(logits)
                    [0, 0]
                    .float()
                    .cpu()
                    .numpy()
                )

                target = None

                if (
                    "mask_path" in row.index
                    and isinstance(
                        row["mask_path"],
                        str,
                    )
                    and os.path.exists(
                        row["mask_path"]
                    )
                ):
                    target = load_mask(
                        row["mask_path"]
                    )

                if target is not None:

                    for threshold in thresholds:

                        prediction = (
                            probability
                            >= threshold
                        ).astype(
                            np.uint8
                        )

                        (
                            dice,
                            iou,
                            precision,
                            recall,
                        ) = calculate_metrics(
                            prediction,
                            target,
                        )

                        metric_storage[
                            threshold
                        ]["dice"].append(
                            dice
                        )

                        metric_storage[
                            threshold
                        ]["iou"].append(
                            iou
                        )

                        metric_storage[
                            threshold
                        ]["precision"].append(
                            precision
                        )

                        metric_storage[
                            threshold
                        ]["recall"].append(
                            recall
                        )

                if (
                    count % 25 == 0
                    or count == len(df)
                ):
                    print(
                        f"Processed "
                        f"{count}/{len(df)}"
                    )

        sweep_results = []

        for threshold in thresholds:

            row = {
                "threshold": threshold,
                "dice": np.mean(
                    metric_storage[
                        threshold
                    ]["dice"]
                ),
                "iou": np.mean(
                    metric_storage[
                        threshold
                    ]["iou"]
                ),
                "precision": np.mean(
                    metric_storage[
                        threshold
                    ]["precision"]
                ),
                "recall": np.mean(
                    metric_storage[
                        threshold
                    ]["recall"]
                ),
            }

            sweep_results.append(row)

        sweep_df = pd.DataFrame(
            sweep_results
        )

        sweep_path = os.path.join(
            args.output_dir,
            "threshold_sweep.csv",
        )

        sweep_df.to_csv(
            sweep_path,
            index=False,
        )

        best_dice_row = sweep_df.loc[
            sweep_df["dice"].idxmax()
        ]

        best_iou_row = sweep_df.loc[
            sweep_df["iou"].idxmax()
        ]

        print()
        print("=" * 70)
        print("THRESHOLD SWEEP RESULTS")
        print("=" * 70)

        print(
            sweep_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        print()
        print(
            "BEST DICE THRESHOLD:",
            f"{best_dice_row['threshold']:.2f}",
        )

        print(
            "Best Dice:",
            f"{best_dice_row['dice']:.4f}",
        )

        print(
            "Precision:",
            f"{best_dice_row['precision']:.4f}",
        )

        print(
            "Recall:",
            f"{best_dice_row['recall']:.4f}",
        )

        print()
        print(
            "BEST IOU THRESHOLD:",
            f"{best_iou_row['threshold']:.2f}",
        )

        print(
            "Best IoU:",
            f"{best_iou_row['iou']:.4f}",
        )

        print(
            "Precision:",
            f"{best_iou_row['precision']:.4f}",
        )

        print(
            "Recall:",
            f"{best_iou_row['recall']:.4f}",
        )

        print()
        print(
            "Threshold results:",
            sweep_path,
        )

        print("=" * 70)

        return

    # ------------------------------------------------------------------
    # NORMAL SINGLE-THRESHOLD INFERENCE
    # ------------------------------------------------------------------

    results = []

    with torch.no_grad():

        for index, row in df.iterrows():

            image = load_image(
                row["image_path"]
            )

            image_tensor = (
                torch.from_numpy(image)
                .unsqueeze(0)
                .to(device)
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(
                    device.type == "cuda"
                ),
            ):
                logits = model(
                    image_tensor
                )

            probability = (
                torch.sigmoid(logits)
                [0, 0]
                .float()
                .cpu()
                .numpy()
            )

            prediction = (
                probability
                >= args.threshold
            ).astype(
                np.uint8
            )

            prediction_path = os.path.join(
                args.output_dir,
                f"prediction_{index:04d}.tif",
            )

            with rasterio.open(
                row["image_path"]
            ) as src:

                profile = src.profile.copy()

                profile.update(
                    count=1,
                    dtype="uint8",
                    compress="lzw",
                )

                with rasterio.open(
                    prediction_path,
                    "w",
                    **profile,
                ) as dst:

                    dst.write(
                        prediction,
                        1,
                    )

            result = {
                "index": index,
                "image_path": row["image_path"],
                "mask_path": row["mask_path"],
                "prediction_path": prediction_path,
                "threshold": args.threshold,
                "predicted_pixels": int(
                    prediction.sum()
                ),
            }

            if (
                "mask_path" in row.index
                and isinstance(
                    row["mask_path"],
                    str,
                )
                and os.path.exists(
                    row["mask_path"]
                )
            ):

                target = load_mask(
                    row["mask_path"]
                )

                (
                    dice,
                    iou,
                    precision,
                    recall,
                ) = calculate_metrics(
                    prediction,
                    target,
                )

                result.update(
                    {
                        "dice": dice,
                        "iou": iou,
                        "precision": precision,
                        "recall": recall,
                    }
                )

            results.append(result)

            if (
                len(results) % 25 == 0
                or len(results) == len(df)
            ):
                print(
                    f"Processed "
                    f"{len(results)}/{len(df)}"
                )

    results_df = pd.DataFrame(
        results
    )

    results_path = os.path.join(
        args.output_dir,
        "prediction_results.csv",
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    metric_columns = [
        "dice",
        "iou",
        "precision",
        "recall",
    ]

    available_metrics = [
        c
        for c in metric_columns
        if c in results_df.columns
    ]

    if available_metrics:

        print()
        print("AVERAGE METRICS")
        print("-" * 40)

        for metric in available_metrics:

            print(
                f"{metric.capitalize():10s}: "
                f"{results_df[metric].mean():.4f}"
            )

    print()
    print(
        "Results:",
        results_path,
    )

    print(
        "Predictions:",
        args.output_dir,
    )

    print("=" * 70)


if __name__ == "__main__":
    main()