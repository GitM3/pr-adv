import argparse
import datetime
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

from phenobench import PhenoBench
from phenobench.evaluation.evaluate_semantics import evaluate_semantics

from load import DATA_DIR, IMAGE_SIZE, OUTPUT_CHANNELS, unet_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate PhenoBench semantic predictions and run devkit metrics."
    )
    parser.add_argument(
        "--phenobench_dir",
        default=DATA_DIR,
        type=str,
        help="Path to the PhenoBench root containing train/val.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
        help="Dataset split for evaluation.",
    )
    parser.add_argument(
        "--prediction_dir",
        default=None,
        type=str,
        help="Output directory for predictions (creates semantics/).",
    )
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        help="Path to a saved Keras model (.keras/.h5).",
    )
    parser.add_argument(
        "--weights",
        default=None,
        type=str,
        help="Path to model weights for the local U-Net.",
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch size for prediction.",
    )
    parser.add_argument(
        "--image_size",
        nargs=2,
        default=IMAGE_SIZE,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="Image size for model input.",
    )
    return parser.parse_args()


def load_model_from_args(args):
    image_size = tuple(args.image_size)
    if args.model:
        return tf.keras.models.load_model(args.model)
    if not args.weights:
        raise ValueError("Provide --model or --weights to run validation.")
    model = unet_model(OUTPUT_CHANNELS, image_size=image_size, backbone="mobilenetv3")
    model.load_weights(args.weights)
    return model


def preprocess_image(image, image_size):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, image_size, method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    return image


def _flush_batch(model, batch_images, batch_meta, image_size, semantics_dir):
    images = tf.stack(batch_images, axis=0)
    logits = model(images, training=False)
    pred_mask = tf.argmax(logits, axis=-1)

    for idx, (fname, original_size) in enumerate(batch_meta):
        mask = tf.expand_dims(pred_mask[idx], axis=-1)
        mask = tf.image.resize(
            mask, original_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        mask = tf.squeeze(mask, axis=-1)
        mask = tf.cast(mask, tf.uint8).numpy()
        Image.fromarray(mask).save(semantics_dir / fname)


def write_predictions(model, data, prediction_dir, image_size, batch_size):
    semantics_dir = prediction_dir / "semantics"
    semantics_dir.mkdir(parents=True, exist_ok=True)

    batch_images = []
    batch_meta = []

    for idx in range(len(data)):
        sample = data[idx]
        fname = sample["image_name"]
        image = np.asarray(sample["image"], dtype=np.uint8)
        original_size = (image.shape[0], image.shape[1])

        batch_images.append(preprocess_image(image, image_size))
        batch_meta.append((fname, original_size))

        if len(batch_images) == batch_size:
            _flush_batch(model, batch_images, batch_meta, image_size, semantics_dir)
            batch_images = []
            batch_meta = []

    if batch_images:
        _flush_batch(model, batch_images, batch_meta, image_size, semantics_dir)


def main():
    args = parse_args()
    image_size = tuple(args.image_size)

    prediction_dir = (
        Path(args.prediction_dir)
        if args.prediction_dir
        else Path("output")
        / f"{datetime.datetime.now():%Y_%m_%d_%H_%M}-phenobench-preds"
    )
    prediction_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_from_args(args)
    data = PhenoBench(args.phenobench_dir, split=args.split, target_types=["semantics"])

    write_predictions(model, data, prediction_dir, image_size, args.batch_size)

    metrics = evaluate_semantics(
        {
            "phenobench_dir": Path(args.phenobench_dir),
            "prediction_dir": prediction_dir,
            "split": args.split,
        }
    )

    metrics_path = prediction_dir / "eval_semantics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("Semantic metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
