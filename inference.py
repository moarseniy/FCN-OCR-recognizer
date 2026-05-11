from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import yaml

from model import FullyConvTextRecognizer, decode_greedy_batch_tensor
from synth_generators.line_generator.dataset import SingleLineDataset, SingleLineDatasetConfig


DEFAULT_CONFIG = "synth_generators/line_generator/configs/example.yaml"


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    image = image_tensor.detach().cpu().float().clamp(0.0, 1.0)
    if image.dim() == 4:
        image = image[0]

    if image.shape[0] == 1:
        array = (image[0].numpy() * 255).astype(np.uint8)
        return Image.fromarray(array, mode="L")

    array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


class TextRecognizer:
    def __init__(self, checkpoint_path: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.alphabet = self.checkpoint["alphabet"]
        self.idx_to_char = {idx: char for idx, char in enumerate(self.alphabet)}

        model_config = self.checkpoint.get("model_config", {})
        checkpoint_config = self.checkpoint.get("config", {})
        self.in_channels = int(model_config.get("in_channels", 3))
        self.num_classes = int(model_config.get("num_classes", len(self.alphabet) + 1))
        self.blank_idx = int(model_config.get("blank_idx", len(self.alphabet)))
        self.space_char = checkpoint_config.get("space_char", " ")
        self.space_idx = self.alphabet.index(self.space_char) if self.space_char in self.alphabet else None
        self.image_height = int(checkpoint_config.get("image_height", 48))

        self.model = FullyConvTextRecognizer(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
        ).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

        epoch = self.checkpoint.get("epoch", "?")
        loss = self.checkpoint.get("loss")
        loss_text = f", loss: {loss:.8f}" if isinstance(loss, float) else ""
        print(f"Using device: {self.device}")
        print(f"Model loaded from epoch {epoch}{loss_text}")
        print(f"Alphabet size: {len(self.alphabet)}")
        print(f"Blank index: {self.blank_idx}")

    def preprocess_image(self, image_path: str | Path) -> torch.Tensor:
        image = Image.open(image_path)
        image = image.convert("RGB" if self.in_channels == 3 else "L")

        if image.height != self.image_height:
            new_width = max(1, round(image.width * self.image_height / image.height))
            image = image.resize((new_width, self.image_height), Image.Resampling.BICUBIC)

        array = np.asarray(image, dtype=np.float32) / 255.0
        if self.in_channels == 1:
            tensor = torch.from_numpy(array).unsqueeze(0)
        else:
            tensor = torch.from_numpy(array).permute(2, 0, 1)

        return tensor.unsqueeze(0).to(self.device)

    def decode_predictions(self, logits: torch.Tensor) -> tuple[str, list[int]]:
        pred_ids = logits.argmax(dim=1)
        collapsed, lengths = decode_greedy_batch_tensor(pred_ids)

        chars: list[str] = []
        for idx in collapsed[0, : lengths[0]].detach().cpu().tolist():
            if idx == self.blank_idx:
                continue
            if idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])

        text = "".join(chars)
        return text, pred_ids[0].detach().cpu().tolist()

    @torch.no_grad()
    def recognize_tensor(self, image_tensor: torch.Tensor) -> tuple[str, list[int]]:
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device).float()
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

        logits = self.model(image_tensor)
        return self.decode_predictions(logits)

    def recognize(self, image_path: str | Path) -> tuple[str, list[int]]:
        return self.recognize_tensor(self.preprocess_image(image_path))


def load_dataset_config(
    config_path: str | Path | None,
    checkpoint_config: dict | None = None,
) -> SingleLineDatasetConfig:
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset config not found: {path}")
        with path.open("r") as file:
            return SingleLineDatasetConfig.model_validate_with_paths(yaml.safe_load(file), path)

    if checkpoint_config:
        return SingleLineDatasetConfig.model_validate(checkpoint_config)

    default_path = Path(DEFAULT_CONFIG)
    with default_path.open("r") as file:
        return SingleLineDatasetConfig.model_validate_with_paths(yaml.safe_load(file), default_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FCN OCR inference.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file.")
    parser.add_argument("--image", help="Path to an image file for recognition.")
    parser.add_argument(
        "--config",
        default=None,
        help="Dataset config for --sample-index mode. Defaults to checkpoint config, then example config.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        help="Recognize a generated synthetic sample instead of --image.",
    )
    parser.add_argument(
        "--save-sample",
        default="temp.png",
        help="Where to save the generated sample image in --sample-index mode.",
    )
    parser.add_argument("--device", default=None, help="Device to use: cuda or cpu.")
    parser.add_argument("--show-raw", action="store_true", help="Print raw timestep predictions.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    recognizer = TextRecognizer(str(checkpoint_path), args.device)

    if args.image:
        text, raw_indices = recognizer.recognize(args.image)
        print(f"Image: {args.image}")
    else:
        sample_index = args.sample_index if args.sample_index is not None else 0
        dataset_config = load_dataset_config(args.config, recognizer.checkpoint.get("config"))
        dataset_config = dataset_config.model_copy(
            update={
                "alphabet": recognizer.alphabet,
                "channels": recognizer.in_channels,
                "image_height": recognizer.image_height,
            }
        )
        dataset = SingleLineDataset(dataset_config)

        rng = random.Random((dataset_config.seed or 0) + sample_index)
        sample = dataset.generate_sample(rng)
        tensor_to_pil(sample.image).save(args.save_sample)

        text, raw_indices = recognizer.recognize_tensor(sample.image)
        print(f"Synthetic sample index: {sample_index}")
        print(f"Saved sample image: {args.save_sample}")
        print(f"Expected text: '{sample.text}'")

    print(f"Recognized text: '{text}'")

    if args.show_raw:
        raw_chars = [
            "<blank>" if idx == recognizer.blank_idx else recognizer.idx_to_char.get(idx, "?")
            for idx in raw_indices
        ]
        print(f"Raw indices: {raw_indices}")
        print(f"Raw chars: {raw_chars}")


if __name__ == "__main__":
    main()
