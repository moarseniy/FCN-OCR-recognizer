
# train.py
from synth_generators.line_generator.dataset import SingleLineDatasetConfig, SingleLineDataset
import argparse
import yaml
from torch.utils.data import DataLoader, random_split

import torch
from model import FullyConvTextRecognizer, transform_back
from loss import ctc_loss, logreg_loss

from tqdm import tqdm
from datetime import datetime
import os
from pathlib import Path

import numpy as np
from PIL import Image

def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    val_loss,
    alphabet,
    config,
    train_losses,
    val_losses,
    checkpoint_dir="checkpoints",
):
    """Сохраняет чекпоинт модели"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_{timestamp}.pth')

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'val_loss': val_loss,
        'alphabet': alphabet,
        'config': config,
        'model_config': {
            'in_channels': config.get('channels', 3),
            'num_classes': len(alphabet) + (1 if config.get('target_mode', 'ctc') == 'ctc' else 0),
            'blank_idx': len(alphabet) if config.get('target_mode', 'ctc') == 'ctc' else None,
            'target_mode': config.get('target_mode', 'ctc'),
        },
        'train_losses': train_losses,
        'val_losses': val_losses,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Сохраняем также последнюю модель
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    print(f"Latest checkpoint saved to {latest_path}")

    return checkpoint_path

def compute_loss(logits, targets, lengths, blank_idx, target_mode):
    if target_mode == "ctc":
        return ctc_loss(logits, targets, lengths, blank_idx)
    return logreg_loss(logits, targets)


def validate(model, loader, device, blank_idx, target_mode, max_batches=50, preview_saver=None):
    """Валидация модели"""
    model.eval()
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for imgs, targets, lengths in tqdm(loader, desc="Validation"):
            if batches >= max_batches:
                break

            if preview_saver is not None:
                preview_saver.save_batch(imgs, targets, lengths)

            imgs = imgs.to(device)
            targets = targets.long().to(device)
            lengths = lengths.long().to(device)

            logits = model(imgs)
            if target_mode == "column":
                logits = transform_back(logits, imgs.shape[3])

            loss = compute_loss(logits, targets, lengths, blank_idx, target_mode)
            total_loss += loss.item()
            batches += 1

            # print(torch.isnan(loss), torch.isinf(loss))

    if batches == 0:
        raise RuntimeError("Validation loader produced no batches")

    return total_loss / batches

def train_one_epoch(model, loader, optimizer, device, blank_idx, target_mode, max_batches=None, preview_saver=None):
    model.train()
    total_loss = 0.0
    batches = 0

    for imgs, targets, lengths in tqdm(loader, desc="Training"):
        if max_batches is not None and batches >= max_batches:
            break

        if preview_saver is not None:
            preview_saver.save_batch(imgs, targets, lengths)

        imgs = imgs.to(device)
        targets = targets.long().to(device)
        lengths = lengths.long().to(device)

        logits = model(imgs)
        if target_mode == "column":
            logits = transform_back(logits, imgs.shape[3])

        loss = compute_loss(logits, targets, lengths, blank_idx, target_mode)

        # print(torch.isnan(loss), torch.isinf(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1

    if batches == 0:
        raise RuntimeError("Training loader produced no batches")

    return total_loss / batches

def tensor_to_pil(image_tensor):
    image = image_tensor.detach().cpu().float().clamp(0.0, 1.0)
    if image.dim() == 4:
        image = image[0]

    if image.shape[0] == 1:
        array = (image[0].numpy() * 255).astype(np.uint8)
        return Image.fromarray(array, mode="L")

    array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")

def decode_target_for_preview(target, length, alphabet, target_mode, space_char):
    if target_mode == "ctc":
        return "".join(alphabet[idx] for idx in target[:length].tolist())

    chars = []
    previous_idx = None
    for idx in target.tolist():
        if idx != previous_idx:
            chars.append(alphabet[idx])
            previous_idx = idx
    return "".join(chars).strip(space_char)

class InputPreviewSaver:
    def __init__(self, output_dir, count, alphabet, target_mode, space_char):
        self.output_path = Path(output_dir)
        self.count = count
        self.alphabet = alphabet
        self.target_mode = target_mode
        self.space_char = space_char
        self.saved = 0
        self.labels_file = None

        if count > 0:
            self.output_path.mkdir(parents=True, exist_ok=True)
            self.labels_file = (self.output_path / "labels.tsv").open("w")
            self.labels_file.write("file\ttext\tlength\n")

    def save_batch(self, images, targets, lengths):
        if self.count <= 0 or self.saved >= self.count:
            return

        for image, target, length in zip(images, targets, lengths):
            if self.saved >= self.count:
                return

            filename = f"{self.saved:04d}.png"
            text = decode_target_for_preview(
                target.long(),
                int(length),
                self.alphabet,
                self.target_mode,
                self.space_char,
            )
            tensor_to_pil(image).save(self.output_path / filename)
            self.labels_file.write(f"{filename}\t{text}\t{int(length)}\n")
            self.labels_file.flush()
            self.saved += 1

    def close(self):
        if self.labels_file is not None:
            self.labels_file.close()
            self.labels_file = None
            print(f"Saved {self.saved} input previews to {self.output_path}")

def plot_losses(train_losses, val_losses, save_path='loss_plot.png'):
    """Строит график лоссов"""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        print(f"Skipping loss plot: matplotlib is not available ({error})")
        return

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Добавляем аннотации для лучшей модели
    best_val_epoch = val_losses.index(min(val_losses)) + 1
    best_val_loss = min(val_losses)
    plt.scatter(best_val_epoch, best_val_loss, color='green', s=100, zorder=5)
    plt.annotate(f'Best: {best_val_loss:.4f}', 
                xy=(best_val_epoch, best_val_loss),
                xytext=(best_val_epoch + 0.5, best_val_loss + 0.1),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Loss plot saved to {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train the FCN OCR recognizer on synthetic lines.")
    parser.add_argument(
        "--config",
        default="synth_generators/line_generator/example_config.yaml",
        help="Path to a SingleLineDataset YAML config.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for checkpoints.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional train batches per epoch limit.")
    parser.add_argument("--max-val-batches", type=int, default=50, help="Validation batches per epoch limit.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest_checkpoint.pth.")
    parser.add_argument("--target-mode", choices=["ctc", "column"], default=None, help="Override config target_mode.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of samples used for validation.")
    parser.add_argument("--preview-samples", type=int, default=0, help="Save N actual input images seen by train and val loops.")
    parser.add_argument("--preview-dir", default="input_previews", help="Directory for saved input previews.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = args.config
    print("START!")
    with open(config, "r") as f:
        config_data = yaml.safe_load(f)
        if args.target_mode:
            config_data["target_mode"] = args.target_mode
        dataset_config = SingleLineDatasetConfig.model_validate_with_paths(config_data, config)

    dataset = SingleLineDataset(dataset_config)
    print(f"Dataset ready! Total samples: {len(dataset)}")

    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")

    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Dataset is too small for the requested validation split")

    split_generator = torch.Generator().manual_seed(dataset_config.seed or 0)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=split_generator,
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    alphabet = dataset_config.alphabet
    target_mode = dataset_config.target_mode
    blank_idx = len(alphabet) if target_mode == "ctc" else None
    print("Alphabet: ", alphabet)
    print("Alphabet length: ", len(alphabet))
    print("Target mode: ", target_mode)
    if blank_idx is not None:
        print("Blank index: ", blank_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    train_preview_saver = None
    val_preview_saver = None
    if args.preview_samples > 0:
        train_preview_saver = InputPreviewSaver(
            Path(args.preview_dir) / "train",
            args.preview_samples,
            alphabet,
            target_mode,
            dataset_config.space_char,
        )
        val_preview_saver = InputPreviewSaver(
            Path(args.preview_dir) / "val",
            args.preview_samples,
            alphabet,
            target_mode,
            dataset_config.space_char,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device ", device)

    model = FullyConvTextRecognizer(
        in_channels=dataset_config.channels,
        num_classes=len(alphabet) + (1 if target_mode == "ctc" else 0)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Списки для хранения истории лоссов
    train_losses = []
    val_losses = []

    # Создаем директорию для чекпоинтов и графиков
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    best_val_loss = float('inf')
    best_train_loss = float('inf')

    # Можно загрузить последний чекпоинт если нужно продолжить обучение
    latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    if args.resume and os.path.exists(latest_checkpoint):
        print("Found latest checkpoint, loading...")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        # Загружаем историю лоссов если она есть
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            best_val_loss = min(val_losses) if val_losses else float('inf')
            best_train_loss = min(train_losses) if train_losses else float('inf')

        print(f"Resuming from epoch {start_epoch}")

    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    for epoch in range(start_epoch, args.epochs):
        # Тренировка
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            blank_idx,
            target_mode,
            args.max_train_batches,
            train_preview_saver,
        )
        train_losses.append(train_loss)

        # Валидация
        val_loss = validate(model, val_loader, device, blank_idx, target_mode, args.max_val_batches, val_preview_saver)
        val_losses.append(val_loss)

        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.8f}")
        print(f"  Val Loss:   {val_loss:.8f}")
        print(f"  Difference: {abs(train_loss - val_loss):.8f}")

        # Проверяем на переобучение
        if train_loss < val_loss * 0.7:
            print(f"  ⚠️  Warning: Possible overfitting detected!")

        # Сохраняем чекпоинт каждые 5 эпох
        if epoch % 5 == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                alphabet,
                config_data,
                train_losses,
                val_losses,
                checkpoint_dir,
            )

            # Строим график каждые 5 эпох
            plot_losses(train_losses, val_losses, 
                       save_path=os.path.join(checkpoint_dir, f'loss_plot_epoch_{epoch}.png'))
            # Сохраняем лучшую модель по валидационному лоссу
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'val_loss': val_loss,
                'alphabet': alphabet,
                'config': config_data,
                'model_config': {
                    'in_channels': dataset_config.channels,
                    'num_classes': len(alphabet) + (1 if target_mode == "ctc" else 0),
                    'blank_idx': blank_idx,
                    'target_mode': target_mode,
                },
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            torch.save(checkpoint, best_checkpoint_path)
            print(f"  ✅ Best model saved (val_loss={val_loss:.8f})")

        # Сохраняем лучшую модель по тренировочному лоссу (для сравнения)
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_checkpoint_path = os.path.join(checkpoint_dir, 'best_train_model.pth')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'val_loss': val_loss,
                'alphabet': alphabet,
                'config': config_data,
                'model_config': {
                    'in_channels': dataset_config.channels,
                    'num_classes': len(alphabet) + (1 if target_mode == "ctc" else 0),
                    'blank_idx': blank_idx,
                    'target_mode': target_mode,
                },
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            torch.save(checkpoint, best_train_checkpoint_path)

        print("-" * 60)

    if train_preview_saver is not None:
        train_preview_saver.close()
    if val_preview_saver is not None:
        val_preview_saver.close()

    # Финальный график
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.8f}")
    print(f"Best training loss:   {best_train_loss:.8f}")
    print("="*60)

    plot_losses(train_losses, val_losses, 
               save_path=os.path.join(checkpoint_dir, 'final_loss_plot.png'))
