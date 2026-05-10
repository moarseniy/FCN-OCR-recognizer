
# train.py
from synth_generators.line_generator.dataset import SingleLineDatasetConfig, SingleLineDataset
import yaml
from torch.utils.data import DataLoader, random_split

import torch
from model import FullyConvTextRecognizer, transform_back
from loss import logreg_loss, simple_logreg_loss

from functools import partial
# from utils import TextEncoder, collate_fn

from tqdm import tqdm
from itertools import islice
from datetime import datetime
import os
import matplotlib.pyplot as plt

def save_checkpoint(model, optimizer, epoch, loss, alphabet, config, checkpoint_dir='checkpoints'):
    """Сохраняет чекпоинт модели"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_{timestamp}.pth')

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'alphabet': alphabet,
        'config': config,
        'model_config': {
            'in_channels': 3,
            'num_classes': len(alphabet)
        }
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Сохраняем также последнюю модель
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    print(f"Latest checkpoint saved to {latest_path}")

    return checkpoint_path

def validate(model, loader, device):
    """Валидация модели"""
    model.eval()
    total_loss = 0.0

    iterations = [20, 40]
    cur_iter = 0
    with torch.no_grad():
        for imgs, targets, lengths in tqdm(loader, desc="Validation"):
            cur_iter += 1
            if cur_iter == iterations[1]:
                break
            if cur_iter < iterations[0]:
                continue

            imgs = imgs.to(device)
            targets = targets.long().to(device)

            logits = model(imgs)
            logits = transform_back(logits, imgs.shape[3])

            loss = logreg_loss(logits, targets)
            total_loss += loss.item()

            # print(torch.isnan(loss), torch.isinf(loss))

    return total_loss / 20

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    iterations = 20
    cur_iter = 0
    for imgs, targets, lengths in tqdm(loader, desc="Training"):
        cur_iter += 1
        if cur_iter == iterations:
            break

        imgs = imgs.to(device)
        targets = targets.long().to(device)

        logits = model(imgs)
        logits = transform_back(logits, imgs.shape[3])

        loss = logreg_loss(logits, targets)

        # print(torch.isnan(loss), torch.isinf(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / iterations

def plot_losses(train_losses, val_losses, save_path='loss_plot.png'):
    """Строит график лоссов"""
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

if __name__ == "__main__":
    config = "/home/mokin-a/proj/test_dataset_config_50_samples.yaml"
    print("START!")
    with open(config, "r") as f:
        config_data = yaml.safe_load(f)
        dataset_config = SingleLineDatasetConfig.model_validate(config_data)

    dataset = SingleLineDataset(dataset_config)
    print(f"Dataset ready! Total samples: {len(dataset)}")

    alphabet = config_data['alphabet']
    print("Alphabet: ", alphabet)
    print("Alphabet length: ", len(alphabet))

    dataloader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=False,  # шаффлим тренировочные данные
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device ", device)

    model = FullyConvTextRecognizer(
        in_channels=3,
        num_classes=len(alphabet)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Списки для хранения истории лоссов
    train_losses = []
    val_losses = []

    # Создаем директорию для чекпоинтов и графиков
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    best_val_loss = float('inf')
    best_train_loss = float('inf')

    # Можно загрузить последний чекпоинт если нужно продолжить обучение
    latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    if os.path.exists(latest_checkpoint):
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

    for epoch in range(start_epoch, 50):
        # Тренировка
        train_loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            device
        )
        train_losses.append(train_loss)

        # Валидация
        val_loss = validate(model, dataloader, device)
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
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'val_loss': val_loss,
                'alphabet': alphabet,
                'config': config_data,
                'model_config': {
                    'in_channels': 3,
                    'num_classes': len(alphabet)
                },
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            save_checkpoint(model, optimizer, epoch, train_loss, alphabet, config_data, checkpoint_dir)

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
                    'in_channels': 3,
                    'num_classes': len(alphabet)
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
                    'in_channels': 3,
                    'num_classes': len(alphabet)
                },
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            torch.save(checkpoint, best_train_checkpoint_path)

        print("-" * 60)

    # Финальный график
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.8f}")
    print(f"Best training loss:   {best_train_loss:.8f}")
    print("="*60)

    plot_losses(train_losses, val_losses, 
               save_path=os.path.join(checkpoint_dir, 'final_loss_plot.png'))
