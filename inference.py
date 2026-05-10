# inference.py
import torch
import numpy as np
from PIL import Image
import cv2
from model import FullyConvTextRecognizer, transform_back, decode_greedy_batch_tensor
import argparse
import os

from synth_generators.line_generator.dataset import SingleLineDatasetConfig, SingleLineDataset
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

def get_batch_element(dataloader, k):
    for i, batch in tqdm(enumerate(dataloader)):
        if i == k:
            return batch

def save_image_safe(image_tensor, filepath):
    # Приводим к float и нормализуем в [0, 1]
    img = image_tensor.float()
    img = (img - img.min()) / (img.max() - img.min())
    # Сохраняем
    save_image(img, filepath)

class TextRecognizer:
    def __init__(self, checkpoint_path, device=None):
        """
        Инициализация распознавателя текста

        Args:
            checkpoint_path: путь к файлу чекпоинта
            device: устройство для инференса ('cuda' или 'cpu')
        """

        config = "/home/mokin-a/proj/test_dataset_config_50_samples.yaml"
        print("START!")
        with open(config, "r") as f:
            config_data = yaml.safe_load(f)
            dataset_config = SingleLineDatasetConfig.model_validate(config_data)

        dataset = SingleLineDataset(dataset_config)
        print("Dataset ready!")

        # alphabet = "0123456789абвгдеёжзийклмнопрстуфхцчьыъэюя"
        # alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
        # encoder = TextEncoder(alphabet)
        self.dataloader = DataLoader(
            dataset,
            batch_size=1,
            # shuffle=True,
            # collate_fn=partial(collate_fn, encoder=encoder)
        )

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Загружаем чекпоинт
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Получаем алфавит и параметры модели
        self.alphabet = self.checkpoint['alphabet']
        self.idx_to_char = {i: char for i, char in enumerate(self.alphabet)}

        model_config = self.checkpoint['model_config']

        # Создаем и загружаем модель
        self.model = FullyConvTextRecognizer(
            in_channels=model_config['in_channels'],
            num_classes=model_config['num_classes']
        ).to(self.device)

        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded from epoch {self.checkpoint['epoch']}, loss: {self.checkpoint['loss']:.8f}")
        print(f"Alphabet: {self.alphabet}")
        print(f"Alphabet size: {len(self.alphabet)}")

    def get_alphabet(self):
        return self.alphabet

    def preprocess_image(self, image_path):
        """
        Предобработка изображения

        Args:
            image_path: путь к изображению

        Returns:
            torch.Tensor: обработанное изображение формы (1, 3, H, W)
        """
        # Загружаем изображение
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image from {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path  # предполагаем что это numpy array

        # Конвертируем в float и нормализуем
        image = image.astype(np.float32) / 255.0

        # Преобразуем в тензор и меняем размерности (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)

        # Добавляем batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor.to(self.device)

    def decode_predictions(self, logits):
        """
        Декодирует предсказания модели в текст

        Args:
            logits: выход модели формы (1, num_classes, width)

        Returns:
            str: распознанный текст
            list: сырые предсказания (индексы)
        """
        # Получаем предсказанные классы
        pred_ids = logits.argmax(dim=1)  # (1, width)

        print("BBBB", pred_ids)
        # Применяем greedy decoding для удаления повторяющихся символов
        collapsed, lengths = decode_greedy_batch_tensor(pred_ids)

        # Конвертируем индексы в символы
        text = []
        raw_indices = pred_ids[0].cpu().numpy().tolist()

        for idx in collapsed[0][:lengths[0]].cpu().numpy():
            if idx < len(self.alphabet):
                text.append(self.idx_to_char[idx])

        return ''.join(text), raw_indices

    @torch.no_grad()
    def recognize_tensor(self, image_tensor):
        """
        Распознает текст из тензора (как в DataLoader)

        Args:
            image_tensor: тензор формы (B, C, H, W) или (C, H, W)

        Returns:
            list: список распознанных текстов
            list: сырые предсказания
        """
        # Проверяем размерность
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # добавляем batch dimension

        # Убеждаемся что тензор на правильном устройстве и правильного типа
        image_tensor = image_tensor.to(self.device)

        # Проверяем тип данных
        if image_tensor.dtype != torch.float32:
            print(f"Warning: Converting tensor from {image_tensor.dtype} to float32")
            image_tensor = image_tensor.float()

        # Проверяем диапазон значений
        if image_tensor.max() > 1.0:
            print(f"Warning: Tensor values > 1.0 detected (max={image_tensor.max():.2f}), normalizing to [0,1]")
            image_tensor = image_tensor / 255.0

        # Инференс
        logits = self.model(image_tensor)

        # Применяем transform_back если нужно
        # logits = transform_back(logits, image_tensor.shape[3])

        # print(logits)

        # Декодирование
        texts, raw_indices = self.decode_predictions(logits)

        return texts, raw_indices

    def recognize(self, image_path, return_raw=False):
        """
        Распознает текст на изображении

        Args:
            image_path: путь к изображению или numpy array
            return_raw: возвращать ли сырые предсказания

        Returns:
            str: распознанный текст
            list (опционально): сырые предсказания
        """
        if image_path:
            image_tensor = self.preprocess_image(image_path)
        else:
            print("AAAAAAAAAAAAAA")
            image_tensor, tgts, lngths = get_batch_element(self.dataloader, 3)
            image_tensor = image_tensor.to(self.device)
            print("TARGETS:", tgts.long().tolist()[0])
            tgts = [self.alphabet[i] for i in tgts.long().tolist()[0]]
            print(tgts)
            save_image_safe(image_tensor[0], "temp.png")
            return self.recognize_tensor(image_tensor)

        # print(image_tensor.size(), type(image_tensor))
        # save_image_safe(image_tensor[0], "temp.png")

        # Инференс
        with torch.no_grad():
            logits = self.model(image_tensor)
            # Применяем transform_back если нужно
            logits = transform_back(logits, image_tensor.shape[3])

        # Декодирование
        text, raw_indices = self.decode_predictions(logits)

        if return_raw:
            return text, raw_indices

        return text

    def recognize_batch(self, images):
        """
        Распознает текст на батче изображений

        Args:
            images: список путей к изображениям или список numpy array

        Returns:
            list: список распознанных текстов
        """
        texts = []
        for image in images:
            text = self.recognize(image)
            texts.append(text)
        return texts

def main():
    parser = argparse.ArgumentParser(description='Text Recognition Inference')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to checkpoint file')
    parser.add_argument('--image', type=str, required=False,
                       help='Path to image file for recognition')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Проверяем существование файлов
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} not found!")
        return

    # if not os.path.exists(args.image):
    #     print(f"Error: Image file {args.image} not found!")
    #     return

    # Инициализируем распознаватель
    recognizer = TextRecognizer(args.checkpoint, args.device)

    image = None
    if not args.image:
        image = args.image

    # Распознаем текст
    text, raw_indices = recognizer.recognize(image, return_raw=True)

    print(f"\n{'='*50}")
    # print(f"Image: {args.image}")
    print(f"Recognized text: '{text}'")
    print(f"{'='*50}")

    # Опционально: показываем сырые предсказания
    print(f"\nRaw predictions (indices): {raw_indices}")

    raw_preds = [recognizer.get_alphabet()[i] for i in raw_indices]
    print(raw_preds)

if __name__ == "__main__":
    # Пример использования без аргументов командной строки
    # recognizer = TextRecognizer('checkpoints/best_model.pth')
    # text = recognizer.recognize('path/to/image.jpg')
    # print(f"Recognized text: {text}")

    main()
