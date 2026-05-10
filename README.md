# FCN-OCR-recognizer

Полносверточный распознаватель строк и простой синтетический генератор строк
для обучения без ручной разметки.

## Synthetic line generator

Генератор находится в `synth_generators/line_generator`.

Он возвращает элементы в формате, который подходит текущей FCN-идее:

- `image`: тензор `C x H x W`;
- `target`: padded-тензор длиной `max_text_length` с индексами символов;
- `length`: реальная длина строки.

Модель обучается через CTC loss. Последний класс модели — `blank`, поэтому
`num_classes = len(alphabet) + 1`. На инференсе повторы схлопываются, а blank
удаляется.

Пример конфига: `synth_generators/line_generator/example_config.yaml`.

Сохранить один пример изображения:

```bash
python -m synth_generators.line_generator.preview \
  --config synth_generators/line_generator/example_config.yaml \
  --output synthetic_line_preview.png
```

Запустить обучение на синтетике:

```bash
python train.py --config synth_generators/line_generator/example_config.yaml
```

Продолжать старые чекпоинты после перехода на CTC не стоит: у модели изменился
последний слой. Для продолжения новой CTC-тренировки используйте:

```bash
python train.py --resume --config synth_generators/line_generator/example_config.yaml
```

Инференс на синтетическом примере:

```bash
python inference.py --checkpoint checkpoints/best_model.pth --sample-index 0
```
