# FCN-OCR-recognizer

Полносверточный распознаватель строк и простой синтетический генератор строк
для обучения без ручной разметки.

## Synthetic line generator

Генератор находится в `synth_generators/line_generator`.

Он возвращает элементы в формате, который подходит текущей FCN-идее:

- `image`: тензор `C x H x W`;
- в режиме `ctc`: `target` — padded-тензор длиной `max_text_length`, `length` — реальная длина строки;
- в режиме `column`: `target` — тензор длиной `W`, `length` — ширина изображения.

Есть два режима обучения:

- `target_mode: ctc` — обучение через CTC loss. Последний класс модели — `blank`, поэтому `num_classes = len(alphabet) + 1`.
- `target_mode: column` — старый вариант с классификацией каждого столбца через cross-entropy. Фон и отступы размечаются пробелом, поэтому пробел должен быть в `alphabet`.

В примерах `alphabet` начинается с пробела, а `sample_alphabet` пробел не
содержит. Так пробел есть как класс фона, но синтетические строки не состоят из
случайных пробелов.

Примеры конфигов:

- `synth_generators/line_generator/example_config.yaml` — CTC;
- `synth_generators/line_generator/example_column_config.yaml` — старый column-вариант с пробелом для фона.

Аугментации задаются двумя словарями:

```yaml
augmentation_probabilities:
  rotate: 0.8
  gaussian_blur: 0.35
  gaussian_noise: 0.8
  brightness: 0.3
  contrast: 0.3
  invert: 0.0
augmentations:
  rotate:
    max_degrees: 1.0
    fillcolor: 255
  gaussian_blur:
    radius_min: 0.0
    radius_max: 0.25
  gaussian_noise:
    std_min: 0.0
    std_max: 5.0
  brightness:
    factor_min: 0.85
    factor_max: 1.15
  contrast:
    factor_min: 0.85
    factor_max: 1.2
  invert: {}
```

Вероятность `0.0` выключает преобразование, `1.0` применяет всегда. Для
`column`-режима лучше держать геометрические преобразования мягкими, потому что
target размечен по горизонтальным столбцам.

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

Сохранить примеры именно тех тензоров, которые подаются в train/validation:

```bash
python train.py \
  --config synth_generators/line_generator/example_config.yaml \
  --preview-samples 16 \
  --preview-dir input_previews
```

Картинки будут сохранены в `input_previews/train` и `input_previews/val`.
Сохранение происходит внутри train/validation loop прямо перед `model(imgs)`,
поэтому это ровно те изображения, которые подаются на вход сети, уже с
применёнными аугментациями. Рядом создаётся `labels.tsv` с именем файла,
текстом и длиной target. Размер validation-части задаётся через
`--val-fraction`, по умолчанию `0.1`.

Запустить старый column-вариант:

```bash
python train.py --config synth_generators/line_generator/example_column_config.yaml
```

Продолжать чекпоинт можно только в том же режиме, в котором он был создан. Для
продолжения используйте:

```bash
python train.py --resume --config synth_generators/line_generator/example_config.yaml
```

Инференс на синтетическом примере:

```bash
python inference.py --checkpoint checkpoints/best_model.pth --sample-index 0
```
