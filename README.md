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

Если нужно рисовать текст поверх реальных/синтетических фонов, укажите папку:

```yaml
background_dir: /path/to/backgrounds
background_extensions:
  - .png
  - .jpg
  - .jpeg
  - .bmp
  - .webp
```

Генератор рекурсивно берёт изображения из `background_dir`, делает случайный
crop/resize под размер строки и рисует текст поверх. Если `background_dir:
null`, используется однотонный фон из поля `background`. Относительный путь
считается относительно YAML-конфига, а не относительно текущей директории
запуска.

Аугментации задаются двумя словарями:

```yaml
augmentation_probabilities:
  cycle_shift: 0.05
  strong_blur: 0.08
  motion_blur: 0.08
  scale: 0.15
  darkening: 0.2
  noise: 0.75
  projective: 0.12
  rotate: 0.8
  crop_x: 0.08
  crop_y: 0.05
  morphology: 0.08
  unsharp_mask: 0.12
  gaussian_blur: 0.0
  gaussian_noise: 0.0
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

Доступные OCR-аугментации: `cycle_shift`, `strong_blur`, `motion_blur`,
`scale`, `darkening`, `noise`, `projective`, `rotate`, `crop_x`, `crop_y`,
`morphology`, `unsharp_mask`. Старые `gaussian_blur` и `gaussian_noise`
оставлены как совместимые алиасы.

Сохранить один пример изображения:

```bash
python -m synth_generators.line_generator.preview \
  --config synth_generators/line_generator/example_config.yaml \
  --output synthetic_line_preview.png
```

Сохранить чистый датасет на диск в виде `uint8` torch-чанков:

```bash
python -m synth_generators.line_generator.materialize \
  --config synth_generators/line_generator/example_config.yaml \
  --output-dir data/line_chunks \
  --chunk-size 1024 \
  --overwrite
```

В каждом `chunk_*.pt` лежат `images` (`uint8`, `N x C x H x W`), `targets`,
`lengths` и исходные `texts`; рядом пишется `manifest.pt`. Offline-генерация
по умолчанию не применяет аугментации, но сохраняет их настройки в manifest,
чтобы обучение могло применять их на GPU. Если всё же нужно запечь CPU-
аугментации прямо в чанки, добавьте `--with-augmentations`.

Запустить обучение на синтетике:

```bash
python train.py --config synth_generators/line_generator/example_config.yaml
```

Запустить обучение из сохранённых чанков:

```bash
python train.py --chunks-dir data/line_chunks
```

По умолчанию `train.py` применяет настроенные аугментации на устройстве
обучения (`cuda`, если доступна). Отключить это можно флагом:

```bash
python train.py --chunks-dir data/line_chunks --no-gpu-augmentations
```

Обучение пишет компактный лог по эпохам в консоль и TSV-файл:

```text
checkpoints/training_log.tsv
```

Разбиение на батчи настраивается явно:

```bash
python train.py \
  --config synth_generators/line_generator/example_config.yaml \
  --batch-size 128 \
  --num-workers 4 \
  --drop-last \
  --log-every 10
```

При старте печатаются размеры train/validation split, batch size, количество
батчей и лимиты `--max-train-batches` / `--max-val-batches`, если они заданы.
По умолчанию `--log-every 1`, то есть loss печатается на каждом batch; значение
`0` отключает batch-логи.

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
