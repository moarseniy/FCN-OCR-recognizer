# FCN-OCR-recognizer

Полносверточный распознаватель строк и простой синтетический генератор строк
для обучения без ручной разметки.

## Synthetic line generator

Генератор находится в `synth_generators/line_generator`.

Он возвращает элементы в формате, который подходит текущей FCN-идее:

- `image`: тензор `C x H x W`;
- `target` — padded-тензор длиной `max_text_length`;
- `length` — реальная длина строки.

Обучение сейчас использует один основной режим: CTC loss. Последний класс
модели — `blank`, поэтому `num_classes = len(alphabet) + 1`.

В примерах `alphabet` начинается с пробела. Пробел является обычным классом,
но генератор нормализует строки: пробелы в начале и конце убираются, несколько
пробелов подряд схлопываются в один.

Примеры конфигов:

- `synth_generators/line_generator/configs/example.yaml` — генерация;
- `configs/example_train.yaml` — обучение.

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

Аугментации задаются в training-конфиге двумя словарями и применяются единым
torch/GPU-пайплайном во время обучения:

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

Вероятность `0.0` выключает преобразование, `1.0` применяет всегда.

Доступные OCR-аугментации: `cycle_shift`, `strong_blur`, `motion_blur`,
`scale`, `darkening`, `noise`, `projective`, `rotate`, `crop_x`, `crop_y`,
`morphology`, `unsharp_mask`. Старые `gaussian_blur` и `gaussian_noise`
оставлены как совместимые алиасы.

Сохранить один пример изображения:

```bash
python -m synth_generators.line_generator.preview \
  --config synth_generators/line_generator/configs/example.yaml \
  --output synthetic_line_preview.png
```

Сохранить чистый датасет на диск в виде `uint8` torch-чанков:

```bash
python -m synth_generators.line_generator.materialize \
  --config synth_generators/line_generator/configs/example.yaml
```

В каждом `chunk_*.pt` лежат только данные: `images` (`uint8`,
`N x C x H x W`) и исходные `texts` как текстовая разметка. Конфиг, алфавит,
настройки обучения и настройки аугментаций в offline-датасет не сохраняются.
`output_dir`, `chunk_size` и `overwrite` задаются в generation-конфиге.
Offline-генерация сохраняет чистые строки без аугментаций.

Посмотреть пример из чанка с теми же аугментациями, которые использует
обучение:

```bash
python synth_generators/line_generator/render_text.py \
  --chunks-dir data/line_chunks \
  --index 0 \
  --config configs/example_train.yaml \
  --output output/render_chunk.png
```

Запустить обучение на синтетике:

```bash
python train.py --config configs/example_train.yaml
```

В training-конфиге задаются `chunks_dir` или `generator_config`, алфавит,
learning rate, batch size, workers, checkpoint path, preview-настройки и
GPU-аугментации. При старте обучения `train.py` читает `texts` из датасета,
сравнивает символы с training-алфавитом и сохраняет статистику в:

```text
checkpoints/alphabet_stats.tsv
```

По умолчанию `train.py` применяет настроенные аугментации на устройстве
обучения (`cuda`, если доступна). Отключить это можно в training-конфиге:

```yaml
gpu_augmentations: false
```

Для offline-чанков batch-и по умолчанию группируются по `chunk_*.pt`, чтобы
один batch не заставлял читать десятки файлов с диска. Для реального обучения
обычно имеет смысл включить `num_workers`, `prefetch_factor` и подобрать
`batch_size` в training-конфиге.

Картинки из чанков остаются `uint8` до переноса batch на устройство обучения и
нормализуются уже там, поэтому CPU RAM и host-to-device transfer не раздуваются
до `float32` раньше времени.

Обучение пишет компактный лог по эпохам в консоль и TSV-файл:

```text
checkpoints/training_log.tsv
```

Разбиение на батчи настраивается явно:

```yaml
batch_size: 128
num_workers: 4
drop_last: true
log_every: 10
```

При старте печатаются размеры train/validation split, batch size, количество
батчей и лимиты `max_train_batches` / `max_val_batches`, если они заданы.
По умолчанию `log_every: 1`, то есть loss печатается на каждом batch; значение
`0` отключает batch-логи.

Сохранить примеры именно тех тензоров, которые подаются в train/validation:

```yaml
preview_samples: 16
preview_dir: input_previews
```

Картинки будут сохранены в `input_previews/train` и `input_previews/val`.
Сохранение происходит внутри train/validation loop прямо перед `model(imgs)`,
поэтому это ровно те изображения, которые подаются на вход сети, уже с
применёнными аугментациями. Рядом создаётся `labels.tsv` с именем файла,
текстом и длиной target. Размер validation-части задаётся через
`val_fraction`, по умолчанию `0.1`.

Для продолжения обучения выставьте в training-конфиге:

```yaml
resume: true
```

Инференс на синтетическом примере:

```bash
python inference.py --checkpoint checkpoints/best_model.pth --sample-index 0
```
