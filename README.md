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

В generation-конфиге `sample_alphabet` задает символы, из которых синтезируются
строки. В training-конфиге `alphabet` задает классы модели. В примерах оба
набора начинаются с пробела: пробел является обычным классом, но генератор
нормализует строки, убирая пробелы в начале/конце и схлопывая несколько
пробелов подряд в один.

Примеры конфигов:

- `synth_generators/line_generator/configs/eng_001.yaml` — генерация;
- `configs/eng_train_001.yaml` — обучение.

Шрифты можно задавать папкой, путь считается относительно YAML-конфига:

```yaml
font_dir: ../fonts
font_extensions:
  - .ttf
  - .otf
  - .ttc
  - .otc
```

При создании генератора запускается fonts check: шрифты без полного покрытия
`sample_alphabet` отбрасываются, а в терминал выводятся количество найденных,
принятых и отклоненных шрифтов, примеры отклонений и часто отсутствующие
символы. `font_paths` тоже поддерживается, если нужно явно перечислить файлы.

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
  preprocess_geometry: 0.3
  strong_blur: 0.08
  motion_blur: 0.08
  scale: 0.15
  darkening: 0.2
  noise: 0.75
  projective: 0.12
  rotate: 0.8
  crop_x: 0.08
  crop_y: 0.05
  random_line: 0.1
  morphology: 0.08
  unsharp_mask: 0.12
  gaussian_blur: 0.0
  gaussian_noise: 0.0
  brightness: 0.3
  contrast: 0.3
  invert: 0.0
augmentations:
  preprocess_geometry:
    scale_x_min: -0.15
    scale_x_max: 0.15
    y_pad_min: -0.25
    y_pad_max: 0.10
    fillcolor: 255
  rotate:
    max_degrees: 1.0
    fillcolor: 255
  random_line:
    angle_degrees_min: -4.0
    angle_degrees_max: 4.0
    line_width_min: 1.0
    line_width_max: 2.5
    alpha_min: 0.35
    alpha_max: 0.9
    value_min: 0.0
    value_max: 80.0
    y_min: 0.15
    y_max: 0.9
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

Доступные OCR-аугментации: `cycle_shift`, `preprocess_geometry`,
`strong_blur`, `motion_blur`, `scale`, `darkening`, `noise`, `projective`,
`rotate`, `crop_x`, `crop_y`, `random_line`, `morphology`, `unsharp_mask`.
`preprocess_geometry` повторяет смысл inference-параметров `scale_x/y_pad`.
`random_line` добавляет почти горизонтальную линию под небольшим углом.
Старые `gaussian_blur` и `gaussian_noise` оставлены как совместимые алиасы.

Сохранить один пример изображения:

```bash
python -m synth_generators.line_generator.preview \
  --config synth_generators/line_generator/configs/eng_001.yaml \
  --output synthetic_line_preview.png
```

Сохранить чистый датасет на диск в виде `uint8` torch-чанков:

```bash
python -m synth_generators.line_generator.generate_dataset \
  --config synth_generators/line_generator/configs/eng_001.yaml
```

В каждом `chunk_*.pt` лежат только данные: `images` (`uint8`,
`N x C x H x W`) и исходные `texts` как текстовая разметка. Рядом создается
`metadata.yaml` с параметрами датасета: алфавитом, `space_char`, размерами
картинок, числом каналов и максимальной длиной текста. Настройки обучения и
настройки аугментаций в offline-датасет не сохраняются. `output_dir`,
`chunk_size`, `num_workers` и `overwrite` задаются в generation-конфиге.
Датасет сохраняется в подпапку с именем generation-конфига, например
`data/eng_001`. Если `num_workers > 0`, чанки генерируются параллельно.
Offline-генерация сохраняет чистые строки без аугментаций.

Посмотреть пример из чанка с теми же аугментациями, которые использует
обучение:

```bash
python synth_generators/line_generator/render_text.py \
  --chunks-dir data/eng_001 \
  --index 0 \
  --config configs/eng_train_001.yaml \
  --output output/render_chunk.png
```

Запустить обучение на синтетике:

```bash
python train.py --config configs/eng_train_001.yaml
```

В training-конфиге задаются `chunks_dir` или `generator_config`, learning rate,
batch size, workers, checkpoint path, preview-настройки и GPU-аугментации.
Алфавит, размеры картинок, число каналов и `max_text_length` берутся из
`metadata.yaml` в папке чанков или из `generator_config`. При необходимости эти
поля можно явно указать в training-конфиге как override. При старте обучения
`train.py` читает `texts` из датасета, сравнивает символы с effective-алфавитом
и сохраняет статистику в:

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
batch_count: 500
num_workers: 4
drop_last: true
log_every: 10
```

Learning rate scheduler задаётся в training-конфиге. По умолчанию используется
`ReduceLROnPlateau`: если validation loss несколько эпох не улучшается, lr
уменьшается.

```yaml
lr: 0.001
scheduler: reduce_on_plateau
scheduler_factor: 0.5
scheduler_patience: 3
scheduler_min_lr: 0.000001
scheduler_threshold: 0.0001
scheduler_cooldown: 0
```

Также поддерживаются `scheduler: none`, `scheduler: cosine` и
`scheduler: step`. Состояние scheduler сохраняется в checkpoint и
восстанавливается при `resume: true`.

`batch_count` ограничивает train-эпоху фиксированным числом случайно выбранных
batch-ей. Если `batch_count: null`, эпоха проходит весь train split. Для
offline-чанков sampled-batch режим выбирает каждый batch из одного `chunk_*.pt`,
чтобы чтение с диска оставалось локальным и быстрым.

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

Сохранить читаемую debug-картинку с исходным изображением, изображением после
inference-preprocessing, итоговым ответом, строками decoded-символов в порядке
ответа и top-k парами `символ confidence`:

```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --image path/to/line.png \
  --scale-x 0.0 \
  --y-pad 0.0 \
  --debug-image output/inference_debug.png \
  --debug-top-k 8
```

Python API для использования из других скриптов:

```python
from fcn_ocr import TextRecognizer

recognizer = TextRecognizer(
    "checkpoints/best_model.pth",
    device="cuda",
    scale_x=0.0,
    y_pad=0.0,
)

for path, result in recognizer.recognize_paths(["line_1.png", "line_2.png"]):
    print(path, result.text)
```

`scale_x` и `y_pad` — inference-only гиперпараметры предобработки. `scale_x:
0.2` растягивает ширину на 20%, `scale_x: -0.2` сжимает на 20%. `y_pad: 0.2`
добавляет вертикальный паддинг перед resize, `y_pad: -0.2` симметрично обрезает
20% высоты перед resize.

Если внешний скрипт лежит вне репозитория, добавьте корень проекта в
`PYTHONPATH`:

```bash
PYTHONPATH=/path/to/FCN-OCR-recognizer python my_script.py
```

Оценка Label Studio export JSON из корня проекта:

```bash
python evaluate_ocr.py \
  --json path/to/export.json \
  --images path/to/images \
  --checkpoint checkpoints/best_model.pth \
  --out output/ocr_metrics.csv \
  --scale-x 0.0 \
  --y-pad 0.0 \
  --batch-size 32
```

Подбор inference-preprocessing через Optuna:

```bash
python evaluate_ocr.py \
  --json path/to/export.json \
  --images path/to/images \
  --checkpoint checkpoints/best_model.pth \
  --out output/ocr_metrics.csv \
  --optuna-trials 30 \
  --optuna-scale-x-min -0.25 \
  --optuna-scale-x-max 0.25 \
  --optuna-y-pad-min -0.25 \
  --optuna-y-pad-max 0.25 \
  --optuna-metric global_char_accuracy \
  --optuna-trials-out output/optuna_trials.tsv
```

Если Optuna не установлена:

```bash
pip install optuna
```

Обучение с оценкой OCR после каждой эпохи:

```bash
python train_with_eval.py \
  --train-config configs/eng_train_101.yaml \
  --eval-json path/to/export.json \
  --eval-images path/to/images \
  --eval-out-dir output/train_eval \
  --eval-batch-size 32 \
  --eval-log-every 0
```

После каждой эпохи сохраняется текущий чекпоинт, запускается `evaluate_ocr`,
пишется per-epoch CSV и общий `eval_summary.tsv`. Для подбора `scale_x/y_pad`
на каждой эпохе добавьте, например, `--optuna-trials 20`.
