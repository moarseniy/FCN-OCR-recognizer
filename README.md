# FCN-OCR-recognizer

Полносверточный распознаватель строк и простой синтетический генератор строк
для обучения без ручной разметки.

## Synthetic line generator

Генератор находится в `synth_generators/line_generator`.

Он возвращает элементы в формате, который подходит текущей FCN-идее:

- `image`: тензор `C x H x W`;
- `target` — padded-тензор длиной `max_text_length`;
- `length` — реальная длина строки.

Обучение по умолчанию использует CTC loss. Последний класс модели — `blank`,
поэтому `num_classes = len(alphabet) + 1`.

Для экспериментов со старой схемой `final -> softmax -> logreg` есть
`loss_mode: legacy_logreg`. В этом режиме `final` обычно имеет ровно
`len(alphabet)` выходов, без `blank`. Если в данных есть плотная symbol-map
разметка, можно выбрать `legacy_target_mode: dense_symbols`: таргет будет
выравниваться как в старом графе через `max_pool2d(kernel=(4, 1),
stride=(4, 1), padding=(1, 0))` и `cropX=[6, -5]`. Для вертикального
сегментатора есть `legacy_target_mode: binary_gaps`: `final` имеет 2 выхода,
а таргет содержит 0/1 по X-колонкам, где `1` означает промежуток между
символами. Для сегментатора в стиле cut-projection есть
`loss_mode: cut_projection`: `final` имеет 1 выход, а таргет содержит
одномерную heatmap-проекцию с пиками в координатах правильных разрезов между
символами. Для текущих текстовых чанков доступен
`legacy_target_mode: uniform_text`, который равномерно раскладывает строку по
выходной ширине модели. Это удобный режим для проверки идеи, но CTC остается более корректным
для строк переменной длины и повторов одинаковых символов.

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
`crop_x` и `crop_y` обрезают края, а затем ресайзят результат обратно в
исходный размер тензора.
`random_line` добавляет почти горизонтальную линию под небольшим углом.
Старые `gaussian_blur` и `gaussian_noise` оставлены как совместимые алиасы.

Сохранить один пример изображения по указанному тексту:

```bash
python synth_generators/line_generator/render_text.py \
  --text "ABC 123" \
  --config synth_generators/line_generator/configs/eng_001.yaml \
  --output synthetic_line_preview.png
```

Сохранить чистый датасет на диск в виде `uint8` torch-чанков:

```bash
python -m synth_generators.line_generator.generate_dataset \
  --config synth_generators/line_generator/configs/eng_001.yaml
```

В generation-конфиге можно задавать межсимвольные интервалы. Значения
сэмплятся один раз на всю строку, поэтому строка остается написанной одним
стилем:

```yaml
char_spacing_min: -0.4
char_spacing_max: 1.6
word_spacing_multiplier_min: 0.75
word_spacing_multiplier_max: 1.7
```

`char_spacing_*` добавляет единый tracking между соседними символами внутри
слова, а `word_spacing_multiplier_*` отдельно меняет ширину пробелов.

В каждом `chunk_*.pt` лежат только данные: `images` (`uint8`,
`N x C x H x W`) и исходные `texts` как текстовая разметка. Если нужен
`legacy_logreg` с плотной разметкой, добавьте в generation-конфиг:

```yaml
save_dense_targets: true
save_binary_gap_targets: true
save_cut_projection_targets: true
binary_gap_min_width: 1
binary_gap_include_spaces: false
binary_gap_include_margins: false
cut_projection_peak_radius: 1
cut_projection_include_margins: false
```

Тогда в чанки также попадет `dense_targets` (`N x W`) — класс символа для
каждой X-колонки исходного кропа. Если включен `save_binary_gap_targets`, в
чанки попадет `binary_gap_targets` (`N x W`) — бинарная разметка промежутков
между символами. Если включен `save_cut_projection_targets`, в чанки попадет
`cut_projection_targets` (`N x W`, `uint8`) — heatmap правильных вертикальных
разрезов. При обучении loss делает crop и при необходимости
пересэмплирует эту разметку к выходной ширине сети `T`. Для
`vertical_segmentator_fcn` ширина выхода сохраняется 1:1, поэтому в конфиге
используются crop `0/0` и strict-width. Рядом создается `metadata.yaml` с
параметрами.
датасета: алфавитом, `space_char`, размерами картинок, числом каналов и
максимальной длиной текста. Настройки обучения и настройки аугментаций в
offline-датасет не сохраняются. `output_dir`, `chunk_size`, `num_workers` и
`overwrite` задаются в generation-конфиге. Датасет сохраняется в подпапку с
именем generation-конфига, например `data/eng_001`. Если `num_workers > 0`,
чанки генерируются параллельно. Offline-генерация сохраняет чистые строки без
аугментаций.

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

FCN-архитектуры лежат в `fcn_architectures/`: одна архитектура - один файл.
Новый файл должен определить `ARCHITECTURE_NAME` и `create_model(...)`.
После этого архитектуру можно выбрать в любом training-конфиге:

```yaml
architecture: legacy_fcn
architecture_params: {}
```

Архитектура не привязана к задаче: один и тот же файл можно использовать и для
OCR, и для вертикального сегментатора. Роль задается остальными полями конфига:
`loss_mode`, `legacy_target_mode`, числом классов и разметкой в чанках.

Для экспериментов с вертикальным сегментатором есть пример архитектуры, которая
сохраняет горизонтальное разрешение выхода 1:1 с входной картинкой:

```yaml
architecture: vertical_segmentator_fcn
architecture_params:
  base_channels: 16
  temporal_kernel: 5
  dropout: 0.05
```

Имя архитектуры сохраняется в checkpoint, поэтому `inference.py`,
`evaluate_ocr.py` и `VerticalSegmentator` автоматически собирают такую же сеть
при загрузке модели. Старые checkpoint без этого поля считаются
`legacy_fcn`.

Для обучения в старом плотном режиме на чанках с `dense_targets`:

```yaml
loss_mode: legacy_logreg
legacy_target_mode: dense_symbols
legacy_crop_left: 6
legacy_crop_right: 5
```

Для обучения вертикального сегментатора на `binary_gap_targets`:

```yaml
loss_mode: legacy_logreg
legacy_target_mode: binary_gaps
legacy_crop_left: 0
legacy_crop_right: 0
legacy_strict_width: true
segmentator_gap_threshold: 0.5
segmentator_min_gap_width: 1
segmentator_merge_gap_width: 0
```

Пример конфига: `configs/eng_train_101_gaps.yaml`.

Для обучения вертикального сегментатора на heatmap разрезов:

```yaml
loss_mode: cut_projection
cut_projection_crop_left: 0
cut_projection_crop_right: 0
cut_projection_strict_width: true
cut_projection_loss: mse
cut_projection_positive_weight: 4.0
segmentator_gap_threshold: 0.35
segmentator_peak_min_distance: 3
segmentator_cut_postprocess: widths
segmentator_cut_min_width: 3
segmentator_cut_max_width: 24
segmentator_cut_candidate_threshold: 0.12
segmentator_cut_smooth_radius: 1
```

Пример конфига: `configs/eng_train_101_cuts.yaml`.

Для уже обученного cuts-чекпоинта эти параметры можно переопределять прямо в
`inference.py`: `--segmentator-cut-postprocess`, `--segmentator-cut-min-width`,
`--segmentator-cut-max-width`, `--segmentator-cut-candidate-threshold` и
`--segmentator-cut-smooth-radius`.

Подобрать параметры вертикального сегментатора без OCR, сравнивая число
предсказанных межбуквенных промежутков с длиной строки из Label Studio:

```bash
python evaluate_segmentator.py \
  --json labels.json \
  --images images \
  --checkpoint checkpoints/gap_segmentator/best_model.pth \
  --out output/segmentator_lengths.csv \
  --optuna-trials 100 \
  --optuna-trials-out output/segmentator_trials.tsv \
  --optuna-tune-baseline-crop \
  --optuna-tune-baseline-params
```

Оцениваемая длина считается как `число gap-runs + 1`, поэтому пробел в
разметке считается обычным символом.

В training-конфиге задаются `chunks_dir`, optimizer, learning rate, batch size,
workers, checkpoint path, preview-настройки и GPU-аугментации. Online-генерация
во время обучения удалена: данные нужно сначала сохранить чанками через
`synth_generators.line_generator.generate_dataset`.
Алфавит, размеры картинок, число каналов и `max_text_length` берутся из
`metadata.yaml` в папке чанков. При необходимости эти поля можно явно указать в
training-конфиге как override. При старте обучения
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
optimizer: adam
weight_decay: 0.0
scheduler: reduce_on_plateau
scheduler_factor: 0.5
scheduler_patience: 3
scheduler_min_lr: 0.000001
scheduler_threshold: 0.0001
scheduler_cooldown: 0
```

Поддерживаются оптимизаторы:

```yaml
optimizer: adam   # adam, adamw, sgd, rmsprop

# adam/adamw
adam_beta1: 0.9
adam_beta2: 0.999
adam_eps: 0.00000001

# sgd
sgd_momentum: 0.9
sgd_nesterov: false

# rmsprop
rmsprop_alpha: 0.99
rmsprop_momentum: 0.0
rmsprop_eps: 0.00000001
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
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --sample-index 0 \
  --config synth_generators/line_generator/configs/eng_001.yaml
```

Сохранить читаемую debug-картинку с исходным изображением, изображением после
inference-preprocessing, итоговым ответом, строками decoded-символов в порядке
ответа и top-k парами `символ confidence`:

```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --segmentator-checkpoint checkpoints/gap_segmentator/best_model.pth \
  --image path/to/line.png \
  --decode-with-segmentator \
  --scale-x 0.0 \
  --y-pad 0.0 \
  --baseline-crop \
  --debug-image output/inference_debug.png \
  --debug-top-k 8
```

`--segmentator-checkpoint` опционален. Если он передан вместе с
`--debug-image`, в debug-картинку дополнительно попадет дорожка вертикального
сегментатора: `0` для не-промежутка и `1` для промежутка между символами.
Gap-runs рисуются как четкие вертикальные красные линии поверх входа
сегментатора. В текущей разметке пробел считается таким же символом, поэтому
сам span пробела не размечается как gap; gap-ами остаются только границы между
соседними символами, включая границы вокруг пробела.

Для legacy OCR можно дополнительно включить декодирование через вертикальный
сегментатор: `--decode-with-segmentator`. Тогда обычный ответ OCR останется в
логе и debug-отчете как `result`, а рядом появится `legacy+cuts`: каждый символ
берется как top-класс по средним вероятностям OCR внутри интервала между
соседними cut-точками.

Порог и постобработку сегментатора можно хранить в train-конфиге/checkpoint и
при необходимости переопределить на инференсе:

```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --segmentator-checkpoint checkpoints/gap_segmentator/best_model.pth \
  --image path/to/line.png \
  --segmentator-gap-threshold 0.6 \
  --segmentator-min-gap-width 2 \
  --segmentator-merge-gap-width 1 \
  --debug-image output/inference_debug.png
```

Python API для использования из других скриптов:

```python
from fcn_ocr import TextRecognizer

recognizer = TextRecognizer(
    "checkpoints/best_model.pth",
    device="cuda",
    scale_x=0.0,
    y_pad=0.0,
    baseline_crop=True,
)

for path, result in recognizer.recognize_paths(["line_1.png", "line_2.png"]):
    print(path, result.text)
```

`scale_x` и `y_pad` — inference-only гиперпараметры предобработки. `scale_x:
0.2` растягивает ширину на 20%, `scale_x: -0.2` сжимает на 20%. `y_pad: 0.2`
добавляет вертикальный паддинг перед resize, `y_pad: -0.2` симметрично обрезает
20% высоты перед resize.

`baseline_crop` перед этим пытается найти базовую линию текста, убрать небольшой
наклон, вертикально обрезать строку относительно этой линии и только потом
применить `y_pad`/resize. Детектор строит очищенную текстовую маску, нижний
профиль строки и робастно фитит линию с confidence; если уверенность низкая,
crop не применяется. Для настройки доступны `baseline_top_pad`,
`baseline_bottom_pad`, `baseline_deskew` и `baseline_max_angle`; в
`--debug-image` дополнительно попадают маска, найденная линия, inlier-точки,
confidence и кроп.

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
  --baseline-crop \
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
  --baseline-crop \
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
