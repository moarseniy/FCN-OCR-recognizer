# FCN-OCR-recognizer

Полносверточный распознаватель строк и простой синтетический генератор строк
для обучения без ручной разметки.

## Synthetic line generator

Генератор находится в `synth_generators/line_generator`.

Он сохраняет элементы в формате, который подходит текущей FCN-идее:

- `images`: тензор `N x C x H x W` в `uint8`;
- `texts`: исходная текстовая разметка;
- опционально `dense_targets`, `cut_projection_targets` и `baseline_targets` для конкретных
  режимов обучения.

Основной OCR-режим сейчас — `loss_mode: legacy_logreg` и
`legacy_target_mode: dense_symbols`. В этом режиме `final` имеет
`len(alphabet)` выходов, а таргет выравнивается как в старом графе через
`max_pool2d(kernel=(4, 1), stride=(4, 1), padding=(1, 0))` и `cropX=[6, -5]`.
Для вертикального сегментатора используется `loss_mode: cut_projection`:
`final` имеет 1 выход, а таргет содержит одномерную heatmap-проекцию с пиками
в координатах правильных разрезов между символами.
Для нейронного детектора базовых линий используется `loss_mode:
baseline_heatmap`: сеть выдает 2D heatmap `2 x H x W`, где канал 0 отвечает за
верхнюю линию текстового поля, а канал 1 - за нижнюю.

В generation-конфиге `sample_alphabet` задает символы, из которых синтезируются
строки. В training-конфиге `alphabet` задает классы модели. В примерах оба
набора начинаются с пробела: пробел является обычным классом, но генератор
нормализует строки, убирая пробелы в начале/конце и схлопывая несколько
пробелов подряд в один.

Примеры конфигов:

- `synth_generators/line_generator/configs/eng_001.yaml` — генерация;
- `configs/eng_train_001.yaml` — обучение.
- `synth_generators/line_generator/configs/eng_101.yaml` — генерация OCR-чанков с `dense_targets`;
- `synth_generators/line_generator/configs/eng_101_cuts.yaml` — генерация чанков для вертикального cut-сегментатора.
- `synth_generators/line_generator/configs/eng_101_baselines.yaml` — генерация чанков для top/bottom baseline-детектора.

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
crop под размер строки и рисует текст поверх. Resize фона не используется.
Если длинная строка шире доступных фонов, подложка собирается из нескольких
crop-only фрагментов. Если `background_dir: null`, используется однотонный фон
из поля `background`. Относительный путь
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
  x_pad: 0.1
  crop_x: 0.08
  crop_y: 0.05
  rescale_quality: 0.2
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
  x_pad:
    pad_min: 0.02
    pad_max: 0.10
    fillcolor: 255
  rescale_quality:
    factor_min: 0.35
    factor_max: 0.75
    down_mode: bilinear
    up_mode: nearest
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
`rotate`, `x_pad`, `crop_x`, `crop_y`, `rescale_quality`, `random_line`,
`morphology`, `unsharp_mask`.
`preprocess_geometry` повторяет смысл inference-параметров `scale_x/y_pad`.
`x_pad` сжимает содержимое по X внутрь исходного размера тензора и заполняет
поля `fillcolor`; для target-ов применяется такое же преобразование.
`crop_x` и `crop_y` обрезают края, а затем ресайзят результат обратно в
исходный размер тензора.
`rescale_quality` уменьшает картинку до доли `factor`, затем возвращает к
исходному размеру, чтобы имитировать потерю разрешения/JPEG-подобную грубость
без изменения геометрической разметки.
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
save_cut_projection_targets: true
save_baseline_targets: true
cut_projection_peak_radius: 1
cut_projection_include_margins: false
baseline_target_radius: 1
```

Тогда в чанки также попадет `dense_targets` (`N x W`) — класс символа для
каждой X-колонки исходного кропа. Если включен `save_cut_projection_targets`,
в чанки попадет `cut_projection_targets` (`N x W`, `uint8`) — heatmap
правильных вертикальных разрезов. При обучении loss делает crop и при необходимости
пересэмплирует эту разметку к выходной ширине сети `T`. Для
`vertical_segmentator_fcn` ширина выхода сохраняется 1:1, поэтому в конфиге
используются crop `0/0` и strict-width. Если включен `save_baseline_targets`, в чанки попадет
`baseline_targets` (`N x 2 x H x W`, `uint8`) — две горизонтальные heatmap-линии
для верхней и нижней границы основной строки. Для baseline-датасета можно
добавлять соседние строки как вертикальный мусор: основная строка остается
целиком в середине, а верхняя и нижняя строки рисуются так, чтобы больше
заданной доли каждой из них было обрезано верхним/нижним краем картинки:

```yaml
neighbor_lines_probability: 0.7
neighbor_line_min_crop_ratio: 0.65
neighbor_line_visible_ratio_min: 0.06
neighbor_line_gap_min: 0
neighbor_line_gap_max: 5
```

`neighbor_lines_probability` применяется независимо к верхней и нижней строке,
поэтому в датасете встречаются все варианты: обе соседние строки, только
верхняя, только нижняя или чистая основная строка.
`neighbor_line_min_crop_ratio: 0.65` означает, что у каждой добавленной
мусорной строки будет видно не больше 35% высоты. `neighbor_line_gap_*`
задает расстояние в пикселях между основной
строкой и видимым фрагментом соседней строки.
Рядом создается `metadata.yaml` с параметрами датасета: алфавитом,
`space_char`, размерами картинок, числом каналов и
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

Для детекции базовых линий есть `baseline_detector_fcn`: она сохраняет и
ширину, и высоту, а на выходе дает два канала top/bottom heatmap:

```yaml
architecture: baseline_detector_fcn
architecture_params:
  base_channels: 24
  depth: 6
  dropout: 0.05
```

Имя архитектуры сохраняется в checkpoint, поэтому `inference.py`,
`evaluate_ocr.py` и `VerticalSegmentator` автоматически собирают такую же сеть
при загрузке модели. Старые checkpoint без этого поля считаются
`legacy_fcn`.

Для более тяжелых FCN-экспериментов добавлены:

- `legacy_fcn_wide`: drop-in вариант старой OCR-сети с теми же kernel/stride и
  такой же шириной выхода, но с большим числом каналов.
- `legacy_fcn_highres`: plain FCN в стиле старой OCR-сети, но без
  горизонтального stride=2 в `conv2`; для 48x64 дает более плотный выход
  `T=48` вместо `T=19`.
- `residual_temporal_fcn`: width-preserving FCN с residual-блоками и temporal
  convolutions по X. Для OCR ее удобнее запускать с `legacy_crop_left: 0`,
  `legacy_crop_right: 0`, `legacy_strict_width: true`; для cuts она также
  совместима с `cut_projection_strict_width: true`.

Готовые примеры: `configs/eng_train_101_wide.yaml`,
`configs/eng_train_101_highres.yaml`,
`configs/eng_train_101_residual.yaml`,
`configs/eng_train_101_cuts_residual.yaml`.

Для обучения в старом плотном режиме на чанках с `dense_targets`:

```yaml
loss_mode: legacy_logreg
legacy_target_mode: dense_symbols
legacy_crop_left: 6
legacy_crop_right: 5
legacy_label_align: majority_bins
legacy_label_min_majority: 0.6
legacy_space_weight: 0.5
```

`legacy_label_align` управляет тем, как плотная разметка шириной входного кропа
сводится к временной ширине выхода сети. `majority_bins` делит dense-разметку на
интервалы под выходные позиции, берет класс большинства и игнорирует позицию,
если большинство слабее `legacy_label_min_majority`. Старое поведение с выбором
одной центральной точки можно вернуть через `legacy_crop_resample`.
`legacy_space_weight` уменьшает или увеличивает вклад класса пробела в OCR-loss.

Для обучения вертикального сегментатора на heatmap разрезов:

```yaml
loss_mode: cut_projection
cut_projection_crop_left: 0
cut_projection_crop_right: 0
cut_projection_strict_width: true
cut_projection_loss: mse
cut_projection_positive_weight: 4.0
segmentator_cut_threshold: 0.35
segmentator_peak_min_distance: 3
segmentator_cut_postprocess: widths
segmentator_cut_min_width: 3
segmentator_cut_max_width: 24
segmentator_cut_candidate_threshold: 0.12
segmentator_cut_smooth_radius: 1
```

Пример конфига: `configs/eng_train_101_cuts.yaml`.
Соответствующий generation-конфиг: `synth_generators/line_generator/configs/eng_101_cuts.yaml`.

Для обучения нейронного детектора верхней/нижней базовой линии:

```yaml
loss_mode: baseline_heatmap
baseline_heatmap_strict_size: true
baseline_heatmap_loss: bce
baseline_heatmap_positive_weight: 6.0
```

Пример конфига: `configs/eng_train_101_baselines.yaml`.
Соответствующий generation-конфиг:
`synth_generators/line_generator/configs/eng_101_baselines.yaml`.

Для уже обученного cuts-чекпоинта эти параметры можно переопределять прямо в
`inference.py`: `--segmentator-cut-threshold`,
`--segmentator-peak-min-distance`, `--segmentator-cut-postprocess`,
`--segmentator-cut-min-width`, `--segmentator-cut-max-width`,
`--segmentator-cut-candidate-threshold` и `--segmentator-cut-smooth-radius`.

Подобрать параметры вертикального сегментатора без OCR, сравнивая число
предсказанных межбуквенных промежутков с длиной строки из Label Studio:

```bash
python evaluate_segmentator.py \
  --json labels.json \
  --images images \
  --checkpoint checkpoints/cut_segmentator/best_model.pth \
  --out output/segmentator_lengths.csv \
  --optuna-trials 100 \
  --optuna-trials-out output/segmentator_trials.tsv \
  --optuna-tune-baseline-crop \
  --optuna-tune-baseline-params
```

Оцениваемая длина считается как `число cut-точек + 1`, поэтому пробел в
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
  --segmentator-checkpoint checkpoints/cut_segmentator/best_model.pth \
  --image path/to/line.png \
  --decode-with-segmentator \
  --scale-x 0.0 \
  --y-pad 0.0 \
  --x-pad 0.0 \
  --baseline-crop \
  --baseline-detector-checkpoint checkpoints/baseline_detector/best_model.pth \
  --debug-image output/inference_debug.png \
  --debug-top-k 8
```

`--segmentator-checkpoint` опционален. Если он передан вместе с
`--debug-image`, в debug-картинку дополнительно попадет дорожка вертикального
сегментатора: cut-точки рисуются как четкие вертикальные красные линии поверх
входа сегментатора. В текущей разметке пробел считается таким же символом,
поэтому разрезы ставятся по границам между соседними символами, включая
границы вокруг пробела.

Для legacy OCR можно дополнительно включить декодирование через вертикальный
сегментатор: `--decode-with-segmentator`. Тогда обычный ответ OCR останется в
логе и debug-отчете как `result`, а рядом появится `legacy+cuts`: каждый символ
берется как top-класс по средним вероятностям OCR внутри интервала между
соседними cut-точками. Крайние интервалы ограничиваются найденными X-границами
foreground на входе OCR, поэтому левый/правый паддинг не декодируется как
лишние символы. Дополнительно включен trim пустых крайних интервалов и
слияние слишком узких крайних интервалов с соседями; это можно настроить через
`--segmentator-edge-min-width`, `--segmentator-edge-min-ink-ratio`,
`--segmentator-edge-min-pixel-density` или отключить флагом
`--no-segmentator-edge-trim`.

Порог и постобработку сегментатора можно хранить в train-конфиге/checkpoint и
при необходимости переопределить на инференсе:

```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --segmentator-checkpoint checkpoints/cut_segmentator/best_model.pth \
  --image path/to/line.png \
  --segmentator-cut-threshold 0.6 \
  --segmentator-peak-min-distance 2 \
  --segmentator-cut-min-width 2 \
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
    x_pad=0.0,
    baseline_crop=True,
)

for path, result in recognizer.recognize_paths(["line_1.png", "line_2.png"]):
    print(path, result.text)
```

### Inference Pipeline

Инференс устроен как последовательный пайплайн:

1. Загружается OCR checkpoint. Из него берутся `alphabet`, `architecture`,
   `num_classes`, `image_height`, `channels`, режим loss/target и параметры
   legacy crop. Модель создается через `fcn_architectures.create_model`,
   загружает `model_state_dict` и переводится в `eval`.
2. Входная картинка приводится к `RGB` или `L` в зависимости от `channels`.
   В `--debug-image` этот шаг подписан как `preprocess 00 input converted`.
3. Если включен `--baseline-crop`, запускается детектор нижней и верхней
   текстовых линий. Если передан `--baseline-detector-checkpoint`, линии
   берутся из нейронного `baseline_heatmap`-детектора; иначе используется
   эвристика по текстовым маскам. В режиме `--baseline-rectify lines` нижняя
   линия используется для deskew, после поворота линии ищутся повторно, а
   вертикальный crop строится строго по паре верх/низ без bbox-fallback. В
   режиме `--baseline-rectify curved` neural heatmap читается как две кривые
   top/bottom, и строка выпрямляется через dewarp по этим кривым; при неудаче
   curved-режим откатывается к `lines`.
4. `x_pad` применяется до `y_pad`, resize и `scale_x`. Он добавляет слева и
   справа долю текущей ширины, но не отражает символы: поля заполняются
   медианным фоном боковой полосы исходной геометрии. В debug это
   `preprocess 02 x-pad border median`.
5. `y_pad` добавляет или обрезает высоту. Положительное значение добавляет
   поля сверху/снизу, заполненные медианным цветом рамки текущей картинки;
   отрицательное значение симметрично режет высоту.
6. Картинка приводится к высоте `image_height` из checkpoint с сохранением
   пропорций по ширине.
7. `scale_x` применяется последним из геометрических inference-параметров:
   `0.2` растягивает ширину на 20%, `-0.2` сжимает на 20%.
8. Получившаяся картинка нормализуется в tensor `[0, 1]` формата
   `1 x C x H x W` и подается в FCN. Сеть возвращает logits `B x C x T`.
9. Обычный OCR decode берет `argmax` по классам на каждом timestep, схлопывает
   подряд идущие одинаковые классы и переводит индексы в символы alphabet.
10. Если передан `--segmentator-checkpoint`, отдельно запускается вертикальный
    сегментатор на той же исходной картинке и с теми же inference-параметрами.
    Его результат рисуется отдельной дорожкой в debug-отчете.
11. Если включен `--decode-with-segmentator`, OCR logits декодируются через
    интервалы между cut-точками сегментатора: для каждого интервала берется средняя
    вероятность OCR-классов, а top-класс становится символом. Крайние cut-линии
    могут использоваться как границы текста через `--segmentator-boundary-cuts`.

Параметры preprocessing:

- `scale_x`: нормированное растяжение/сжатие ширины после resize по высоте.
- `y_pad`: нормированный вертикальный padding/crop до resize по высоте.
- `x_pad`: нормированный горизонтальный padding до `y_pad`/resize/`scale_x`.
- `baseline_crop`: включает поиск нижней и верхней текстовых линий, deskew и
  вертикальный crop.

### Inference Parameter Reference

#### OCR Preprocessing

| Параметр | Что делает |
| --- | --- |
| `--scale-x` | Растягивает или сжимает ширину после resize по высоте. `0.2` значит шире на 20%, `-0.2` значит уже на 20%. |
| `--y-pad` | Добавляет или обрезает высоту до resize по высоте. `0.2` добавляет поля сверху/снизу, `-0.2` симметрично режет высоту. |
| `--x-pad` | Добавляет горизонтальные поля до `y_pad`, resize и `scale_x`. Поля заполняются медианным фоном боковой полосы, символы не отражаются. |
| `--show-raw` | Печатает raw timestep-классы обычного OCR decode. Полезно, чтобы увидеть, где сеть держит один класс несколько timestep-ов подряд. |
| `--debug-image` | Сохраняет подробную картинку с исходным изображением, preprocessing-шагами, входом сети, результатом OCR, top-k кандидатами и сегментатором. |
| `--debug-top-k` | Сколько top-кандидатов по confidence выводить для каждого decoded-символа или cut-интервала. |

#### Baseline Crop

| Параметр | Что делает |
| --- | --- |
| `--baseline-crop` | Включает поиск нижней и верхней текстовых линий, optional deskew и вертикальный crop вокруг строки. |
| `--no-baseline-strict-lines` | Возвращает старый мягкий crop через bbox/fallback. По умолчанию crop строгий: обе линии обязательны. |
| `--baseline-line-pad` | Запас для строгого crop сверху и снизу как доля высоты строки. `0.08` оставляет примерно 8% высоты с каждой стороны, `0` отключает относительный запас. |
| `--baseline-line-pad-px` | Абсолютный запас в пикселях исходной картинки, добавляется к `--baseline-line-pad`. Полезно, если линии найдены слишком близко к буквам. |
| `--baseline-detector-checkpoint` | Optional checkpoint нейронного top/bottom baseline-детектора. Если задан, `--baseline-crop` использует его вместо эвристики по маскам. |
| `--baseline-detector-threshold` | Порог sigmoid heatmap для колонок верхней/нижней линии нейронного baseline-детектора. |
| `--baseline-rectify` | `lines` оставляет старый режим двух прямых линий. `curved` использует neural heatmap как две кривые top/bottom и делает dewarp строки по ним. |
| `--baseline-curve-smooth-radius` | Радиус сглаживания кривых top/bottom перед dewarp. Больше значение спокойнее, но может съедать резкие локальные изгибы. |
| `--baseline-curve-min-coverage` | Минимальная доля X-колонок, где neural heatmap уверенно нашел каждую линию. Если coverage ниже, curved-режим откатывается к `lines`. |
| `--baseline-top-pad` | Верхний запас для старого мягкого baseline crop. В строгом режиме используйте `--baseline-line-pad`. |
| `--baseline-bottom-pad` | Нижний запас для старого мягкого baseline crop. В строгом режиме используйте `--baseline-line-pad`. |
| `--no-baseline-deskew` | Отключает поворот по найденной baseline, но оставляет сам crop включенным. |
| `--baseline-max-angle` | Максимальный допустимый угол baseline. В строгом режиме слишком большой угол отключает baseline crop; в мягком режиме используется fallback. |

#### Cut Projection Segmentator

Эти параметры относятся к сегментатору с `loss_mode: cut_projection`, где сеть
выдает одну heatmap/projection-оценку cut-линии на X-позицию.

| Параметр | Что делает |
| --- | --- |
| `--segmentator-checkpoint` | Checkpoint вертикального сегментатора. Без него сегментатор не запускается. |
| `--segmentator-cut-threshold` | Порог для основных cut peak-ов. Если score выше порога, peak становится cut-точкой. |
| `--segmentator-peak-min-distance` | Минимальная дистанция между raw peak-ами при первичном выборе пиков. |
| `--segmentator-cut-postprocess` | Постобработка cut-пиков: `peaks` оставляет найденные пики, `widths` дополнительно контролирует ширины интервалов. |
| `--segmentator-cut-min-width` | Минимальная ширина символного интервала между соседними cut-точками после postprocess. Если cut-линии слишком близко, более слабая удаляется. |
| `--segmentator-cut-max-width` | Максимальная допустимая ширина интервала. Если интервал шире, postprocess пытается вставить недостающий cut из кандидатов. `0` отключает вставку. |
| `--segmentator-cut-candidate-threshold` | Нижний порог candidate peak-ов, из которых можно вставлять недостающие cut-точки при `widths`. |
| `--segmentator-cut-smooth-radius` | Радиус треугольного сглаживания cut-score перед поиском peak-ов. |

Важно: `--segmentator-peak-min-distance` и `--segmentator-cut-min-width`
похожи, но отвечают за разные места пайплайна. Первый ограничивает
дистанцию между raw peak-ами при выборе пиков. Второй ограничивает ширину
готовых символных интервалов после postprocess `widths`. Если
`--segmentator-cut-min-width` не передан явно, используется значение
`segmentator_cut_min_width` из checkpoint-конфига, а если его нет —
`segmentator_peak_min_distance`.

#### Legacy OCR + Segmentator Decode

Эти параметры используются только если включен `--decode-with-segmentator`.

| Параметр | Что делает |
| --- | --- |
| `--decode-with-segmentator` | Декодирует OCR не по схлопыванию одинаковых timestep-ов, а по интервалам между cut-точками сегментатора. |
| `--segmentator-decode-top-k` | Сколько OCR class-кандидатов показывать для каждого интервала `legacy+cuts`. |
| `--segmentator-boundary-cuts` | Как трактовать первую/последнюю cut-линию: `auto`, `on`, `off`. `auto` чинит случаи вида `|A|B|C|`, где крайние линии являются границами текста. |
| `--segmentator-boundary-cut-max-edge-ratio` | Порог для auto-режима boundary cuts относительно типичной ширины символного интервала. |
| `--segmentator-edge-min-width` | Сливает слишком узкие крайние интервалы с соседними, чтобы не получать лишние символы по краям. |
| `--segmentator-edge-min-ink-ratio` | Минимальная доля foreground-колонок, чтобы крайний интервал считался настоящим символом. |
| `--segmentator-edge-min-pixel-density` | Минимальная плотность foreground-пикселей, чтобы крайний интервал считался настоящим символом. |
| `--no-segmentator-edge-trim` | Отключает удаление пустых крайних интервалов. Полезно для диагностики, но обычно ухудшает края. |

### Baseline Detector

`baseline_crop` сейчас работает как ensemble, а не как одна эвристика:

1. Строятся несколько кандидатов текстовой маски:
   `otsu`, `clahe_otsu`, `adaptive`, `morph_contrast`.
2. Каждая маска чистится по connected components: мелкий мусор и длинные
   тонкие линии отбрасываются.
3. Для каждой очищенной маски берутся несколько вариантов точек нижней линии:
   нижние профили `lower_q80`, `lower_q88`, `lower_q94`, `lower_edge`, а также
   `component_bottoms` по нижним точкам компонент.
4. Для каждого набора точек robust/RANSAC-фитом ищется линия `y = ax + b`.
   Считаются `angle`, `inlier_ratio`, `profile_coverage`, `residual_mad`,
   `residual_rmse` и итоговый `confidence`.
5. Для выбранного нижнего кандидата отдельно ищется верхняя текстовая линия
   по верхним профилям `upper_edge`, `upper_q04`, `upper_q08`, `upper_q14`.
   Она тоже фитится robust/RANSAC и получает свои confidence/coverage/residual.
6. В строгом режиме, включенном по умолчанию, нижняя и верхняя линии одинаково
   обязательны: кандидат без надежной верхней линии отбрасывается, а итоговый
   confidence пары берется как минимум из confidence нижней и верхней линий.
   Старый мягкий режим с `bbox_fallback` можно вернуть через
   `--no-baseline-strict-lines`.
7. Если `baseline_deskew` включен и найденный угол заметный, картинка
   поворачивается на этот угол, фон новых полей заполняется медианным цветом
   рамки, после чего baseline ищется еще раз на повернутом изображении.
8. В строгом режиме crop строится после поворота только по паре верх/низ:
   верхняя граница берется по верхней линии, нижняя - по нижней линии, без
   расширения через bbox текста. `baseline_line_pad` добавляет небольшой
   симметричный запас относительно `max(расстояние между линиями, bbox-высота
   foreground)`, а `baseline_line_pad_px` добавляет гарантированный пиксельный
   запас. Если после поворота пару линий найти не удалось, baseline crop не
   применяется. В мягком режиме остается прежний crop по линиям плюс bbox и
   `baseline_top_pad`/`baseline_bottom_pad`.

В `--debug-image` для baseline показываются overlay с нижней красной и верхней
синей линиями, inlier-точки,
очищенная маска, crop, а в текстовом блоке пишутся `baseline method`,
`baseline mask`, число кандидатов, angle, confidence, topline stats и crop box.

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
