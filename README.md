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

Основные конфиги для эксперимента `101` теперь разделены по назначению:

- `synth_generators/line_generator/configs/eng_101.yaml` — OCR: чистая основная строка, `dense_targets`, без соседних строк;
- `synth_generators/line_generator/configs/eng_101_cuts.yaml` — cuts-сегментатор: чистая основная строка, `cut_projection_targets`, без соседних строк;
- `synth_generators/line_generator/configs/eng_101_baselines.yaml` — baseline detector: `baseline_targets`, соседние верхние/нижние строки как вертикальный мусор.

Старые минимальные примеры также остаются:

- `synth_generators/line_generator/configs/eng_001.yaml` — генерация;
- `configs/train/eng_train_001.yaml` — обучение.

Конфиги в корне проекта сгруппированы по назначению:

- `configs/train/` — обучение моделей;
- `configs/evaluation/` — evaluation, Optuna и совместный train/evaluation;
- `configs/inference/` — полный пайплайн инференса.

Все относительные пути внутри YAML считаются от папки самого конфига. Параметры,
переданные в командной строке, переопределяют значения из evaluation-конфига.
Неизвестные ключи в generation, training, evaluation и inference YAML считаются ошибкой,
поэтому опечатка или удалённый параметр не могут быть молча проигнорированы.

Настраиваемые значения в evaluation-конфигах задаются в едином блоке
`parameters`. Обычное значение фиксирует параметр, а пара `[min, max]`
передает его Optuna для подбора:

```yaml
parameters:
  cut_threshold: [0.10, 0.95]
  cut_min_width: [1, 10]
  scale_x: 0.0
  baseline_crop: true
  baseline_deskew: [false, true]
```

В примере `cut_threshold` и `cut_min_width` подбираются, а `scale_x` и
`baseline_crop` зафиксированы. Булевый диапазон `[false, true]` включает подбор
переключателя, если evaluator поддерживает его оптимизацию. Служебные параметры
запуска (`json`, `checkpoint`, `optuna_trials`, пути вывода и подобные) остаются
на верхнем уровне конфига. CLI-флаги вида
`--optuna-cut-threshold-min` и `--optuna-cut-threshold-max` сохранены и могут
переопределить отдельную границу для конкретного запуска.

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
  vertical_fade: 0.25
  noise: 0.75
  projective: 0.12
  rotate: 0.8
  x_pad: 0.1
  crop_x: 0.08
  crop_y: 0.05
  rescale_quality: 0.2
  random_line: 0.1
  baseline_line: 0.1
  morphology: 0.08
  unsharp_mask: 0.12
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
  vertical_fade:
    side: random
    extent_min: 0.20
    extent_max: 0.55
    strength_min: 0.15
    strength_max: 0.65
    gamma_min: 0.7
    gamma_max: 1.8
  rotate:
    max_degrees: 1.0
    fillcolor: 255
  x_pad:
    pad_min: 0.02
    pad_max: 0.10
    pad_reference: content
    fill_mode: side_median
    resize_mode: bilinear
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
  baseline_line:
    side: random_or_both
    both_probability: 0.35
    top_y_min: 0.18
    top_y_max: 0.36
    bottom_y_min: 0.62
    bottom_y_max: 0.88
    angle_degrees_min: -2.5
    angle_degrees_max: 2.5
    line_width_min: 0.75
    line_width_max: 2.0
    alpha_min: 0.25
    alpha_max: 0.75
    value_min: 0.0
    value_max: 90.0
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
`strong_blur`, `motion_blur`, `scale`, `darkening`, `vertical_fade`, `noise`, `projective`,
`rotate`, `x_pad`, `crop_x`, `crop_y`, `rescale_quality`, `random_line`,
`baseline_line`, `morphology`, `unsharp_mask`, `brightness`, `contrast`, `invert`.
`preprocess_geometry` повторяет смысл inference-параметров `scale_x/y_pad`.
`x_pad` сжимает содержимое по X внутрь исходного размера тензора и заполняет
края. `fill_mode: side_median` независимо использует медианный цвет левого и
правого края исходной картинки и соответствует inference-preprocessing;
`fill_mode: constant` использует `fillcolor`. Для target-ов применяется такое
же геометрическое преобразование: новые области получают класс пробела для OCR
и ноль для cut/baseline-разметки. `pad_reference: content` трактует долю padding
от исходной ширины содержимого, как inference `x_pad`; legacy-вариант
`pad_reference: output` считает её от фиксированной ширины выходного тензора.
`crop_x` и `crop_y` обрезают края, а затем ресайзят результат обратно в
исходный размер тензора.
`rescale_quality` уменьшает картинку до доли `factor`, затем возвращает к
исходному размеру, чтобы имитировать потерю разрешения/JPEG-подобную грубость
без изменения геометрической разметки.
`vertical_fade` плавно смешивает верхнюю или нижнюю часть изображения с
медианным цветом его рамки. `extent` задаёт долю высоты эффекта, `strength`
силу у края, `gamma` форму перехода; геометрическая разметка не меняется.
`random_line` добавляет почти горизонтальную линию под небольшим углом.
`baseline_line` добавляет верхнюю, нижнюю или обе baseline-like линии.
`side` может быть `top`, `bottom`, `both`, `random` или `random_or_both`;
позиции задаются через `top_y_*` и `bottom_y_*` в долях высоты изображения.

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

Для раздельной генерации данных под OCR, cuts и baseline:

```bash
python -m synth_generators.line_generator.generate_dataset \
  --config synth_generators/line_generator/configs/eng_101.yaml

python -m synth_generators.line_generator.generate_dataset \
  --config synth_generators/line_generator/configs/eng_101_cuts.yaml

python -m synth_generators.line_generator.generate_dataset \
  --config synth_generators/line_generator/configs/eng_101_baselines.yaml
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
`N x C x H x W`) и исходные `texts` как текстовая разметка. Нужный тип
разметки включается в generation-конфиге отдельно:

```yaml
save_dense_targets: true
```

для OCR `legacy_logreg`,

```yaml
save_cut_projection_targets: true
cut_projection_peak_radius: 1
cut_projection_include_margins: true
```

для вертикального cuts-сегментатора,

```yaml
save_baseline_targets: true
baseline_target_radius: 1
```

для top/bottom baseline detector.

Тогда в чанки попадет `dense_targets` (`N x W`) — класс символа для
каждой X-колонки исходного кропа. Если включен `save_cut_projection_targets`,
в чанки попадет `cut_projection_targets` (`N x W`, `uint8`) — heatmap
правильных вертикальных разрезов. Каждый внутренний разрез ставится посередине
между видимыми границами соседних глифов, а не между их типографическими
advance-интервалами. Пробел не имеет видимого контура, поэтому для него
используется его логическая ширина; он по-прежнему считается отдельным символом.
`cut_projection_include_margins: true` добавляет левую и правую границы всей
строки. Они обязательны для декодирования `|A|B|C|`: четыре cut-линии задают
три символные ячейки.
При обучении loss делает crop и при необходимости пересэмплирует эту разметку
к выходной ширине сети `T`. Для
`vertical_segmentator_fcn` ширина выхода сохраняется 1:1, поэтому в конфиге
используются crop `0/0` и strict-width. Если включен `save_baseline_targets`, в чанки попадет
`baseline_targets` (`N x 2 x H x W`, `uint8`) — две горизонтальные heatmap-линии
для верхней и нижней границы основной строки. Их координаты вычисляются по
крайним ненулевым пикселям отдельной растровой маски основной строки, поэтому
метрики шрифта, фон и соседние мусорные строки на границы не влияют. Для
baseline-датасета можно
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
задает точное случайное расстояние в пикселях между основной строкой и
видимым фрагментом соседней строки. Это расстояние считается по фактическим
ненулевым пикселям растровой маски строки, поэтому разные шрифты могут давать
разную визуальную геометрию при одинаковых параметрах конфига. Если две
соседние строки не помещаются одновременно с заданными ограничениями по
высоте, crop-ratio и gap, генератор оставляет одну из них вместо
искусственного увеличения промежутков.

Для OCR и cuts-сегментатора можно дополнительно фильтровать реальные
горизонтальные расстояния между соседними видимыми non-space глифами после
рендера и после нарезки на кропы:

```yaml
ink_spacing_enabled: true
ink_spacing_min_char_gap_px: 0.0
ink_spacing_touch_gap_px: 0.5
ink_spacing_touch_probability: 0.10
```

`ink_spacing_min_char_gap_px: 0.0` запрещает перекрытие соседних глифов.
Пары через пробел не проверяются, потому что пробел является отдельной
логической ячейкой без собственного ink-bbox. Если минимальный gap в кропе не
больше `ink_spacing_touch_gap_px`, такой почти касающийся пример сохраняется
только с вероятностью `ink_spacing_touch_probability`; остальные варианты
пересэмплируются. Так можно сильно уменьшить долю неоднозначных разрезов, но
оставить немного сложных near-touch случаев для устойчивости.
Рядом создается `metadata.yaml` с параметрами датасета: алфавитом,
`space_char`, размерами картинок, числом каналов и
максимальной длиной текста. Настройки обучения и настройки аугментаций в
offline-датасет не сохраняются. `output_dir`, `chunk_size`, `num_workers` и
`overwrite` задаются в generation-конфиге. Датасет сохраняется в подпапку с
именем generation-конфига и временем запуска, например
`data/eng_001_20260607_153045`. Если `num_workers > 0`, чанки генерируются
параллельно. Offline-генерация сохраняет чистые строки без аугментаций.

В training-конфиге и `render_text.py` можно оставить логический путь без
timestamp, например `data/eng_001`: автоматически будет выбран самый свежий
завершённый каталог `eng_001_*`, содержащий `metadata.yaml`. Чтобы зафиксировать
конкретную версию данных, укажите её полное имя с timestamp.

В каталог каждого сгенерированного датасета также копируется исходный
generation-конфиг под фиксированным именем `generation_config.yaml`.

Посмотреть пример из чанка с теми же аугментациями, которые использует
обучение:

```bash
python synth_generators/line_generator/render_text.py \
  --chunks-dir data/eng_001 \
  --index 0 \
  --config configs/train/eng_train_001.yaml \
  --show-full-markup \
  --output output/render_chunk.png
```

`render_text` преобразует `dense_targets` вместе с изображением и под полем
`text` выводит поколоночную разметку как `␠[start:end]`. Крайние пробельные
классы также показываются в самом поле `text`. `--show-full-markup` дополнительно
рисует cut-линии зелёным, а верхнюю и нижнюю baseline — красным и синим.

Запустить обучение на синтетике:

```bash
python train.py --config configs/train/eng_train_001.yaml
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

Готовые примеры: `configs/train/eng_train_101_wide.yaml`,
`configs/train/eng_train_101_highres.yaml`,
`configs/train/eng_train_101_residual.yaml`,
`configs/train/eng_train_101_cuts_residual.yaml`.

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
```

Пример конфига: `configs/train/eng_train_101_cuts.yaml`.
Соответствующий generation-конфиг: `synth_generators/line_generator/configs/eng_101_cuts.yaml`.
Порог и параметры postprocess (`cut_threshold`, ограничения ширины и
сглаживание) в обучении не участвуют. Они задаются при
`evaluate_segmentator.py` и сохраняются в отдельном inference-конфиге.

Для обучения нейронного детектора верхней/нижней базовой линии:

```yaml
loss_mode: baseline_heatmap
baseline_heatmap_strict_size: true
baseline_heatmap_loss: bce
baseline_heatmap_positive_weight: 6.0
```

Пример конфига: `configs/train/eng_train_101_baselines.yaml`.
Соответствующий generation-конфиг:
`synth_generators/line_generator/configs/eng_101_baselines.yaml`.

Обучение baseline-детектора с оценкой на ручной разметке после каждой эпохи:

```bash
python train_baselines_with_eval.py \
  --config configs/evaluation/eng_train_101_baselines_eval.yaml
```

В `configs/evaluation/eng_train_101_baselines_eval.yaml` задаются:

- `train_config`: обычный training-конфиг с `loss_mode: baseline_heatmap`;
- `markup_json`: разметка из `tool.annotation_server`;
- `images_dir`: опциональная замена папки изображений из JSON;
- `threshold`: фиксированный порог детектора для сравнения эпох;
- `evaluate_every`: период оценки в эпохах;
- `best_metric`: метрика выбора лучшей модели;
- `optuna_trials`: опциональный подбор threshold на каждой оцениваемой эпохе.

Рекомендуемый обычный режим использует фиксированный `threshold` и
`optuna_trials: 0`: так изменение метрик отражает обучение модели, а не
изменение постобработки. Скрипт сохраняет:

- per-epoch CSV в `output_dir`;
- общий `eval_summary.tsv`;
- информацию о лучшем результате в `best_manual_baselines.json`;
- лучший по ручной метрике checkpoint как
  `best_manual_baselines_model.pth` в training checkpoint directory.

`best_model.pth` при этом по-прежнему выбирается по synthetic validation loss.
Это намеренно: ручная выборка служит отдельной внешней проверкой и не меняет
scheduler обучения.

Основная manual-метрика после каждой evaluation печатается отдельной строкой в
`stderr`, а полный лог обучения и evaluation остаётся в `stdout`. Поэтому
следующий запуск пишет подробный лог в файл, оставляя в терминале только числа
метрики:

```bash
python train_baselines_with_eval.py \
  --config configs/evaluation/eng_train_101_baselines_eval.yaml \
  > output/baseline_training.log
```

Для уже обученного cuts-чекпоинта параметры `cut_threshold`, `cut_min_width`,
`cut_max_width` и `cut_smooth_radius` задаются в разделе `segmentator`
inference-конфига.

Подобрать параметры вертикального сегментатора без OCR, сравнивая число
предсказанных ячеек с длиной строки из Label Studio:

```bash
python evaluate_segmentator.py \
  --config configs/evaluation/eng_101_segmentator.yaml \
  --json labels.json \
  --images images
```

Все диапазоны Optuna и фиксированные параметры берутся из YAML. Любой
переданный CLI-аргумент переопределяет соответствующее значение конфига.

При наличии внешних границ `|A|B|C|` оцениваемая длина считается как
`число cut-точек - 1`, поэтому пробел в разметке считается обычным символом.

### Ручная разметка cuts и baseline

Для точной оценки вертикального сегментатора и детектора baseline есть
браузерный инструмент:

```bash
python -m tool.annotation_server \
  --images /path/to/images \
  --output output/manual_markup.json \
  --open-browser
```

Без `--open-browser` сервис печатает адрес, который можно открыть вручную.
По умолчанию это `http://127.0.0.1:8765/`. Изображения обходятся в стабильном
алфавитном порядке, включая вложенные папки.

В интерфейсе есть три слоя:

- `Cuts`: вертикальные границы ячеек. Нужно разметить обе крайние границы и
  все границы между символами. Для `|A|B|C|` получится четыре линии и три
  ячейки. Пробел также является отдельной ячейкой.
- `Top`: верхняя baseline как полилиния минимум из двух точек.
- `Bottom`: нижняя baseline как полилиния минимум из двух точек.

Левый клик добавляет линию или точку, существующую точку baseline можно
перетаскивать. Правый клик удаляет ближайший элемент активного слоя.
Колесо мыши над изображением меняет zoom вокруг курсора. Разметка
автоматически сохраняется в JSON в координатах исходной картинки, а переход к
соседнему изображению всегда дожидается принудительного сохранения текущего
результата. Любая сохранённая пригодная разметка участвует в evaluation:
для cuts нужны минимум две вертикальные линии, для baseline нужны минимум две
точки у верхней и нижней полилинии.

Точная оценка cuts:

```bash
python evaluate_segmentator.py \
  --config configs/evaluation/eng_101_segmentator.yaml
```

Для `manual_markup.json` папка изображений автоматически берётся из поля
`images_root`. Параметр `--images` нужен только для переопределения сохранённого
пути или при использовании экспорта Label Studio.

Для ручного JSON дополнительно считаются `cut_precision`, `cut_recall`,
`cut_f1` и средняя ошибка совпавших линий по X. Предсказанные линии
возвращаются в координаты исходника через карту геометрии preprocessing,
включая baseline crop, deskew, padding и resize.
Label Studio JSON по-прежнему поддерживается для старой оценки по длине.

Оценка обеих baseline:

```bash
python evaluate_baselines.py \
  --config configs/evaluation/eng_101_baselines.yaml \
  --optuna-trials 0
```

Подбор threshold:

```bash
python evaluate_baselines.py \
  --config configs/evaluation/eng_101_baselines.yaml
```

`evaluate_baselines.py` считает MAE отдельно для верхней и нижней линии,
общую ошибку в пикселях, ошибку относительно высоты строки, покрытие по X и
штрафует неудачные детекции при Optuna-подборе.

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

Каждый новый запуск добавляет к настроенному `checkpoint_dir` дату и время.
Например, `checkpoint_dir: ../checkpoints/eng_101` создаст каталог
`checkpoints/eng_101_20260607_153045`. При `resume: true` автоматически
выбирается последний каталог с тем же базовым именем, содержащий
`latest_checkpoint.pth`.

Обучение пишет компактный лог по эпохам в `stdout` и TSV-файл внутри
timestamp-каталога:

```text
checkpoints/eng_101_20260607_153045/training_log.tsv
```

При старте туда же копируются:

```text
training_config.yaml
generation_config.yaml
```

Первый файл является снимком текущего training-конфига, второй берётся из
выбранного каталога чанков. Вместе с checkpoint и `metadata.yaml` датасета это
фиксирует конфигурацию генерации и обучения для воспроизведения эксперимента.
Старые датасеты без `generation_config.yaml` по-прежнему поддерживаются, но при
старте обучения выводится предупреждение.

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
  --config configs/inference/eng_101.yaml \
  --generation-config synth_generators/line_generator/configs/eng_101.yaml \
  --sample-index 0 \
  --save-sample output/synthetic.png
```

Обычный запуск и debug-картинка:

```bash
python inference.py \
  --config configs/inference/eng_101.yaml \
  --image path/to/line.png \
  --debug-image output/inference_debug.png
```

Все checkpoint-ы и параметры preprocessing находятся в inference-конфиге.
Пример: `configs/inference/eng_101.yaml`.

```yaml
device: cuda

baseline:
  enabled: true
  detector_checkpoint: ../../checkpoints/baseline_detector/best_model.pth
  detector_threshold: 0.35
  deskew: true
  max_angle: 12.0
  strict_lines: true
  line_pad: 0.08
  line_pad_px: 0.0

ocr:
  checkpoint: ../../checkpoints/eng_101/best_model.pth
  preprocessing:
    scale_x: 0.0
    y_pad: 0.4
    x_pad: 0.03
  decode:
    enabled: true
    top_k: 8
    center_fraction: 0.6
    min_score_width: 1

segmentator:
  checkpoint: ../../checkpoints/cut_segmentator/best_model.pth
  preprocessing:
    scale_x: 0.0
    y_pad: 0.0
    x_pad: 0.03
  cut_threshold: null
  cut_min_width: null
  cut_max_width: null
  cut_smooth_radius: null
```

`null` у параметра сегментатора означает: использовать значение из его
training-конфига, сохраненного в checkpoint.

Разделы `baseline`, `segmentator` и `ocr` независимы. Если раздел отсутствует,
соответствующий этап полностью пропускается и его checkpoint не загружается.
Дополнительно `baseline.enabled: false` отключает существующий раздел baseline.
Раздел `ocr.decode` по умолчанию выключен; `ocr.decode.enabled: true` требует
наличия `segmentator`.

Например, минимальный конфиг только для вертикальной сегментации:

```yaml
device: cuda
segmentator:
  checkpoint: ../../checkpoints/cut_segmentator/best_model.pth
  preprocessing:
    scale_x: 0.0
    y_pad: 0.0
    x_pad: 0.03
```

Минимальный конфиг только для baseline detector:

```yaml
device: cuda
baseline:
  enabled: true
  detector_checkpoint: ../../checkpoints/baseline_detector/best_model.pth
  detector_threshold: 0.35
```

Python API для использования из других скриптов:

```python
from fcn_ocr import OCRPipeline

pipeline = OCRPipeline("configs/inference/eng_101.yaml")
for path in ["line_1.png", "line_2.png"]:
    result = pipeline.recognize_path(path)
    print(path, result.text)
```

### Inference Pipeline

Инференс устроен как последовательный пайплайн из трех опциональных этапов:

1. Загружаются только checkpoint-ы из присутствующих разделов конфига. Из OCR checkpoint
   берутся `alphabet`, `architecture`,
   `num_classes`, `image_height`, `channels`, режим loss/target и параметры
   legacy crop. Модель создается через `fcn_architectures.create_model`,
   загружает `model_state_dict` и переводится в `eval`.
2. Входная картинка приводится к `RGB` или `L` в зависимости от `channels`.
   В `--debug-image` этот шаг подписан как `preprocess 00 input converted`.
3. Если присутствует включенный раздел `baseline`, один раз запускается общий поиск верхней и
   нижней текстовых линий. После deskew линии могут уточняться на повернутой
   картинке, затем строится единый baseline crop для всех следующих этапов.
   При отсутствии baseline исходное изображение передается дальше без crop.
4. Полученное изображение независимо обрабатывается присутствующими профилями
   `ocr.preprocessing` и/или `segmentator.preprocessing`.
5. `x_pad` применяется до `y_pad`, resize и `scale_x`. Он добавляет слева и
   справа долю текущей ширины, но не отражает символы: поля заполняются
   медианным фоном боковой полосы исходной геометрии. В debug это
   `preprocess 02 x-pad border median`.
6. `y_pad` добавляет или обрезает высоту. Положительное значение добавляет
   поля сверху/снизу, заполненные медианным цветом рамки текущей картинки;
   отрицательное значение симметрично режет высоту.
7. Каждая картинка приводится к `image_height` соответствующего checkpoint с сохранением
   пропорций по ширине.
8. `scale_x` применяется последним из геометрических inference-параметров:
   `0.2` растягивает ширину на 20%, `-0.2` сжимает на 20%.
9. Получившиеся картинки нормализуются и подаются только в включенные модели.
10. Cut-координаты сегментатора переводятся через карты исходных X-координат
    в систему OCR. Поэтому разные `x_pad` и `scale_x` не смещают ячейки.
11. Если присутствует раздел `ocr`, OCR FCN возвращает logits `B x C x T`.
12. Обычный OCR decode берет `argmax` по классам на каждом timestep, схлопывает
   подряд идущие одинаковые классы и переводит индексы в символы alphabet.
13. Если `ocr.decode.enabled: true`, OCR logits декодируются через
    интервалы между cut-точками сегментатора: для каждого интервала берется средняя
    вероятность OCR-классов, а top-класс становится символом. Первая и последняя
    cut-линии являются границами текста; внешние области не декодируются.

В `--debug-image` весь пайплайн расположен в трех столбцах:

1. `BASELINE DETECTION`: исходник, heatmap/линии, поворот, crop и общий результат.
2. `VERTICAL SEGMENTATION`: собственный preprocessing, вход сети и найденные cut-линии.
3. `OCR`: собственный preprocessing и изображение, непосредственно поданное в OCR.

Отсутствующий этап сохраняет свое место в канвасе и помечается `SKIPPED`, поэтому
результаты разных конфигураций удобно сравнивать визуально.

Параметры preprocessing:

- `scale_x`: нормированное растяжение/сжатие ширины после resize по высоте.
- `y_pad`: нормированный вертикальный padding/crop до resize по высоте.
- `x_pad`: нормированный горизонтальный padding до `y_pad`/resize/`scale_x`.
- `baseline.enabled`: включает общий поиск нижней и верхней текстовых линий, deskew и
  вертикальный crop.

### Inference Parameter Reference

#### OCR Preprocessing

| Параметр | Что делает |
| --- | --- |
| `ocr.preprocessing.scale_x` | Горизонтальный scale только для OCR. |
| `ocr.preprocessing.y_pad` | Вертикальный padding/crop только для OCR. |
| `ocr.preprocessing.x_pad` | Горизонтальный padding только для OCR. |
| `segmentator.preprocessing.scale_x` | Горизонтальный scale только для сегментатора. |
| `segmentator.preprocessing.y_pad` | Вертикальный padding/crop только для сегментатора. |
| `segmentator.preprocessing.x_pad` | Горизонтальный padding только для сегментатора. |
| `--show-raw` | Печатает raw timestep-классы обычного OCR decode. Полезно, чтобы увидеть, где сеть держит один класс несколько timestep-ов подряд. |
| `--debug-image` | Сохраняет трехколоночный канвас полного pipeline: baseline, segmentator, OCR, а ниже результаты и top-k confidence. |
| `debug.top_k` | Сколько top-кандидатов по confidence выводить для каждого decoded-символа. |

#### Baseline Crop

| Параметр | Что делает |
| --- | --- |
| `baseline.enabled` | Включает общий поиск линий, deskew и crop до раздельного preprocessing. |
| `baseline.strict_lines` | Требует надежную пару верхней/нижней линий; `false` разрешает bbox/fallback. |
| `baseline.line_pad` | Симметричный запас crop как доля высоты строки. |
| `baseline.line_pad_px` | Дополнительный абсолютный запас в исходных пикселях. |
| `baseline.detector_checkpoint` | Обязательный checkpoint нейронного top/bottom baseline-детектора. |
| `baseline.detector_threshold` | Порог sigmoid heatmap нейронного baseline-детектора. |
| `baseline.deskew` | Включает или отключает поворот по найденным линиям. |
| `baseline.max_angle` | Максимальный допустимый угол baseline. |

#### Cut Projection Segmentator

Эти параметры относятся к сегментатору с `loss_mode: cut_projection`, где сеть
выдает одну heatmap/projection-оценку cut-линии на X-позицию.

| Параметр | Что делает |
| --- | --- |
| `segmentator.checkpoint` | Checkpoint вертикального сегментатора. |
| `segmentator.cut_threshold` | Порог основных cut peak-ов. |
| `segmentator.cut_min_width` | Минимальная ширина между итоговыми cut-точками; из слишком близких пиков остается более уверенный. |
| `segmentator.cut_max_width` | Жесткая максимальная ширина; широкий внутренний интервал делится в наиболее уверенной допустимой X-позиции. `0` отключает вставку. |
| `segmentator.cut_smooth_radius` | Радиус сглаживания cut-score перед поиском peak-ов. |

Постобработка едина: scores сглаживаются, затем выбираются пики выше
`cut_threshold`, применяется `cut_min_width`, после чего при
`cut_max_width > 0` слишком широкие ячейки принудительно делятся. Если
`segmentator.cut_min_width: null`, используется значение из checkpoint или
дефолт `1`.

#### Legacy OCR + Segmentator Decode

Эти параметры используются только если `ocr.decode.enabled: true`.

| Параметр | Что делает |
| --- | --- |
| `ocr.decode.enabled` | Включает декодирование OCR по cut-ячейкам. |
| `ocr.decode.top_k` | Сколько OCR class-кандидатов хранить для каждой ячейки. |
| `ocr.decode.center_fraction` | Центральная доля ячейки для усреднения OCR-вероятностей. |
| `ocr.decode.min_score_width` | Минимальное число OCR timestep-ов в области оценки. |

### Baseline Detector

`baseline_crop` использует только нейросетевой top/bottom detector;
`baseline.detector_checkpoint` обязателен при `baseline.enabled: true`:

1. Изображение приводится к входной высоте detector с сохранением пропорций.
2. Сеть выдаёт две sigmoid heatmap: верхнюю и нижнюю границы текстовой строки.
3. Для каждого X берётся наиболее уверенная Y-позиция. Колонки ниже
   `baseline.detector_threshold` отбрасываются.
4. Для оставшихся точек каждой heatmap robust/RANSAC-фитом ищется линия
   `y = ax + b`. Считаются `inlier_ratio`, `profile_coverage`, residual и
   confidence.
5. Обе линии обязательны. Проверяется, что верхняя находится выше нижней и
   что их углы согласованы.
6. Угол deskew вычисляется по обеим линиям как взвешенное среднее их углов.
   Вес каждой линии равен `confidence * profile_coverage`. Если углы расходятся
   сильнее `max(2°, min(6°, baseline_max_angle / 2))`, строгая детекция
   отклоняется. Иначе картинка поворачивается на общий угол, фон новых полей
   заполняется медианным цветом рамки, после чего обе baseline ищутся ещё раз
   на повернутом изображении.
7. В строгом режиме crop строится после поворота только по паре верх/низ:
   верхняя граница берется по верхней линии, нижняя - по нижней линии, без
   расширения через bbox текста. `baseline_line_pad` добавляет небольшой
   симметричный запас относительно `max(расстояние между линиями, bbox-высота
   foreground)`, а `baseline_line_pad_px` добавляет гарантированный пиксельный
   запас. Если после поворота пару линий найти не удалось, baseline crop не
   применяется. При `strict_lines: false` crop может быть расширен bbox-областью
   самой нейросетевой heatmap, но эвристический detector не используется.

В `--debug-image` для baseline показываются overlay с нижней красной и верхней
синей линиями, heatmap mask и crop, а в текстовом блоке пишутся angle,
confidence, line-fit stats и crop box.

Если внешний скрипт лежит вне репозитория, добавьте корень проекта в
`PYTHONPATH`:

```bash
PYTHONPATH=/path/to/FCN-OCR-recognizer python my_script.py
```

Оценка Label Studio export JSON из корня проекта:

```bash
python evaluate_ocr.py \
  --config configs/evaluation/eng_101_ocr.yaml \
  --json path/to/export.json \
  --images path/to/images \
  --optuna-trials 0
```

Подбор inference-preprocessing через Optuna:

```bash
python evaluate_ocr.py \
  --config configs/evaluation/eng_101_ocr.yaml \
  --json path/to/export.json \
  --images path/to/images
```

`--scale-x`, `--y-pad`, `--x-pad` и соответствующие
`--optuna-...` диапазоны относятся только к OCR. Для вертикального
сегментатора используются независимые `--segmentator-scale-x`,
`--segmentator-y-pad`, `--segmentator-x-pad` и
`--optuna-segmentator-...` диапазоны. Диапазоны сегментатора участвуют в
Optuna только вместе с `--decode-with-segmentator` и
`--segmentator-checkpoint`.

`--batch-size` применяется как к обычному OCR evaluation, так и к совместному
OCR+segmentator decode. Для совместного запуска это верхняя граница размера
GPU-батча: изображения автоматически группируются по близкой ширине, поэтому
один широкий пример не заставляет дополнять весь батч до своей ширины. Перед
decode logits обрезаются до реальной выходной ширины каждого изображения.
Итоговый лог показывает фактический размер подбатчей и эффективность padding.

`configs/evaluation/eng_101_ocr.yaml` использует готовый
`configs/inference/eng_101.yaml`, поэтому baseline и вертикальный сегментатор
остаются зафиксированными, а Optuna подбирает только OCR preprocessing.
Эквивалентный запуск можно переопределять через CLI:

```bash
python evaluate_ocr.py \
  --config configs/evaluation/eng_101_ocr.yaml \
  --json path/to/export.json \
  --images path/to/images \
  --optuna-trials 500
```

В этом режиме из YAML берутся OCR checkpoint, baseline detector, вертикальный
сегментатор, их фиксированные preprocessing/postprocessing-параметры и `ocr.decode`.
Optuna меняет только параметры с явно переданными диапазонами. Явные обычные
CLI-параметры имеют приоритет над значениями YAML; например, `--device cuda`
можно использовать независимо от сохраненного в конфиге устройства.

После evaluation рядом с CSV сохраняется готовый inference-конфиг, например
`output/ocr_metrics.inference.yaml`, и в терминал выводится короткая команда
запуска с ним. `evaluate_segmentator.py` и `evaluate_baselines.py` делают так
же.

Если Optuna не установлена:

```bash
pip install optuna
```

Обучение с оценкой OCR после каждой эпохи:

```bash
python train_with_eval.py \
  --train-config configs/train/eng_train_101.yaml \
  --evaluation-config configs/evaluation/eng_101_ocr_per_epoch.yaml
```

Перед запуском достаточно один раз заполнить `json` и `images` в
`eng_101_ocr_per_epoch.yaml`. Все параметры evaluation берутся из этого файла,
а настройки моделей и preprocessing - из указанного в нём inference-конфига.
После каждой эпохи сохраняется текущий чекпоинт, запускается `evaluate_ocr`,
пишется per-epoch CSV и общий `eval_summary.tsv`. Копия evaluation-конфига
сохраняется рядом с чекпоинтами эксперимента. Для Optuna на каждой эпохе можно
создать отдельный вариант этого YAML с ненулевым `optuna_trials` и нужными
диапазонами.

Выбранная в конфиге `optuna_metric` печатается как одно число в `stderr`, весь
остальной лог идёт в `stdout`. Например:

```bash
python train_with_eval.py \
  --train-config configs/train/eng_train_101.yaml \
  --evaluation-config configs/evaluation/eng_101_ocr_per_epoch.yaml \
  > output/ocr_training.log
```
