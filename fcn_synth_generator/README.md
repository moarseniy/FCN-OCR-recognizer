# FCN Synth Generator

`fcn_synth_generator` создает чистые синтетические датасеты для трех задач:

- `fcn_ocr`: класс символа для каждой X-колонки;
- `vertical_segmentation`: heatmap вертикальных границ символных ячеек;
- `baseline_detection`: две 2D heatmap верхней и нижней границ строки.

Генератор не применяет обучающие аугментации. Он отвечает только за рендер,
разметку и сохранение воспроизводимых `uint8`-чанков. Онлайн-аугментации
описаны в [`fcn_augmentations/README.md`](../fcn_augmentations/README.md).

## Быстрый старт

```bash
python -m fcn_synth_generator.generate_dataset \
  --config fcn_synth_generator/configs/eng_101.yaml
```

Для каждой задачи есть отдельный пример:

```text
configs/eng_101.yaml
configs/eng_101_vertical_segmentation.yaml
configs/eng_101_baseline_detection.yaml
```

Результат сохраняется в timestamp-каталог с именем конфига, например
`data/eng_101_20260831_120000`. В него входят:

```text
chunk_000000.pt
chunk_000001.pt
metadata.yaml
generation_config.yaml
```

`generation_config.yaml` является точной копией исходного конфига.

## Формат чанка

Каждый `chunk_*.pt` содержит только:

- `images`: `uint8`, форма `N x C x H x W`;
- `texts`: исходные строки;
- `targets`: один target, формат которого определяется `task`.

`metadata.yaml` фиксирует задачу, алфавит, размеры, dtype, статистику классов
и manifest чанков. Обучение берет контракт данных только из metadata.

Проверка датасета:

```bash
python check_chunk.py data/eng_101_YYYYMMDD_HHMMSS
python check_chunk.py data/eng_101_YYYYMMDD_HHMMSS --all
```

## Алфавит и пробел

`alphabet` задает точный порядок классов. `space_char` должен входить в него:

```yaml
alphabet: " -0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
space_char: " "
```

Сгенерированные строки не начинаются и не заканчиваются пробелом, а несколько
пробелов подряд схлопываются. В dense OCR-target пробел остается обычным
классом. Фон до видимой границы первого глифа и после последнего также
размечается пробелом.

## Режимы target

Для OCR:

```yaml
task: fcn_ocr
```

`targets` имеет форму `N x W` и dtype `int16`. Каждая колонка содержит индекс
символа из `alphabet`.

Для вертикальной сегментации:

```yaml
task: vertical_segmentation
vertical_segmentation_target_radius: 1
vertical_segmentation_include_margins: true
```

`targets` имеет форму `N x W` и dtype `uint8`. Внутренний cut ставится по
центру между видимыми границами соседних глифов. При
`vertical_segmentation_include_margins: true` добавляются обе внешние границы:
разметка `|A|B|C|` задает три ячейки.

Для базовых линий:

```yaml
task: baseline_detection
baseline_detection_target_radius: 1
```

`targets` имеет форму `N x 2 x H x W` и dtype `uint8`. Канал 0 хранит верхнюю,
канал 1 нижнюю линию основной строки.

## Строки и кропы

`line_crops: true` сначала рендерит длинную строку из случайных слов, затем
последовательно нарезает ее на кропы `image_width`:

```yaml
line_crops: true
word_count_min: 6
word_count_max: 18
word_length_min: 2
word_length_max: 8
crop_stride: 64
min_crop_text_length: 1
```

Граничные фрагменты управляются параметрами:

```yaml
edge_char_min_visible_ratio: 0.75
edge_fragment_max_visible_ratio: 0.25
```

Достаточно видимый символ сохраняет свой класс. Маленький обрезанный фрагмент
служит краевым шумом и размечается пробелом.

Интервалы выбираются один раз на всю строку, поэтому стиль внутри строки
остается единым:

```yaml
char_spacing_min: -0.4
char_spacing_max: 1.6
word_spacing_multiplier_min: 0.75
word_spacing_multiplier_max: 1.7
```

`ink_spacing_*` фильтрует физический зазор между видимыми non-space глифами:

```yaml
ink_spacing_enabled: true
ink_spacing_min_char_gap_px: 0.0
ink_spacing_touch_gap_px: 0.5
ink_spacing_touch_probability: 0.10
```

## Соседние строки

Для baseline detection можно независимо добавить обрезанный верхний и нижний
текстовый сосед:

```yaml
neighbor_lines_probability: 0.85
neighbor_line_min_crop_ratio: 0.65
neighbor_line_visible_ratio_min: 0.25
neighbor_line_gap_min: 0
neighbor_line_gap_max: 8
main_line_y_jitter_px: 2
```

Основная строка сначала центрируется независимо от соседей.
`main_line_y_jitter_px` задает ее максимальное случайное отклонение вверх или
вниз; значение `0` включает строгое центрирование.
`neighbor_lines_probability` применяется отдельно к каждой стороне.
`neighbor_line_min_crop_ratio: 0.65` оставляет видимым не более 35% соседней
строки. Gap измеряется между фактическими видимыми пикселями. Если выбранный
сосед при фиксированном положении основной строки не удовлетворяет одновременно
gap и ограничениям видимости, он пропускается, а основная строка не сдвигается.

## Шрифты и фоны

Пути считаются относительно generation YAML:

```yaml
font_dir: ../fonts
font_extensions: [.ttf, .otf, .ttc, .otc]
background_dir: ../backgrounds
background_extensions: [.png, .jpg, .jpeg, .bmp, .webp]
```

Fonts check оставляет только шрифты, покрывающие весь `alphabet`, и печатает
статистику принятых и отклоненных файлов. Для установки надежной проверки:

```bash
python -m pip install fonttools
```

Размер текста задается долей высоты изображения, а не размером шрифта в
типографских пунктах:

```yaml
main_text_height_ratio_min: 0.50
main_text_height_ratio_max: 0.78
```

Для изображения высотой 48 пикселей это означает целевую высоту видимого
контура алфавита примерно 24-37 пикселей. Генератор отдельно калибрует каждый
шрифт, поэтому два шрифта с одинаковым визуальным размером больше не расходятся
из-за разных внутренних метрик. Размер не зависит от конкретного текста строки:
`IIII` и `QMW0` используют одну и ту же типографскую шкалу.

Фон только кропается. Resize фона не применяется; длинная подложка собирается
из нескольких crop-only фрагментов.

Визуальная проверка всех принятых шрифтов:

```bash
python -m fcn_synth_generator.font_validation \
  --config fcn_synth_generator/configs/eng_101.yaml
```

`--include-rejected` дополнительно отрисует шрифты, не прошедшие проверку.

## Render Text

Сгенерировать одну чистую строку:

```bash
python -m fcn_synth_generator.render_text \
  --text "012 345" \
  --config fcn_synth_generator/configs/eng_101.yaml \
  --output output/synthetic_line.png
```

В режиме `--text` вся строка должна целиком помещаться в `image_width` с
учетом `horizontal_padding`. Увеличение `main_text_height_ratio_*` увеличивает
также естественную ширину глифов. Для просмотра крупного текста используйте
короткую строку, увеличьте `image_width` либо смотрите уже нарезанные
`line_crops` через `--chunks-dir`.

Посмотреть элемент чанка с аугментациями training-конфига:

```bash
python -m fcn_synth_generator.render_text \
  --chunks-dir data/eng_101 \
  --index 0 \
  --config configs/train/eng_train_101.yaml \
  --show-full-markup \
  --output output/render_chunk.png
```

Generation YAML всегда дает чистый render. Training YAML разрешен только
вместе с `--chunks-dir` и подключает его online-аугментации. Флаг
`--no-augmentations` отключает их. `--show-full-markup` показывает доступный
target: cuts или обе baseline. Геометрические аугментации применяются к
изображению и target одинаково.

## Производительность

```yaml
samples: 100000
chunk_size: 1024
num_workers: 4
output_dir: ../../data
overwrite: false
```

При `num_workers > 0` работа распараллеливается по чанкам. Каждый worker
использует собственный seed, а сохраненные изображения остаются `uint8`.
