# FCN-OCR-recognizer

Полносверточный пайплайн распознавания текстовых строк переменной ширины.
Проект объединяет три независимые FCN-задачи:

1. `baseline_detection` находит верхнюю и нижнюю границы строки.
2. `vertical_segmentation` находит границы символных ячеек.
3. `fcn_ocr` классифицирует символы по X и декодирует их внутри найденных ячеек.

Все стадии опциональны в inference-конфиге. Baseline detection при совместном
запуске вычисляется один раз, после чего его результат используется
вертикальной сегментацией и OCR.

## Документация

- [Синтетический генератор](fcn_synth_generator/README.md): generation YAML,
  шрифты, фоны, targets, чанки и `render_text`.
- [GPU-аугментации](fcn_augmentations/README.md): online-пайплайн,
  преобразования, параметры и синхронная геометрия target.
- [FCN-архитектуры](fcn_architectures/README.md): реестр моделей и добавление
  новых архитектур.

Конфиги сгруппированы по назначению:

```text
configs/train/       обучение
configs/inference/   полный runtime pipeline
configs/evaluation/  метрики и Optuna
```

Неизвестные ключи во всех YAML запрещены. Старые форматы данных и checkpoint
не мигрируются: эксперимент должен использовать текущий строгий контракт.

## Быстрый старт

Создать чистые данные:

```bash
python -m fcn_synth_generator.generate_dataset \
  --config fcn_synth_generator/configs/eng_101.yaml
```

Обучить модель:

```bash
python train.py --config configs/train/eng_train_101.yaml
```

Запустить полный inference:

```bash
python inference.py \
  --config configs/inference/eng_101.yaml \
  --image path/to/line.png \
  --debug-image output/inference_debug.png
```

Оценить или подобрать параметры:

```bash
python evaluate.py fcn_ocr \
  --config configs/evaluation/eng_101_ocr.yaml

python evaluate.py vertical_segmentation \
  --config configs/evaluation/eng_101_vertical_segmentation.yaml

python evaluate.py baseline_detection \
  --config configs/evaluation/eng_101_baseline_detection.yaml
```

## Структура кода

Runtime OCR находится в `fcn_ocr/`:

- `checkpoint.py` проверяет checkpoint и создает FCN;
- `preprocessing.py` выполняет padding, resize и `scale_x`;
- `baseline_processing.py` обрабатывает две baseline, deskew и crop;
- `cut_processing.py` сопоставляет cuts с выходом OCR;
- `decoding.py` содержит raw, cells и DP decode;
- `pipeline.py` координирует все три стадии;
- `evaluation/` содержит общие загрузчики, метрики, Optuna и reporting.

Обучение находится в `fcn_training/`. Задачи из `fcn_training/tasks/` одним
контрактом задают target, число выходов, loss и допустимые training-параметры.
Корневые `train.py`, `inference.py` и `evaluate.py` являются тонкими CLI.

## Dataset Contract

`metadata.yaml` обязателен для каждого offline-датасета. Он фиксирует:

- `task`, алфавит и индекс пробела;
- размеры и dtype images/targets;
- manifest чанков;
- статистику текста и OCR-классов;
- параметры геометрической разметки задачи.

Training YAML не может переопределять эти поля. Проверить датасет:

```bash
python check_chunk.py data/eng_101_YYYYMMDD_HHMMSS
python check_chunk.py data/eng_101_YYYYMMDD_HHMMSS --all
```

## Обучение

Training-конфиг содержит только параметры модели, оптимизации, загрузчика и
online-аугментаций. Алфавит, каналы и размеры приходят из metadata.

```yaml
chunks_dir: ../../data/eng_101
architecture: fcn_ocr
architecture_params: {}
task: fcn_ocr

epochs: 50
batch_size: 1024
batch_count: 100
lr: 0.001
optimizer: adam
scheduler: reduce_on_plateau

num_workers: 0
chunk_cache_size: 2
chunk_aware_batches: true

gpu_augmentations: true
gpu_augment_val: false
```

Поддерживаются `adam`, `adamw`, `sgd`, `rmsprop` и scheduler-ы `none`,
`reduce_on_plateau`, `cosine`, `step`. `batch_count` ограничивает эпоху
фиксированным числом случайных batch-ей; `null` проходит весь train split.

Каждый запуск создает timestamp-каталог checkpoint. В него сохраняются:

```text
latest_checkpoint.pth
best_model.pth
training_log.tsv
training_config.yaml
generation_config.yaml
alphabet_stats.tsv
```

`resume: true` выбирает последний совместимый запуск с тем же базовым именем.

Примеры реальных входов сети сохраняются непосредственно перед forward:

```yaml
preview_samples: 16
preview_dir: ../../output/input_previews
```

## Inference

Минимальная структура полного pipeline:

```yaml
device: cuda

baseline_detection:
  enabled: true
  detector_checkpoint: ../../checkpoints/baseline_detection/best_model.pth
  detector_threshold: 0.35
  deskew: true
  max_angle: 12.0
  line_pad: 0.08
  line_pad_px: 0.0

vertical_segmentation:
  checkpoint: ../../checkpoints/vertical_segmentation/best_model.pth
  preprocessing:
    scale_x: 0.0
    y_pad: 0.0
    x_pad: 0.03
  cut_threshold: 0.5
  cut_min_width: 3
  cut_max_width: 30
  cut_smooth_radius: 1

fcn_ocr:
  checkpoint: ../../checkpoints/eng_101/best_model.pth
  preprocessing:
    scale_x: 0.0
    y_pad: 0.4
    x_pad: 0.03
  decode:
    enabled: true
    method: dp
    top_k: 8
    center_fraction: 0.6
    min_score_width: 1
```

Отсутствующий раздел полностью пропускает соответствующую стадию.
`fcn_ocr.decode.enabled: true` требует vertical segmentation.

`--debug-image` сохраняет три столбца: baseline detection, vertical
segmentation и OCR. В каждом показаны собственный preprocessing, вход сети и
результат стадии.

Python API:

```python
from fcn_ocr import FCNPipeline

pipeline = FCNPipeline("configs/inference/eng_101.yaml")
result = pipeline.recognize_path("line.png")
print(result.text)
```

Для внешнего скрипта добавьте корень репозитория в `PYTHONPATH`.

## Evaluation и Optuna

Evaluation YAML содержит данные, inference config, метрику и блок
`parameters`. Скаляр фиксирует параметр, `[min, max]` передает его Optuna:

```yaml
json: /path/to/labels.json
images: /path/to/images
inference_config: ../inference/eng_101.yaml

parameters:
  scale_x: [-0.25, 0.25]
  y_pad: 0.4
  x_pad: [0.0, 0.12]

optuna_trials: 200
optuna_metric: global_char_accuracy
optuna_seed: 0
optuna_image_cache_mb: 512
```

Checkpoint загружается один раз на study. Декодированные изображения
кешируются с лимитом RAM. Neural-output cache переиспользует baseline heatmap
или segmentation logits, когда параметры до соответствующей сети фиксированы.

Persistent study через SQLite нужен только для продолжения и истории trials.
У study строгий контракт checkpoint, датасета, метрики и search space. После
их изменения задайте новое `optuna_study_name`.

После evaluation сохраняются CSV, TSV trials и готовый inference YAML с
лучшими параметрами. Основная метрика печатается в `stderr`, обычный лог в
`stdout`.

## Ручная разметка

Браузерный разметчик cuts и обеих baseline:

```bash
python -m tools.annotation.server \
  --images /path/to/images \
  --output output/manual_markup.json \
  --open-browser
```

`Cuts` содержит обе внешние границы и внутренние границы ячеек. `Top` и
`Bottom` являются полилиниями минимум из двух точек. Переход к другой картинке
автоматически сохраняет текущую разметку. Полученный JSON напрямую принимают
`evaluate.py vertical_segmentation` и `evaluate.py baseline_detection`.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

Characterization-тесты фиксируют dense OCR targets, edge spaces, синхронную
геометрию аугментаций, baseline crop, cuts, cells/DP decode и parity между
inference и evaluation.
