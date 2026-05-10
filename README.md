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
