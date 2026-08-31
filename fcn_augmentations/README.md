# FCN Augmentations

`fcn_augmentations` содержит единый Torch-пайплайн online-аугментаций. Он не
зависит от синтетического генератора и работает с батчами `B x C x H x W` в
диапазоне `[0, 1]` на том же устройстве, что и модель.

## Где настраивать

Аугментации задаются только в training YAML:

```yaml
gpu_augmentations: true
gpu_augment_val: false

augmentation_probabilities:
  preprocess_geometry: 0.35
  noise: 0.75
  rotate: 0.30
  x_pad: 0.50

augmentations:
  preprocess_geometry:
    scale_x_min: -0.18
    scale_x_max: 0.18
    y_pad_min: -0.08
    y_pad_max: 0.08
    fillcolor: 255
  noise:
    kind: gaussian
    std_min: 1.5
    std_max: 12.0
  rotate:
    max_degrees: 2.0
    fillcolor: 255
  x_pad:
    pad_min: 0.03
    pad_max: 0.25
    pad_reference: content
    fill_mode: side_median
    resize_mode: bilinear
```

Вероятность проверяется в диапазоне `[0, 1]`. Несколько преобразований могут
примениться к одному изображению в порядке `SUPPORTED_AUGMENTATIONS`. На новой
эпохе маски и параметры сэмплируются заново.

Параметр можно зафиксировать как `factor: 0.8` или сэмплировать равномерно как
`factor_min: 0.6` и `factor_max: 1.0`.

## Геометрия target

Для `cycle_shift`, `preprocess_geometry`, `scale`, `projective`, `rotate`,
`x_pad`, `crop_x` и `crop_y` target преобразуется той же геометрией, что и
изображение.

- FCN OCR использует nearest-перенос классов, новые области получают индекс
  `space_char`;
- vertical segmentation и baseline detection получают нули в новых областях;
- фотометрические преобразования target не меняют.

`augment_with_metadata` возвращает фактически выбранные параметры, а
`apply_metadata_to_targets` позволяет повторить ту же геометрию на другом
target-тензоре.

## Преобразования

| Имя | Основные параметры | Эффект |
| --- | --- | --- |
| `cycle_shift` | `max_x`, `max_y` | Циклический сдвиг по X/Y. |
| `preprocess_geometry` | `scale_x`, `y_pad`, `fillcolor` | Повторяет геометрию inference preprocessing. |
| `strong_blur` | `radius` | Сильный Gaussian blur. |
| `motion_blur` | `size`, `angle` | Линейное размытие движения. |
| `scale` | `factor`, `factor_x`, `factor_y`, `fillcolor` | Масштаб содержимого внутри прежнего размера. |
| `darkening` | `factor` | Только затемнение, типовой default меньше 1. |
| `vertical_fade` | `side`, `extent`, `strength`, `gamma` | Выцветание верхней или нижней части. |
| `noise` | `kind`, `std`, `amount` | Gaussian или salt-and-pepper noise. |
| `projective` | `max_dx`, `max_dy`, `fillcolor` | Небольшой shift/shear-подобный warp. |
| `rotate` | `max_degrees`, `fillcolor` | Случайный поворот. |
| `x_pad` | `pad`, `left`, `right`, `pad_reference`, `fill_mode` | Сжимает контент и добавляет боковые поля. |
| `crop_x` | `left`, `right` | Кроп по X с resize обратно. |
| `crop_y` | `top`, `bottom` | Кроп по Y с resize обратно. |
| `rescale_quality` | `factor`, `down_mode`, `up_mode` | Downscale и upscale без изменения размера target. |
| `random_line` | `angle_degrees`, `line_width`, `alpha`, `value`, `y` | Почти горизонтальная линия. |
| `baseline_line` | `side`, `both_probability`, позиции и стиль | Верхняя, нижняя или обе baseline-like линии. |
| `morphology` | `operation`, `size` | Erode, dilate или случайная морфология. |
| `unsharp_mask` | `radius`, `percent`, `threshold` | Усиление локальной резкости. |
| `brightness` | `factor` | Изменение яркости в обе стороны. |
| `contrast` | `factor` | Изменение контраста. |
| `invert` | нет | Инверсия интенсивности. |

Для каждого числового параметра из таблицы поддерживаются формы `name` и
`name_min`/`name_max`, если конкретное преобразование не описывает дискретный
параметр.

## X Pad

`x_pad` не увеличивает размер тензора. Он сжимает исходное содержимое по X,
оставляя поля внутри прежней ширины.

```yaml
x_pad:
  pad_min: 0.03
  pad_max: 0.20
  pad_reference: content
  fill_mode: side_median
  resize_mode: bilinear
```

- `pad_reference: content` считает долю от ширины исходного содержимого и
  соответствует inference `x_pad`;
- `pad_reference: output` считает долю от итоговой ширины;
- `fill_mode: side_median` независимо продолжает медианный цвет левого и
  правого края;
- `fill_mode: constant` использует `fillcolor`;
- новые OCR-колонки размечаются пробелом.

Вместо долей можно использовать `left_px`, `right_px` или `pad_px`.

## Линии и выцветание

Пример выцветания:

```yaml
vertical_fade:
  side: random
  extent_min: 0.20
  extent_max: 0.55
  strength_min: 0.15
  strength_max: 0.65
  gamma_min: 0.7
  gamma_max: 1.8
```

`side` принимает `top`, `bottom` или `random`. `extent` задает долю высоты,
`strength` интенсивность у края, `gamma` форму перехода.

Пример baseline-like линий:

```yaml
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
```

`side` принимает `top`, `bottom`, `both`, `random`, `random_or_both`.

## Просмотр результата

`render_text` может применить train-конфиг к сохраненному чанку и подписать
фактически использованные параметры:

```bash
python -m fcn_synth_generator.render_text \
  --chunks-dir data/eng_101 \
  --index 0 \
  --config configs/train/eng_train_101.yaml \
  --show-full-markup \
  --output output/augmented_sample.png
```

Для проверки реальных входов обучения используйте:

```yaml
preview_samples: 16
preview_dir: ../../output/input_previews
```

Preview сохраняется непосредственно перед `model(images)`, поэтому в нем
ровно тот тензор, который получила сеть.

## Python API

```python
from fcn_augmentations import AugmentationConfig, GpuTextAugmenter

config = AugmentationConfig.from_alphabet(
    alphabet=" AB",
    space_char=" ",
    probabilities={"noise": 0.5},
    parameters={"noise": {"std_min": 2.0, "std_max": 10.0}},
)
augmenter = GpuTextAugmenter(config)
images, targets = augmenter.augment_batch(images, targets, task="fcn_ocr")
```
