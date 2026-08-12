# FCN Architectures

Одна FCN-архитектура - один `.py` файл в этой папке.

Минимальный контракт:

```python
ARCHITECTURE_NAME = "my_fcn"


def create_model(in_channels: int, num_classes: int, **kwargs):
    return MyFCN(in_channels=in_channels, num_classes=num_classes, **kwargs)
```

После этого архитектуру можно выбрать в любом training-конфиге:

```yaml
architecture: my_fcn
architecture_params:
  hidden_channels: 32
```

Одна и та же архитектура может использоваться и для OCR, и для вертикального
сегментатора. Это определяется не файлом архитектуры, а training-конфигом:

- FCN OCR: `loss_mode: fcn_ocr`
- вертикальный сегментатор разрезов: `loss_mode: cut_projection`
- детектор верхней/нижней базовой линии: `loss_mode: baseline_heatmap`

Поле `architecture` обязательно. Алиасы и неявная архитектура по умолчанию не
поддерживаются.

Текущие встроенные варианты:

- `fcn_ocr` - исходная OCR-архитектура с компактной геометрией выхода.
- `fcn_ocr_highres` - plain FCN без горизонтального
  stride=2 в `conv2`; для кропа 48x64 дает `T=48` вместо `T=19`.
- `fcn_ocr_wide` - тот же набор kernel/stride, но с увеличенным числом
  каналов через `width_multiplier`; это прямой более тяжёлый эксперимент
  для OCR, потому что ширина выхода совпадает с `fcn_ocr`.
- `vertical_segmentator_fcn` - легкая width-preserving сеть для cut projection.
- `baseline_detector_fcn` - height/width-preserving сеть с выходом
  `B x 2 x H x W` для top/bottom baseline heatmap.
- `residual_temporal_fcn` - более тяжелая width-preserving FCN с residual-блоками
  и dilated temporal convolutions по X; подходит и для OCR, и для cut projection.

Примеры конфигов:

- `configs/train/eng_train_101_wide.yaml`
- `configs/train/eng_train_101_highres.yaml`
- `configs/train/eng_train_101_residual.yaml`
- `configs/train/eng_train_101_cuts_residual.yaml`
- `configs/train/eng_train_101_baselines.yaml`
