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

- legacy OCR: `loss_mode: legacy_logreg`, `legacy_target_mode: dense_symbols`
- вертикальный сегментатор разрезов: `loss_mode: cut_projection`

Старые конфиги и checkpoint без поля `architecture` используют `legacy_fcn`.

Текущие встроенные варианты:

- `legacy_fcn` - исходная архитектура со старой геометрией выхода.
- `legacy_fcn_wide` - тот же набор kernel/stride, но с увеличенным числом
  каналов через `width_multiplier`; это самый безопасный drop-in эксперимент
  для OCR, потому что ширина выхода совпадает с `legacy_fcn`.
- `vertical_segmentator_fcn` - легкая width-preserving сеть для cut projection.
- `residual_temporal_fcn` - более тяжелая width-preserving FCN с residual-блоками
  и dilated temporal convolutions по X; подходит и для OCR, и для cut projection.

Примеры конфигов:

- `configs/eng_train_101_wide.yaml`
- `configs/eng_train_101_residual.yaml`
- `configs/eng_train_101_cuts_residual.yaml`
