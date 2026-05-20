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

- OCR CTC: `loss_mode: ctc`
- legacy OCR: `loss_mode: legacy_logreg`, `legacy_target_mode: dense_symbols`
- вертикальный сегментатор: `loss_mode: legacy_logreg`, `legacy_target_mode: binary_gaps`
- вертикальный сегментатор разрезов: `loss_mode: cut_projection`

Старые конфиги и checkpoint без поля `architecture` используют `legacy_fcn`.
