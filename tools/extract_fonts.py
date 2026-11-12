import argparse
import shutil
from pathlib import Path


def extract_fonts(input_dir, output_dir):
    src = Path(input_dir).resolve()
    dst = Path(output_dir).resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"Папка не найдена: {src}")

    if dst == src or src in dst.parents:
        raise ValueError("Выходная папка не должна находиться внутри исходной")

    dst.mkdir(parents=True, exist_ok=True)

    fonts = sorted(
        p for p in src.rglob("*")
        if p.is_file() and p.suffix.lower() in {".ttf", ".otf"}
    )

    for font in fonts:
        # Уникальное имя на основе относительного пути
        name = "__".join(font.relative_to(src).parts)
        shutil.copy2(font, dst / name)

    print(f"Найдено шрифтов: {len(fonts)}")
    print(f"Скопировано в: {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)

    args = parser.parse_args()
    extract_fonts(args.input, args.output)
