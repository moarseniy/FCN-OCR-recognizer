from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from fcn_synth_generator.chunk_dataset import (
    load_torch_chunk,
    validate_chunk_payload,
)
from fcn_synth_generator.chunk_metadata import load_chunk_metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an FCN synthetic dataset chunk against metadata.yaml."
    )
    parser.add_argument(
        "path",
        help="Dataset directory or one chunk_XXXXXX.pt file.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate every manifest chunk when PATH is a dataset directory.",
    )
    return parser.parse_args(argv)


def _selected_chunk_paths(path: Path, validate_all: bool) -> tuple[Path, list[Path]]:
    path = path.expanduser().resolve()
    if path.is_file():
        return path.parent, [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    metadata = load_chunk_metadata(path)
    entries = metadata.chunks if validate_all else metadata.chunks[:1]
    return path, [path / entry.file for entry in entries]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source_path = Path(args.path)
    root, chunk_paths = _selected_chunk_paths(source_path, args.all)
    metadata = load_chunk_metadata(root)
    manifest = {entry.file: entry for entry in metadata.chunks}

    print(f"Dataset:  {root}")
    print(f"Task:     {metadata.task}")
    print(f"Samples:  {metadata.samples}")
    print(
        f"Image:    {metadata.channels}x{metadata.image_height}x"
        f"{metadata.image_width} uint8"
    )
    print(f"Target:   {metadata.target_dtype}")
    print(f"Chunks:   {metadata.chunk_count}")

    for chunk_path in chunk_paths:
        entry = manifest.get(chunk_path.name)
        if entry is None:
            raise ValueError(
                f"Chunk {chunk_path.name!r} is absent from {root / 'metadata.yaml'}"
            )
        chunk = load_torch_chunk(chunk_path)
        validate_chunk_payload(chunk, chunk_path, metadata, entry)
        images = chunk["images"]
        targets = chunk["targets"]
        print(
            f"OK {chunk_path.name}: samples={entry.samples}, "
            f"images={tuple(images.shape)} {images.dtype}, "
            f"targets={tuple(targets.shape)} {targets.dtype}"
        )

    if not args.all and len(metadata.chunks) > 1 and source_path.is_dir():
        print("Only the first chunk was checked; pass --all to validate the full dataset.")


if __name__ == "__main__":
    main()
