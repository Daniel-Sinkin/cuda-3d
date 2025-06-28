#!/usr/bin/env python3
"""Convert a single PPM file to PNG, keeping the same name and directory."""

from pathlib import Path
import sys

from PIL import Image


def ppm_to_png(ppm_path: Path) -> Path:
    if not ppm_path.is_file():
        raise FileNotFoundError(f"{ppm_path} does not exist or is not a file")

    png_path = ppm_path.with_suffix(".png")

    with Image.open(ppm_path) as img:
        img.save(png_path, "PNG")

    return png_path


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python ppm_to_png.py <file.ppm>")

    ppm_file = Path(sys.argv[1]).resolve()
    try:
        png_file = ppm_to_png(ppm_file)
    except Exception as err:
        sys.exit(f"Conversion failed: {err}")

    print(f"Written {png_file}")


if __name__ == "__main__":
    main()
