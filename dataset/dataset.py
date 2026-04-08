import argparse
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def normalize_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    return [
        x1 / width,
        y1 / height,
        x2 / width,
        y2 / height,
    ]


def _maybe_resize(img: Image.Image, image_size: int | None) -> Image.Image:
    if image_size is None:
        return img
    im = img.copy()
    im.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    return im


def _save_image(img: Image.Image, path: Path, image_size: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = _maybe_resize(img, image_size)
    # Preserve alpha when present; JPEG would drop it.
    out.save(path, optimize=True)


def _media_path_for_json(saved: Path, output_json: Path) -> str:
    """Path of saved file relative to the JSON file's directory."""
    out_dir = output_json.parent.resolve()
    saved_r = saved.resolve()
    return Path(os.path.relpath(saved_r, out_dir)).as_posix()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=here / "genposter_subset.json",
        help="Output JSON path",
    )
    p.add_argument(
        "--images-dir",
        type=Path,
        default=here / "images",
        help="Directory for downloaded images (when --download-images)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Max training rows to scan from the streaming dataset",
    )
    p.add_argument(
        "--download-images",
        action="store_true",
        help="Download background and per-layer images (large disk use vs metadata-only)",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=None,
        metavar="N",
        help="If set, fit each image inside N×N (aspect preserved; PIL thumbnail)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = load_dataset(
        "creative-graphic-design/GenPoster100K", split="train", streaming=True
    )

    processed_data = []
    sid: str | int | None = None
    try:
        for i, sample in enumerate(tqdm(dataset, total=args.max_samples)):
            if i >= args.max_samples:
                break
            sid = sample.get("id")
            try:
                layer_columns = sample["layers"]
                bboxes = layer_columns["bbox"]
                n = len(bboxes)
                if n == 0:
                    continue

                width, height = layer_columns["psd_size"][0]
                labels = layer_columns.get("label", [""] * n)
                texts = layer_columns.get("text", [""] * n)
                font_sizes = layer_columns.get("font_size", [0] * n)
                layer_images = layer_columns.get("layer_image") if args.download_images else None

                elements = []
                for j in range(n):
                    bbox = bboxes[j]
                    if bbox is None:
                        continue
                    el = {
                        "type": labels[j] if j < len(labels) else "unknown",
                        "text": texts[j] if j < len(texts) else "",
                        "bbox": normalize_bbox(bbox, width, height),
                        "font_size": float(font_sizes[j])
                        if j < len(font_sizes) and font_sizes[j] is not None
                        else 0,
                    }
                    if args.download_images and layer_images is not None:
                        if j < len(layer_images) and layer_images[j] is not None:
                            layer_name = f"{sid}_layer_{len(elements)}.png"
                            layer_path = args.images_dir / layer_name
                            _save_image(layer_images[j], layer_path, args.image_size)
                            el["layer_image_path"] = _media_path_for_json(
                                layer_path, args.output
                            )
                    elements.append(el)

                if len(elements) == 0:
                    continue

                row = {
                    "id": sid,
                    "canvas_size": [width, height],
                    "elements": elements,
                }
                if args.download_images:
                    bg = sample.get("background_image")
                    if bg is not None:
                        bg_name = f"{sid}_bg.png"
                        bg_path = args.images_dir / bg_name
                        _save_image(bg, bg_path, args.image_size)
                        row["image_path"] = _media_path_for_json(bg_path, args.output)

                processed_data.append(row)

            except Exception:
                continue

    except KeyboardInterrupt:
        print(f"\nInterrupted after {len(processed_data)} samples (last id={sid!r}).", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(processed_data, f)

    print(f"Saved {len(processed_data)} samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
