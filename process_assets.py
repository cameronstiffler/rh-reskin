import base64
import io
import os
from pathlib import Path
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image


BASE_DIR = Path(__file__).parent
TEST_ASSETS_DIR = BASE_DIR / "test_assets"
SWATCHES_DIR = BASE_DIR / "swatches"
OUTPUT_DIR = BASE_DIR / "output"

# File types that we will convert to PNG alongside already-present PNGs.
CONVERTIBLE_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".webp"}
MAX_DIMENSION = 4096
ENABLE_RETEXTURE = os.getenv("ENABLE_RETEXTURE", "1").lower() not in {"0", "false", "no"}
REQUEST_MAX_DIMENSION = int(os.getenv("REQUEST_MAX_DIMENSION", "2048"))

# Allow large TIFFs to load; we downscale after opening.
Image.MAX_IMAGE_PIXELS = None


def convert_folder_to_png(folder: Path) -> List[Path]:
    """Convert images in a folder to PNG and return the PNG paths."""
    pngs: List[Path] = []
    seen: set[Path] = set()
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".png":
            if path not in seen:
                pngs.append(path)
                seen.add(path)
            continue
        if suffix not in CONVERTIBLE_EXTENSIONS:
            continue

        output_path = path.with_suffix(".png")
        if output_path.exists():
            if output_path not in seen:
                pngs.append(output_path)
                seen.add(output_path)
            continue
        try:
            with Image.open(path) as img:
                # Normalize to RGBA so we keep any transparency and avoid palette issues.
                converted = img.convert("RGBA")
                if max(converted.size) > MAX_DIMENSION:
                    converted.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                converted.save(output_path)
            if output_path not in seen:
                pngs.append(output_path)
                seen.add(output_path)
            print(f"Converted to PNG: {path} -> {output_path}")
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[WARN] Failed to convert {path} to PNG: {exc}")
    return pngs


def configure_model() -> genai.GenerativeModel:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "models/gemini-3-pro-image-preview")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required (set it in .env).")

    genai.configure(api_key=api_key)
    model_path = model_name if model_name.startswith("models/") else f"models/{model_name}"
    return genai.GenerativeModel(model_path)


def inline_image(path: Path) -> dict:
    """Load a PNG and downscale for the request payload if needed."""
    with Image.open(path) as img:
        converted = img.convert("RGBA")
        if REQUEST_MAX_DIMENSION and max(converted.size) > REQUEST_MAX_DIMENSION:
            converted.thumbnail((REQUEST_MAX_DIMENSION, REQUEST_MAX_DIMENSION), Image.LANCZOS)
        buffer = io.BytesIO()
        converted.save(buffer, format="PNG")
        return {"mime_type": "image/png", "data": buffer.getvalue()}


def extract_image_bytes(response) -> bytes:
    # The image is returned as inline_data; decode if base64-encoded.
    for candidate in getattr(response, "candidates", []):
        for part in getattr(candidate.content, "parts", []):
            inline_data = getattr(part, "inline_data", None)
            if not inline_data:
                continue
            data = inline_data.data
            if isinstance(data, bytes):
                return data
            if isinstance(data, str):
                return base64.b64decode(data)
    raise RuntimeError("No inline image data found in the response.")


def retexture_image(model: genai.GenerativeModel, asset: Path, swatch: Path, output_path: Path) -> None:
    prompt = (
        "Retexture the product in the first image using the material from the second image. "
        "Preserve the product's shape, lighting, camera angle, and proportions. "
        "Apply the swatch as the new surface finish realistically. Return a high-quality PNG result."
    )

    response = model.generate_content(
        [
            inline_image(asset),
            {"text": "Product to retexture."},
            inline_image(swatch),
            {"text": "Swatch/material to apply."},
            {"text": prompt},
        ],
        request_options={"timeout": 600},
    )

    image_bytes = extract_image_bytes(response)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)
    print(f"Saved retextured image: {output_path}")


def process_folder(
    model: genai.GenerativeModel, asset_folder: Path, swatch_folder: Path, enable_retexture: bool
) -> None:
    asset_pngs = convert_folder_to_png(asset_folder)
    swatch_pngs = convert_folder_to_png(swatch_folder)

    if not asset_pngs:
        print(f"[WARN] No assets to process in {asset_folder}")
        return
    if not swatch_pngs:
        print(f"[WARN] No swatches to process in {swatch_folder}")
        return

    if not enable_retexture:
        print(f"Skipping retexture for {asset_folder.name} because ENABLE_RETEXTURE is off.")
        return

    output_folder = OUTPUT_DIR / asset_folder.name
    output_folder.mkdir(parents=True, exist_ok=True)

    for asset in asset_pngs:
        for swatch in swatch_pngs:
            output_path = output_folder / f"{asset.stem}__{swatch.stem}.png"
            if output_path.exists():
                print(f"Skipping existing output: {output_path}")
                continue
            try:
                retexture_image(model, asset, swatch, output_path)
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"[WARN] Failed to retexture {asset.name} with {swatch.name}: {exc}")


def main() -> None:
    model = configure_model()

    for asset_folder in TEST_ASSETS_DIR.iterdir():
        if not asset_folder.is_dir():
            continue
        swatch_folder = SWATCHES_DIR / asset_folder.name
        if not swatch_folder.exists():
            print(f"[WARN] No matching swatch folder for {asset_folder.name}, skipping.")
            continue

        print(f"Processing folder: {asset_folder.name}")
        process_folder(model, asset_folder, swatch_folder, ENABLE_RETEXTURE)


if __name__ == "__main__":
    main()
