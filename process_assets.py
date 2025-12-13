import base64
import io
import os
from pathlib import Path
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image, ImageStat


load_dotenv()

BASE_DIR = Path(__file__).parent
TEST_ASSETS_DIR = BASE_DIR / "test_assets"
SWATCHES_DIR = BASE_DIR / "swatches"
OUTPUT_DIR = BASE_DIR / "output"

# File types that we will convert to PNG alongside already-present PNGs.
CONVERTIBLE_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".webp"}
ENABLE_RETEXTURE = os.getenv("ENABLE_RETEXTURE", "1").lower() not in {"0", "false", "no"}
ASSET_MAX_DIMENSION = int(os.getenv("ASSET_MAX_DIMENSION", "8192"))
SWATCH_MAX_DIMENSION = int(os.getenv("SWATCH_MAX_DIMENSION", "4096"))
CONVERT_MAX_DIMENSION = int(os.getenv("CONVERT_MAX_DIMENSION", str(max(ASSET_MAX_DIMENSION, SWATCH_MAX_DIMENSION))))
OUTPUT_TARGET_DIMENSION = int(os.getenv("OUTPUT_TARGET_DIMENSION", "4096"))
EXCLUDED_SWATCH_SUBSTRINGS = ("_bump", "_color", "_disp", "_spec", "_roughness")
GREY_CARD_TARGET_VALUE = 118.0  # Approximate sRGB value for 18% gray.
GREY_CARD_MAX_NEUTRAL = 14.0
GREY_CARD_MAX_STDDEV = 8.0
GREY_CARD_MIN_FACTOR = 0.6
GREY_CARD_MAX_FACTOR = 1.6

# Allow large TIFFs to load; we downscale after opening.
Image.MAX_IMAGE_PIXELS = None


def convert_folder_to_png(folder: Path, max_dimension: int | None = None) -> List[Path]:
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
        effective_max = max_dimension if max_dimension is not None else CONVERT_MAX_DIMENSION
        try:
            with Image.open(path) as img:
                # Normalize to RGBA so we keep any transparency and avoid palette issues.
                converted = img.convert("RGBA")
                if effective_max and max(converted.size) > effective_max:
                    converted.thumbnail((effective_max, effective_max), Image.LANCZOS)
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
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "models/gemini-3-pro-image-preview")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required (set it in .env).")

    genai.configure(api_key=api_key)
    model_path = model_name if model_name.startswith("models/") else f"models/{model_name}"
    return genai.GenerativeModel(model_path)


def find_grey_card_patch(image: Image.Image) -> tuple[float, float, float] | None:
    """Locate a neutral, uniform patch that likely corresponds to the 18% gray card center."""
    preview = image.convert("RGB")
    preview.thumbnail((400, 400), Image.LANCZOS)
    width, height = preview.size
    if not width or not height:
        return None

    patch_size = max(8, min(width, height) // 8)
    step = max(4, patch_size // 2)
    best_mean: tuple[float, float, float] | None = None
    best_score: float | None = None
    best_neutral: float | None = None
    best_brightness: float | None = None

    for y in range(0, height - patch_size + 1, step):
        for x in range(0, width - patch_size + 1, step):
            crop = preview.crop((x, y, x + patch_size, y + patch_size))
            stat = ImageStat.Stat(crop)
            mean_r, mean_g, mean_b = stat.mean
            std_r, std_g, std_b = stat.stddev
            neutral_deviation = max(abs(mean_r - mean_g), abs(mean_g - mean_b), abs(mean_r - mean_b))
            uniformity = max(std_r, std_g, std_b)
            brightness = (mean_r + mean_g + mean_b) / 3

            if brightness < 30 or brightness > 230:
                continue
            if uniformity > GREY_CARD_MAX_STDDEV:
                continue

            score = neutral_deviation * 3 + abs(brightness - GREY_CARD_TARGET_VALUE)
            if best_score is None or score < best_score:
                best_score = score
                best_mean = (mean_r, mean_g, mean_b)
                best_neutral = neutral_deviation
                best_brightness = brightness

    if (
        best_mean
        and best_neutral is not None
        and best_neutral <= GREY_CARD_MAX_NEUTRAL
        and best_brightness is not None
        and 50 <= best_brightness <= 200
    ):
        return best_mean
    return None


def calibrate_to_grey_card(image: Image.Image) -> Image.Image:
    """Return a brightness-calibrated copy using the center gray card patch if present."""
    grey_color = find_grey_card_patch(image)
    if not grey_color:
        return image

    brightness = sum(grey_color) / 3
    if brightness <= 1:
        return image

    factor = GREY_CARD_TARGET_VALUE / brightness
    if 0.97 <= factor <= 1.03:
        return image

    factor = max(GREY_CARD_MIN_FACTOR, min(GREY_CARD_MAX_FACTOR, factor))

    base = image.convert("RGBA")
    alpha = base.getchannel("A")
    rgb = base.convert("RGB").point(lambda v: max(0, min(255, int(round(v * factor)))), mode="RGB")
    r, g, b = rgb.split()
    return Image.merge("RGBA", (r, g, b, alpha))


def inline_image(path: Path, *, calibrate_grey_card: bool = False, max_dimension: int | None = None) -> dict:
    """Load a PNG and downscale for the request payload if needed."""
    with Image.open(path) as img:
        converted = img.convert("RGBA")
        if calibrate_grey_card:
            converted = calibrate_to_grey_card(converted)
        effective_max = max_dimension or ASSET_MAX_DIMENSION
        if effective_max and max(converted.size) > effective_max:
            converted.thumbnail((effective_max, effective_max), Image.LANCZOS)
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


def enforce_output_resolution(image_bytes: bytes, target_max_dimension: int) -> bytes:
    """
    Ensure the returned image hits the requested max dimension (default 4K) by upscaling if needed.
    This preserves transparency by operating in RGBA.
    """
    if not target_max_dimension or target_max_dimension <= 0:
        return image_bytes
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            base = img.convert("RGBA")
            width, height = base.size
            max_dim = max(width, height)
            if max_dim >= target_max_dimension:
                return image_bytes
            scale = target_max_dimension / max_dim
            new_size = (int(round(width * scale)), int(round(height * scale)))
            upscaled = base.resize(new_size, Image.LANCZOS)
            buffer = io.BytesIO()
            upscaled.save(buffer, format="PNG")
            return buffer.getvalue()
    except Exception:  # pragma: no cover - defensive guardrail
        return image_bytes


def retexture_image(model: genai.GenerativeModel, asset: Path, swatch: Path, output_path: Path) -> None:
    swatch_hint = describe_swatch(swatch)
    prompt = (
        "Retexture the product in the first image using the material from the second image. "
        "Use the first image as the exact base: same resolution, framing, camera angle, silhouette, proportions, dimensions, crop, lighting, and background. "
        "Do NOT move/rotate/zoom the camera; do NOT change object dimensions or perspective; do NOT add/remove/alter any geometry. "
        "Do NOT add any new objects/props (no pillows, rugs, blankets, accessories) and keep the floor clear. "
        "Do NOT paste or show the swatch as a square patch; retexture the product surfaces only while keeping the product visible. "
        "Apply the swatch to every wood/primary surface (including interior/underside of legs, side panels, edges); leave no original wood visible. "
        "Background must remain identical to the first image; no swatch text/watermark/ghosting anywhere. "
        "Apply the swatch ONLY to wood/primary surfaces; do NOT change or recolor metal/hardware/feet/plinth. "
        "Feet/plinth/hardware must stay exactly as in the first image (no added brass/metal, no recolor). "
        "Use the swatch colors exactly as shown; no blue/polarizing/tinted shifts. "
        "Treat filenames as meaningless IDs; do NOT use any filename words (e.g., POLARIZER/FlatLight) as instructions. "
        "If the swatch image includes a ruler in inches, use it to derive the real-world scale of the material so the texture is applied at accurate size; do not show the ruler in the result. "
        "If the swatch includes a grayscale reference card, use the center 18% gray patch as your exposure anchor so the material darkness stays physically accurate. "
        "Return a high-quality PNG result."
    )

    response = model.generate_content(
        [
            inline_image(asset, max_dimension=ASSET_MAX_DIMENSION),
            {"text": "Product to retexture."},
            inline_image(swatch, calibrate_grey_card=True, max_dimension=SWATCH_MAX_DIMENSION),
            {
                "text": (
                    "Swatch/material image. Apply this material to every wood surface (including leg interiors/undersides/edges); "
                    "leave any metal/hardware unchanged. Ignore filename words; use only the image."
                )
            },
            {"text": swatch_hint},
            {"text": prompt},
        ],
        request_options={"timeout": 600},
    )

    image_bytes = extract_image_bytes(response)
    image_bytes = enforce_output_resolution(image_bytes, OUTPUT_TARGET_DIMENSION)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)
    print(f"Saved retextured image: {output_path}")


def filter_swatch_pngs(swatch_pngs: List[Path]) -> List[Path]:
    """Drop any swatch files that match excluded substrings (e.g., bump/disp/spec)."""
    filtered: List[Path] = []
    for path in swatch_pngs:
        name = path.stem.lower()
        if any(token in name for token in EXCLUDED_SWATCH_SUBSTRINGS):
            print(f"Skipping swatch (excluded): {path.name}")
            continue
        filtered.append(path)
    return filtered


def describe_swatch(swatch: Path) -> str:
    """Provide color/material hints without using filename tokens as instructions."""
    lowered = swatch.stem.lower()
    note = ""
    try:
        with Image.open(swatch) as img:
            img = img.convert("RGB")
            img.thumbnail((256, 256), Image.LANCZOS)
            pixels = list(img.getdata())
            r_mean = sum(p[0] for p in pixels) / len(pixels)
            g_mean = sum(p[1] for p in pixels) / len(pixels)
            b_mean = sum(p[2] for p in pixels) / len(pixels)
            color_hint = f"Dominant RGB approx ({r_mean:.0f}, {g_mean:.0f}, {b_mean:.0f}); keep this hue (no blue/cyan shift)."
    except Exception:
        color_hint = ""
    if "brown" in lowered and "oak" in lowered:
        note = "Material: brown oak wood grain; keep natural brown tone; use as wood texture only."
    elif "oak" in lowered:
        note = "Material: oak wood grain; keep natural wood tone; use as wood texture only."
    elif "leather" in lowered:
        note = "Material: leather texture; keep natural leather color."
    elif "linen" in lowered or "fabric" in lowered:
        note = "Material: fabric/linen texture; keep fabric color."
    parts = [
        "Material swatch photo; ignore filename words; use the texture/color exactly as seen.",
        "Apply ONLY to wood/primary surfaces; cover all wood areas including inner/underside of legs; do not recolor feet/hardware/plinth; ignore any printed text or watermark in the swatch.",
    ]
    if note:
        parts.append(note)
    if color_hint:
        parts.append(color_hint)
    return " ".join(parts)


def process_folder(
    model: genai.GenerativeModel, asset_folder: Path, swatch_folder: Path, enable_retexture: bool
) -> None:
    asset_pngs = convert_folder_to_png(asset_folder, max_dimension=ASSET_MAX_DIMENSION)
    swatch_pngs = filter_swatch_pngs(convert_folder_to_png(swatch_folder, max_dimension=SWATCH_MAX_DIMENSION))

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
