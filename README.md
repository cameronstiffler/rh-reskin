# Surface Swap Test Harness

Process product assets with material swatches using the Gemini image model, with optional retexturing.

## Prerequisites
- Python 3.10+ installed.
- Dependencies installed: `pip install -r requirements.txt` (or `pip install google-generativeai pillow python-dotenv` if no requirements file).
- `.env` file containing `GEMINI_API_KEY=...` (copy from `.env.example` as a template). Optional: `ENABLE_RETEXTURE=1`, `REQUEST_MAX_DIMENSION=2048`.

## Run
From the project root:
```sh
python3 process_assets.py
```
The script converts assets/swatch images to PNG as needed, calibrates swatch brightness using the 18% gray card if present, and writes outputs to `output/`.

Notes:
- Product requests stay at up to 8192px on the longest side by default (`ASSET_MAX_DIMENSION` env) so we submit the highest detail the API will accept; swatches are downscaled separately (`SWATCH_MAX_DIMENSION`, default 4096) to balance payload size with detail.
- Source conversions honor `CONVERT_MAX_DIMENSION` (default: max of asset/swatch limits) so TIFFs/JPEGs become high-res PNGs before sending; assets convert up to `ASSET_MAX_DIMENSION`, swatches up to `SWATCH_MAX_DIMENSION`.
- Outputs are upscaled to 4K if needed (`OUTPUT_TARGET_DIMENSION`, default 4096) to maintain high-res deliverables.
- `ENABLE_RETEXTURE=0` skips generation while still running conversions.
