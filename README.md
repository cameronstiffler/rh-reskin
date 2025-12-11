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
