# gam

Streamlit apps for odds/vig + bet-back EV, with optional OCR to paste a screenshot and auto-fill odds.

## Local run (Windows)

Install deps:

- `pip install -r requirements.txt`

Run:

- Preferred (always uses the project venv): `./run_horse.ps1`
- Or: `C:\Users\logan\Documents\HT\gam\.venv\Scripts\python.exe -m streamlit run horse.py`

### OCR on Windows

`pytesseract` requires the **Tesseract OCR** executable.

- Recommended: `winget install --id UB-Mannheim.TesseractOCR -e`
- Then restart your terminal/VS Code.
- If OCR still can’t find it, set the path inside the app (OCR settings) or set env var `TESSERACT_CMD`.

If `tesseract.exe` is installed but not on PATH, the default location is usually:

- `C:\Program Files\Tesseract-OCR\tesseract.exe`

## Server-hosted / Linux deployment

### Streamlit Community Cloud-style deployment

This repo includes `packages.txt` with:

- `tesseract-ocr`

On Debian/Ubuntu-based hosts that support `packages.txt`, this installs the Tesseract binary so OCR works.

### Docker (recommended for any server)

Build:

- `docker build -t gam-ocr .`

Run:

- `docker run -p 8501:8501 gam-ocr`

Open:

- `http://localhost:8501`

Notes:
- OCR requires the Tesseract binary in the container/host. The Dockerfile installs it.
- If Tesseract is missing, the app will show an OCR error and still work for manual odds entry.
