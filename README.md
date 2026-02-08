# Dither-Dawg

Desktop dither tool for images, GIFs, and videos with palette controls, presets, and live preview.

**Features**
- Import images, GIFs, and videos.
- Multiple ordered and error-diffusion dithering styles.
- Built-in palettes plus palette import (`.ase`, `.gpl`, `.hex`, `.txt`).
- Presets save/load (`.dset`).
- Export to image, GIF, or video.
- GitHub update check (optional).

**Requirements**
- Python 3.12 (recommended)
- Windows (current UI layout is Windows-first, but the code is cross-platform)

**Quick Start**
1. Create and activate a virtual environment.
2. Install dependencies.
3. Run the app.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

**Controls**
- Mouse wheel over preview: zoom
- `Ctrl+=`: zoom in
- `Ctrl+-`: zoom out
- `Ctrl+0`: reset zoom

**Palettes & Presets**
- Palettes live in `palettes`. Use `Extras > Import Palette` or drop files in the folder.
- Presets live in `presets`. Use `File > Export Preset` / `File > Import Preset`.

**Update Check**
- The app checks GitHub Releases on launch if `GITHUB_REPO` is set in `app.py`.
- Disable checks by setting `DITHER_DAWG_DISABLE_UPDATE_CHECK=1`.

**Project Layout**
- UI and app logic: `app.py`
- Dithering core: `dither_core.py`
- Starter image: `dither-dawg-starter-image.png`

**License**
See `LICENSE`.
