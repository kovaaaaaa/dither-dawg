from __future__ import annotations

import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageSequence
from PySide6.QtCore import QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QBrush, QColor, QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)



def pil_to_qimage(image: Image.Image) -> QImage:
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    data = image.tobytes("raw", "RGBA")
    qimage = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
    return qimage.copy()


@dataclass(frozen=True)
class Palette:
    name: str
    colors: list[tuple[int, int, int]]


BUILTIN_PALETTES = {
    "Mono": [(0, 0, 0), (255, 255, 255)],
    "CRT Green": [(0, 0, 0), (64, 255, 144), (12, 72, 40)],
    "CRT Amber": [(0, 0, 0), (255, 176, 64), (96, 48, 8)],
    "Bubblegum": [(34, 24, 68), (120, 88, 255), (227, 116, 255), (255, 230, 248)],
    "CGA": [(0, 0, 0), (85, 255, 255), (255, 85, 255), (255, 255, 255)],
    "Ice": [(10, 14, 26), (52, 86, 128), (170, 210, 255), (240, 248, 255)],
}

MODULATION_METHODS = {
    "Modulated Diffuse Y",
    "Modulated Diffuse X",
    "Modulated Smooth Diffuse",
}


def _palette_luminance(color: tuple[int, int, int]) -> float:
    r, g, b = color
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _sorted_palette(colors: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    return sorted(colors, key=_palette_luminance)


def _build_palette_data(colors: list[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray, int]:
    sorted_colors = _sorted_palette(colors)
    palette_array = np.array(sorted_colors, dtype=np.uint8)
    palette_luma = np.array([_palette_luminance(c) for c in sorted_colors], dtype=np.float32)
    levels = max(2, len(sorted_colors))
    return palette_array, palette_luma, levels


def _read_u16(stream) -> int:
    return struct.unpack(">H", stream.read(2))[0]


def _read_u32(stream) -> int:
    return struct.unpack(">I", stream.read(4))[0]


def _read_f32(stream) -> float:
    return struct.unpack(">f", stream.read(4))[0]


def _read_ase_string(stream) -> str:
    length = _read_u16(stream)
    if length == 0:
        return ""
    raw = stream.read(length * 2)
    return raw.decode("utf-16be", errors="ignore").rstrip("\x00")


def _lab_to_rgb(l_value: float, a_value: float, b_value: float) -> tuple[int, int, int]:
    y = (l_value + 16.0) / 116.0
    x = a_value / 500.0 + y
    z = y - b_value / 200.0

    def pivot(t: float) -> float:
        if t**3 > 0.008856:
            return t**3
        return (t - 16.0 / 116.0) / 7.787

    x = 95.047 * pivot(x)
    y = 100.0 * pivot(y)
    z = 108.883 * pivot(z)

    x /= 100.0
    y /= 100.0
    z /= 100.0

    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570

    def gamma(u: float) -> float:
        if u <= 0.0031308:
            return 12.92 * u
        return 1.055 * (u ** (1.0 / 2.4)) - 0.055

    r = gamma(max(0.0, min(1.0, r)))
    g = gamma(max(0.0, min(1.0, g)))
    b = gamma(max(0.0, min(1.0, b)))

    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def _parse_ase(path: Path) -> list[tuple[int, int, int]]:
    colors: list[tuple[int, int, int]] = []
    try:
        with path.open("rb") as handle:
            if handle.read(4) != b"ASEF":
                return colors
            _ = _read_u16(handle)
            _ = _read_u16(handle)
            blocks = _read_u32(handle)
            for _ in range(blocks):
                block_type = _read_u16(handle)
                block_length = _read_u32(handle)
                data = handle.read(block_length)
                if block_type != 0x0001:
                    continue
                buffer = memoryview(data)
                offset = 0

                name_len = struct.unpack(">H", buffer[offset : offset + 2])[0]
                offset += 2 + name_len * 2
                model = bytes(buffer[offset : offset + 4]).decode("ascii", errors="ignore")
                offset += 4

                def read_f32_local() -> float:
                    nonlocal offset
                    value = struct.unpack(">f", buffer[offset : offset + 4])[0]
                    offset += 4
                    return value

                if model == "RGB ":
                    r = read_f32_local()
                    g = read_f32_local()
                    b = read_f32_local()
                    colors.append((int(round(r * 255)), int(round(g * 255)), int(round(b * 255))))
                elif model == "CMYK":
                    c = read_f32_local()
                    m = read_f32_local()
                    y = read_f32_local()
                    k = read_f32_local()
                    r = 1.0 - min(1.0, c + k)
                    g = 1.0 - min(1.0, m + k)
                    b = 1.0 - min(1.0, y + k)
                    colors.append((int(round(r * 255)), int(round(g * 255)), int(round(b * 255))))
                elif model in ("Gray", "GRAY"):
                    gray = read_f32_local()
                    value = int(round(gray * 255))
                    colors.append((value, value, value))
                elif model == "LAB ":
                    l_value = read_f32_local() * 100.0
                    a_value = read_f32_local()
                    b_value = read_f32_local()
                    colors.append(_lab_to_rgb(l_value, a_value, b_value))
    except Exception:
        return []

    return colors


def _parse_gpl(path: Path) -> list[tuple[int, int, int]]:
    colors: list[tuple[int, int, int]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("Name:") or line.startswith("Columns:"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            if all(part.isdigit() for part in parts[:3]):
                r, g, b = (int(parts[0]), int(parts[1]), int(parts[2]))
                colors.append((r, g, b))
    except Exception:
        return []
    return colors


def _parse_hex_text(text: str) -> list[tuple[int, int, int]]:
    colors: list[tuple[int, int, int]] = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if raw.startswith("0x"):
            raw = raw[2:]
        if raw.startswith("#"):
            raw = raw[1:]
        if len(raw) < 6:
            continue
        raw = raw[:6]
        try:
            r = int(raw[0:2], 16)
            g = int(raw[2:4], 16)
            b = int(raw[4:6], 16)
            colors.append((r, g, b))
        except ValueError:
            continue
    return colors


def _parse_hex_file(path: Path) -> list[tuple[int, int, int]]:
    try:
        return _parse_hex_text(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []


def _load_palette_file(path: Path) -> Palette | None:
    suffix = path.suffix.lower()
    colors: list[tuple[int, int, int]] = []
    if suffix == ".ase":
        colors = _parse_ase(path)
    elif suffix == ".gpl":
        colors = _parse_gpl(path)
    elif suffix in {".hex", ".txt"}:
        colors = _parse_hex_file(path)
    if len(colors) < 2:
        return None
    return Palette(path.stem, colors)


def load_palette_library(palette_dir: Path) -> dict[str, list[Palette]]:
    palettes: dict[str, list[Palette]] = {}
    palette_dir.mkdir(parents=True, exist_ok=True)

    root_palettes: list[Palette] = []
    for entry in sorted(palette_dir.iterdir(), key=lambda p: p.name.lower()):
        if entry.is_dir():
            category = entry.name
            for file_path in sorted(entry.iterdir(), key=lambda p: p.name.lower()):
                if not file_path.is_file():
                    continue
                palette = _load_palette_file(file_path)
                if palette:
                    palettes.setdefault(category, []).append(palette)
        elif entry.is_file():
            palette = _load_palette_file(entry)
            if palette:
                root_palettes.append(palette)

    if root_palettes:
        palettes["Built-in"] = root_palettes
    else:
        palettes["Built-in"] = [Palette(name, colors) for name, colors in BUILTIN_PALETTES.items()]

    return palettes


def apply_image_adjustments(
    img: Image.Image,
    brightness: float,
    contrast: float,
    blur_radius: float,
    sharpen_strength: int,
) -> Image.Image:
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    if sharpen_strength > 0:
        for _ in range(int(sharpen_strength)):
            img = img.filter(ImageFilter.SHARPEN)
    return img


def _ordered_dither_indices(gray, matrix, levels: int, xp=np):
    h, w = gray.shape
    m_h, m_w = matrix.shape
    tiled = xp.tile(matrix, (h // m_h + 1, w // m_w + 1))[:h, :w]
    norm = gray / 255.0
    jitter = (tiled - 0.5) / levels
    indices = xp.clip((norm + jitter) * levels, 0, levels - 1).astype(xp.int32)
    return indices


def _error_diffuse_indices(
    gray: np.ndarray,
    palette_luma: np.ndarray,
    kernel: list[tuple[int, int, float]],
    serpentine: bool = True,
) -> np.ndarray:
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.int32)
    work = gray.astype(np.float32).copy()
    for y in range(h):
        if serpentine and (y % 2 == 1):
            x_range = range(w - 1, -1, -1)
            flip = -1
        else:
            x_range = range(w)
            flip = 1
        for x in x_range:
            old = work[y, x]
            idx = int(np.abs(palette_luma - old).argmin())
            new = palette_luma[idx]
            out[y, x] = idx
            err = old - new
            for dx, dy, weight in kernel:
                nx = x + dx * flip
                ny = y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    work[ny, nx] += err * weight
    return out


def _error_diffuse_1d(gray: np.ndarray, palette_luma: np.ndarray, axis: str) -> np.ndarray:
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.int32)
    work = gray.astype(np.float32).copy()
    if axis == "y":
        for x in range(w):
            for y in range(h):
                old = work[y, x]
                idx = int(np.abs(palette_luma - old).argmin())
                new = palette_luma[idx]
                out[y, x] = idx
                err = old - new
                if y + 1 < h:
                    work[y + 1, x] += err * 0.75
                if y + 2 < h:
                    work[y + 2, x] += err * 0.25
    else:
        for y in range(h):
            for x in range(w):
                old = work[y, x]
                idx = int(np.abs(palette_luma - old).argmin())
                new = palette_luma[idx]
                out[y, x] = idx
                err = old - new
                if x + 1 < w:
                    work[y, x + 1] += err * 0.75
                if x + 2 < w:
                    work[y, x + 2] += err * 0.25
    return out


def _error_diffuse_1d_mono(gray: np.ndarray, low: float, high: float, axis: str) -> np.ndarray:
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.int32)
    work = gray.astype(np.float32).copy()
    threshold = (low + high) * 0.5
    if axis == "y":
        for x in range(w):
            for y in range(h):
                old = work[y, x]
                new = high if old >= threshold else low
                out[y, x] = 1 if old >= threshold else 0
                err = old - new
                if y + 1 < h:
                    work[y + 1, x] += err * 0.75
                if y + 2 < h:
                    work[y + 2, x] += err * 0.25
    else:
        for y in range(h):
            for x in range(w):
                old = work[y, x]
                new = high if old >= threshold else low
                out[y, x] = 1 if old >= threshold else 0
                err = old - new
                if x + 1 < w:
                    work[y, x + 1] += err * 0.75
                if x + 2 < w:
                    work[y, x + 2] += err * 0.25
    return out


def _gaussian_blur_gray(gray: np.ndarray, sigma: float) -> np.ndarray:
    src = gray.astype(np.float32, copy=False)
    return cv2.GaussianBlur(src, (0, 0), sigma)


def apply_dither_optimized(
    img: Image.Image,
    pixel_size: int,
    threshold: int,
    palette_data: tuple[np.ndarray, np.ndarray, int],
    method: str,
    brightness: float,
    contrast: float,
    blur: float,
    sharpen: int,
    invert: bool,
    blend: float,
    mod_strength: float,
    mod_period: int,
) -> Image.Image:
    work = img.convert("RGB")
    if invert:
        work = ImageOps.invert(work)
    work = apply_image_adjustments(work, brightness, contrast, blur, sharpen)

    palette_array, palette_luma, levels = palette_data

    gray = work.convert("L")
    pixel_size = max(1, int(pixel_size))
    new_size = (max(1, gray.width // pixel_size), max(1, gray.height // pixel_size))
    gray = gray.resize(new_size, Image.Resampling.NEAREST)
    gray_array = np.array(gray, dtype=np.float32)

    bias = int(np.clip(threshold, 0, 255)) - 128
    gray_array = np.clip(gray_array + bias, 0, 255)

    method = method or "Bayer"
    def bayer_matrix(size: int) -> np.ndarray:
        matrix = np.array([[0]], dtype=np.int32)
        n = 1
        base = np.array([[0, 2], [3, 1]], dtype=np.int32)
        while n < size:
            matrix = np.block(
                [
                    [4 * matrix + base[0, 0], 4 * matrix + base[0, 1]],
                    [4 * matrix + base[1, 0], 4 * matrix + base[1, 1]],
                ]
            )
            n *= 2
        return matrix.astype(np.float32)

    ordered_matrices: dict[str, np.ndarray] = {
        "Bayer 2x2": (bayer_matrix(2) + 0.5) / 4.0,
        "Bayer 4x4": (bayer_matrix(4) + 0.5) / 16.0,
        "Bayer 8x8": (bayer_matrix(8) + 0.5) / 64.0,
        "Clustered Dot": (np.array(
            [
                [12, 5, 6, 13],
                [4, 0, 1, 7],
                [11, 3, 2, 8],
                [15, 10, 9, 14],
            ],
            dtype=np.float32,
        ) + 0.5)
        / 16.0,
        "Halftone": (np.array(
            [
                [7, 13, 11, 4],
                [12, 16, 14, 8],
                [10, 15, 6, 2],
                [5, 9, 3, 1],
            ],
            dtype=np.float32,
        )
        - 1.0
        + 0.5)
        / 16.0,
    }

    error_kernels: dict[str, list[tuple[int, int, float]]] = {
        "Floyd-Steinberg": [
            (1, 0, 7 / 16),
            (-1, 1, 3 / 16),
            (0, 1, 5 / 16),
            (1, 1, 1 / 16),
        ],
        "Jarvis-Judice-Ninke": [
            (1, 0, 7 / 48),
            (2, 0, 5 / 48),
            (-2, 1, 3 / 48),
            (-1, 1, 5 / 48),
            (0, 1, 7 / 48),
            (1, 1, 5 / 48),
            (2, 1, 3 / 48),
            (-2, 2, 1 / 48),
            (-1, 2, 3 / 48),
            (0, 2, 5 / 48),
            (1, 2, 3 / 48),
            (2, 2, 1 / 48),
        ],
        "Stucki": [
            (1, 0, 8 / 42),
            (2, 0, 4 / 42),
            (-2, 1, 2 / 42),
            (-1, 1, 4 / 42),
            (0, 1, 8 / 42),
            (1, 1, 4 / 42),
            (2, 1, 2 / 42),
            (-2, 2, 1 / 42),
            (-1, 2, 2 / 42),
            (0, 2, 4 / 42),
            (1, 2, 2 / 42),
            (2, 2, 1 / 42),
        ],
        "Burkes": [
            (1, 0, 8 / 32),
            (2, 0, 4 / 32),
            (-2, 1, 2 / 32),
            (-1, 1, 4 / 32),
            (0, 1, 8 / 32),
            (1, 1, 4 / 32),
            (2, 1, 2 / 32),
        ],
        "Sierra": [
            (1, 0, 5 / 32),
            (2, 0, 3 / 32),
            (-2, 1, 2 / 32),
            (-1, 1, 4 / 32),
            (0, 1, 5 / 32),
            (1, 1, 4 / 32),
            (2, 1, 2 / 32),
            (-1, 2, 2 / 32),
            (0, 2, 3 / 32),
            (1, 2, 2 / 32),
        ],
        "Sierra-2Row": [
            (1, 0, 4 / 16),
            (2, 0, 3 / 16),
            (-2, 1, 1 / 16),
            (-1, 1, 2 / 16),
            (0, 1, 3 / 16),
            (1, 1, 2 / 16),
            (2, 1, 1 / 16),
        ],
        "Sierra Lite": [
            (1, 0, 2 / 4),
            (-1, 1, 1 / 4),
            (0, 1, 1 / 4),
        ],
        "Atkinson": [
            (1, 0, 1 / 8),
            (2, 0, 1 / 8),
            (-1, 1, 1 / 8),
            (0, 1, 1 / 8),
            (1, 1, 1 / 8),
            (0, 2, 1 / 8),
        ],
    }

    h, w = gray_array.shape
    bayer8 = bayer_matrix(8) / 64.0

    def bayer8_threshold_map() -> np.ndarray:
        return np.tile(bayer8, (h // 8 + 1, w // 8 + 1))[:h, :w]

    def wave_values(length: int, period: int, kind: str, phase: float) -> np.ndarray:
        period = max(2, int(period))
        t = np.arange(length, dtype=np.float32) / float(period) + phase
        if kind == "triangle":
            return 2.0 * np.abs(2.0 * (t - np.floor(t + 0.5))) - 1.0
        if kind == "square":
            return np.where(np.sin(2 * np.pi * t) >= 0, 1.0, -1.0).astype(np.float32)
        if kind == "saw":
            return 2.0 * (t - np.floor(t + 0.5)).astype(np.float32)
        return np.sin(2 * np.pi * t).astype(np.float32)

    def modulation_field(axis: str, period: int, kind: str, phase: float) -> np.ndarray:
        if axis == "x":
            wave = wave_values(w, period, kind, phase)
            return np.tile(wave, (h, 1))
        if axis == "y":
            wave = wave_values(h, period, kind, phase)
            return np.tile(wave[:, None], (1, w))
        wave_x = wave_values(w, period, kind, phase)
        wave_y = wave_values(h, period, kind, phase)
        return (np.tile(wave_x, (h, 1)) + np.tile(wave_y[:, None], (1, w))) * 0.5

    def apply_modulation(gray: np.ndarray, axis: str, amplitude, period: int, kind: str, phase: float) -> np.ndarray:
        field = modulation_field(axis, period, kind, phase)
        return np.clip(gray + field * amplitude, 0, 255)

    def apply_crt_modulation_y(gray: np.ndarray, amplitude: float, period: int, phase: float) -> np.ndarray:
        period = max(2, int(period))
        base = gray.astype(np.float32)

        blur_map = _gaussian_blur_gray(gray, 1.2) / 255.0
        phase_map = (blur_map - 0.5) * 1.6

        y = (np.arange(h, dtype=np.float32) / float(period) + phase)[:, None]
        carrier = np.sin(2 * np.pi * (y + phase_map))
        carrier2 = np.sin(2 * np.pi * (y * 2.0 + phase_map * 0.7))
        mod = carrier * amplitude * 0.7 + carrier2 * amplitude * 0.2
        modded = np.clip(base + mod, 0, 255)

        rng = np.random.default_rng(1337)
        noise = rng.normal(0.0, amplitude * 0.05, size=modded.shape).astype(np.float32)
        return np.clip(modded + noise, 0, 255)

    def rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]
        c_max = np.maximum(np.maximum(r, g), b)
        c_min = np.minimum(np.minimum(r, g), b)
        delta = c_max - c_min
        l = (c_max + c_min) / 2.0
        s = np.zeros_like(l)
        mask = delta > 1e-6
        s[mask] = delta[mask] / (1.0 - np.abs(2.0 * l[mask] - 1.0))
        h_val = np.zeros_like(l)
        r_mask = (c_max == r) & mask
        g_mask = (c_max == g) & mask
        b_mask = (c_max == b) & mask
        h_val[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6.0
        h_val[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2.0
        h_val[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4.0
        h_val = (h_val / 6.0) % 1.0
        return np.stack([h_val, s, l], axis=-1)

    def hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
        h = hsl[:, :, 0]
        s = hsl[:, :, 1]
        l = hsl[:, :, 2]
        c = (1.0 - np.abs(2.0 * l - 1.0)) * s
        h6 = h * 6.0
        x = c * (1.0 - np.abs((h6 % 2.0) - 1.0))
        m = l - c / 2.0

        zeros = np.zeros_like(h)
        r = np.zeros_like(h)
        g = np.zeros_like(h)
        b = np.zeros_like(h)

        conds = [
            (0.0 <= h6) & (h6 < 1.0),
            (1.0 <= h6) & (h6 < 2.0),
            (2.0 <= h6) & (h6 < 3.0),
            (3.0 <= h6) & (h6 < 4.0),
            (4.0 <= h6) & (h6 < 5.0),
            (5.0 <= h6) & (h6 < 6.0),
        ]
        r_vals = [c, x, zeros, zeros, x, c]
        g_vals = [x, c, c, x, zeros, zeros]
        b_vals = [zeros, zeros, x, c, c, x]

        for cond, rv, gv, bv in zip(conds, r_vals, g_vals, b_vals):
            r[cond] = rv[cond]
            g[cond] = gv[cond]
            b[cond] = bv[cond]

        rgb = np.stack([r + m, g + m, b + m], axis=-1)
        return np.clip(rgb, 0, 1)

    modulation_methods = MODULATION_METHODS

    phase = float(threshold) / 255.0
    mod_amp = max(1.0, float(mod_strength))
    mod_per = max(2, int(mod_period))

    if method in {"Bayer 8x8 Quantize", "HSL Palette Dither", "Bayer 8x8 Palette Luma"}:
        rgb_small = np.array(work.resize(new_size, Image.Resampling.NEAREST), dtype=np.float32) / 255.0
        threshold_map = bayer8_threshold_map()

        if method == "Bayer 8x8 Quantize":
            color_num = max(2, palette_array.shape[0])
            threshold = threshold_map - 0.88
            rgb_small = np.clip(rgb_small + threshold[:, :, None], 0, 1)
            quant = np.floor(rgb_small * (color_num - 1.0) + 0.5) / (color_num - 1.0)
            rgb_small = quant

        elif method == "Bayer 8x8 Palette Luma":
            steps = max(2, min(4, palette_array.shape[0]))
            threshold = threshold_map - 0.88
            lum = (rgb_small[:, :, 0] * 0.2126 + rgb_small[:, :, 1] * 0.7152 + rgb_small[:, :, 2] * 0.0722)
            lum = np.clip(lum + threshold * 0.2, 0, 1)
            lum_q = np.floor(lum * (steps - 1.0) + 0.5) / (steps - 1.0)
            idx = np.clip((lum_q * (steps - 1.0)).round().astype(int), 0, steps - 1)
            rgb_small = (palette_array[:steps][idx] / 255.0).astype(np.float32)

        else:
            hsl = rgb_to_hsl(rgb_small)
            palette_rgb = palette_array.astype(np.float32) / 255.0
            palette_hsl = rgb_to_hsl(palette_rgb.reshape(1, -1, 3)).reshape(-1, 3)
            palette_hue = palette_hsl[:, 0]

            hue = hsl[:, :, 0].reshape(-1, 1)
            diff = np.abs(hue - palette_hue[None, :])
            dist = np.minimum(diff, 1.0 - diff)
            order = np.argpartition(dist, 1, axis=1)
            idx0 = order[:, 0]
            idx1 = order[:, 1]

            hue0 = palette_hue[idx0]
            hue1 = palette_hue[idx1]
            dist0 = dist[np.arange(dist.shape[0]), idx0]
            dist01 = np.minimum(np.abs(hue1 - hue0), 1.0 - np.abs(hue1 - hue0))
            hue_diff = dist0 / np.maximum(dist01, 1e-6)

            hsl_flat = hsl.reshape(-1, 3)
            l = hsl_flat[:, 2]
            s = hsl_flat[:, 1]
            steps = 16.0
            l1 = np.floor(0.5 + np.maximum(l - 0.125, 0.0) * steps) / steps
            l2 = np.floor(0.5 + np.minimum(l + 0.124, 1.0) * steps) / steps
            l_diff = (l - l1) / np.maximum(l2 - l1, 1e-6)

            s1 = np.floor(0.5 + np.maximum(s - 0.125, 0.0) * steps) / steps
            s2 = np.floor(0.5 + np.minimum(s + 0.124, 1.0) * steps) / steps
            s_diff = (s - s1) / np.maximum(s2 - s1, 1e-6)

            threshold = (threshold_map.reshape(-1) + (1.0 / 64.0) + 0.130)
            choose_second = hue_diff >= threshold

            hsl_out = np.zeros_like(hsl_flat)
            hsl_out[:, 0] = np.where(choose_second, hue1, hue0)
            hsl_out[:, 2] = np.where(l_diff < threshold, l1, l2)
            hsl_out[:, 1] = np.where(s_diff < threshold, s1, s2)
            rgb_small = hsl_to_rgb(hsl_out.reshape(hsl.shape))

        rgb_small = (rgb_small * 255.0).astype(np.uint8)
        dithered = Image.fromarray(rgb_small, mode="RGB")
        dithered = dithered.resize(img.size, Image.Resampling.NEAREST)
        blend = float(np.clip(blend, 0.0, 1.0))
        if blend < 1.0:
            base = np.array(work.resize(img.size, Image.Resampling.BILINEAR), dtype=np.float32)
            mixed = np.clip(base * (1.0 - blend) + np.array(dithered, dtype=np.float32) * blend, 0, 255).astype(
                np.uint8
            )
            return Image.fromarray(mixed, mode="RGB")
        return dithered

    if method in modulation_methods:
        if method == "Modulated Diffuse Y":
            line_period = max(2, int(round(mod_per)))
            crt_amp = min(72.0, max(12.0, mod_amp * 1.1))
            gray_mod = apply_modulation(gray_array, "y", crt_amp, line_period, "sine", phase)
            indices = _error_diffuse_1d_mono(gray_mod, palette_luma[0], palette_luma[-1], axis="y")
        elif method == "Modulated Diffuse X":
            gray_mod = apply_modulation(gray_array, "x", mod_amp * 1.1, mod_per, "sine", phase)
            indices = _error_diffuse_1d_mono(gray_mod, palette_luma[0], palette_luma[-1], axis="x")
        elif method == "Modulated Smooth Diffuse":
            gray_mod = _gaussian_blur_gray(gray_array, 2.4)
            line_period = max(2, int(round(mod_per * 1.6)))
            gray_mod = apply_crt_modulation_y(gray_mod, mod_amp * 0.7, line_period, phase)
            normalized = np.clip(gray_mod / 255.0, 0.0, 1.0)
            normalized = normalized ** 1.4
            gray_mod = normalized * 255.0
            bias = (threshold - 128) * 1.2
            if bias != 0:
                gray_mod = np.clip(gray_mod - bias, 0, 255)
            indices = _error_diffuse_1d_mono(gray_mod, palette_luma[0], palette_luma[-1], axis="x")
        else:
            indices = _ordered_dither_indices(gray_array, ordered_matrices["Bayer 4x4"], levels)
    elif method in ordered_matrices:
        indices = _ordered_dither_indices(gray_array, ordered_matrices[method], levels)
    elif method in error_kernels:
        indices = _error_diffuse_indices(gray_array, palette_luma, error_kernels[method])
    else:
        fallback = ordered_matrices["Bayer 4x4"]
        indices = _ordered_dither_indices(gray_array, fallback, levels)

    rgb_small = palette_array[indices]
    dithered = Image.fromarray(rgb_small, mode="RGB")
    dithered = dithered.resize(img.size, Image.Resampling.NEAREST)

    blend = float(np.clip(blend, 0.0, 1.0))
    if blend < 1.0:
        base = np.array(work.resize(img.size, Image.Resampling.BILINEAR), dtype=np.float32)
        mixed = np.clip(base * (1.0 - blend) + np.array(dithered, dtype=np.float32) * blend, 0, 255).astype(
            np.uint8
        )
        return Image.fromarray(mixed, mode="RGB")

    return dithered


class DitherWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Dither-Dawg")
        self.resize(1280, 760)
        self.setStyleSheet(self._style_sheet())

        base_dir = Path(__file__).resolve().parent
        palette_dir = base_dir / "pallettes"
        if not palette_dir.exists():
            palette_dir = base_dir / "palettes"
        self.palette_dir = palette_dir
        self.preset_dir = base_dir / "presets"
        self.preset_dir.mkdir(parents=True, exist_ok=True)
        self._preset_files: dict[str, Path] = {}
        self.palettes_by_category = load_palette_library(self.palette_dir)
        self.palette_categories = self._sorted_palette_categories()
        self._palette_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, int]] = {}
        self._mono_palette_data = _build_palette_data(BUILTIN_PALETTES["Mono"])
        self._preview_cache_key: tuple[int, int, int] | None = None
        self._preview_cache_image: Image.Image | None = None
        self._last_settings: dict[str, object] | None = None

        self.original_image: Image.Image | None = None
        self.processed_image: Image.Image | None = None
        self._update_scheduled = False
        self.zoom = 1.0
        self._video_mode: str | None = None
        self._video_frames: list[Image.Image] = []
        self._video_durations: list[int] = []
        self._video_index = 0
        self._video_capture: cv2.VideoCapture | None = None
        self._video_interval_ms = 80
        self._video_path: str | None = None
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self._advance_video_frame)

        self._build_menu()

        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        self.preview_area = self._build_preview()
        main_layout.addWidget(self.preview_area, 1)

        controls = self._build_controls()
        self.control_scroll = QScrollArea()
        self.control_scroll.setWidget(controls)
        self.control_scroll.setWidgetResizable(True)
        self.control_scroll.setFrameShape(QFrame.NoFrame)
        self.control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.control_scroll.setMinimumWidth(320)
        self.control_scroll.setMaximumWidth(380)
        main_layout.addWidget(self.control_scroll, 0)

        self._load_presets()
        self._load_starter_image()

    def _load_starter_image(self) -> None:
        starter_path = Path(__file__).resolve().parent / "dither-dawg-starter-image.png"
        if not starter_path.exists():
            return
        try:
            image = Image.open(starter_path)
            image = ImageOps.exif_transpose(image)
            self.original_image = image.convert("RGB")
            self._preview_cache_key = None
            self._preview_cache_image = None
            self.schedule_update()
        except Exception:
            return

    def _style_sheet(self) -> str:
        return """
        QMainWindow { background: #0f0f10; }
        QWidget { color: #d6d6d6; font-family: "Segoe UI"; font-size: 10.5pt; }
        #sidebar { background: #1a1a1a; border: 1px solid #252525; border-radius: 8px; }
        QLabel#sectionTitle { color: #9aa0a6; font-size: 9.5pt; }
        QLabel#logo { font-size: 18pt; font-weight: 700; letter-spacing: 1px; }
        QLabel#hint { color: #a5a5a5; font-size: 9.5pt; }
        QPushButton { background: #2a2a2a; border: 1px solid #3a3a3a; padding: 6px 10px; border-radius: 6px; }
        QPushButton:hover { background: #333333; }
        QPushButton:pressed { background: #262626; }
        QComboBox { background: #242424; border: 1px solid #3a3a3a; padding: 4px 8px; border-radius: 6px; }
        QComboBox::drop-down { border: none; }
        QSlider::groove:horizontal { height: 6px; background: #2b2b2b; border-radius: 3px; }
        QSlider::handle:horizontal { width: 14px; margin: -5px 0; background: #cfcfcf; border-radius: 7px; }
        QCheckBox { spacing: 8px; }
        QScrollArea { background: #0b0b0c; border: 1px solid #232323; border-radius: 8px; }
        QFrame#divider { background: #2a2a2a; max-height: 1px; }
        """

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        edit_menu = menu_bar.addMenu("Edit")
        batch_menu = menu_bar.addMenu("Batch")
        adjustments_menu = menu_bar.addMenu("Adjustments")
        themes_menu = menu_bar.addMenu("Themes")
        extras_menu = menu_bar.addMenu("Extras")
        help_menu = menu_bar.addMenu("Help")

        import_action = QAction("Import", self)
        self.export_image_action = QAction("Export Image", self)
        self.export_gif_action = QAction("Export GIF", self)
        self.export_video_action = QAction("Export Video", self)
        self.export_preset_action = QAction("Export Preset", self)
        self.import_preset_action = QAction("Import Preset", self)
        reset_action = QAction("Reset All", self)
        reset_zoom_action = QAction("Reset Zoom", self)
        quit_action = QAction("Quit", self)
        help_action = QAction("About", self)
        shortcuts_action = QAction("Shortcuts", self)
        self.invert_action = QAction("Invert", self)
        self.invert_action.setCheckable(True)

        import_action.triggered.connect(self.import_image)
        self.export_image_action.triggered.connect(self.export_image)
        self.export_gif_action.triggered.connect(self.export_gif)
        self.export_video_action.triggered.connect(self.export_video)
        self.export_preset_action.triggered.connect(self.save_preset)
        self.import_preset_action.triggered.connect(self.import_preset)
        reset_action.triggered.connect(self.reset_controls)
        reset_zoom_action.triggered.connect(self.reset_zoom)
        quit_action.triggered.connect(self.close)
        help_action.triggered.connect(self.show_about)
        shortcuts_action.triggered.connect(self.show_shortcuts)
        self.invert_action.toggled.connect(self.toggle_invert_from_menu)

        file_menu.addAction(import_action)
        export_menu = file_menu.addMenu("Export")
        export_menu.addAction(self.export_image_action)
        export_menu.addAction(self.export_gif_action)
        export_menu.addAction(self.export_video_action)
        file_menu.addAction(self.export_preset_action)
        file_menu.addAction(self.import_preset_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        edit_menu.addAction(reset_action)
        edit_menu.addAction(reset_zoom_action)
        adjustments_menu.addAction(self.invert_action)

        batch_export_action = QAction("Batch Export (Coming Soon)", self)
        batch_export_action.triggered.connect(lambda: self.show_placeholder("Batch export"))
        batch_menu.addAction(batch_export_action)

        theme_default_action = QAction("Default Theme", self)
        theme_default_action.triggered.connect(lambda: self.show_placeholder("Themes"))
        themes_menu.addAction(theme_default_action)

        import_palette_action = QAction("Import Palette", self)
        import_palette_action.triggered.connect(self.import_palette)
        extras_menu.addAction(import_palette_action)
        show_palettes_action = QAction("Show Palettes Folder", self)
        show_palettes_action.triggered.connect(lambda: self.open_folder("Palettes Folder", self.palette_dir))
        extras_menu.addAction(show_palettes_action)
        show_presets_action = QAction("Show Presets Folder", self)
        show_presets_action.triggered.connect(lambda: self.open_folder("Presets Folder", self.preset_dir))
        extras_menu.addAction(show_presets_action)

        help_menu.addAction(help_action)
        help_menu.addAction(shortcuts_action)
        self._update_export_actions()

    def _build_preview(self) -> QScrollArea:
        self.image_label = QLabel("Import an image to preview")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(500, 400)
        self.image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.image_label.setStyleSheet(
            "QLabel { background: #111114; color: #c0c0c0; border: 1px solid #242424; }"
        )

        area = QScrollArea()
        area.setWidget(self.image_label)
        area.setWidgetResizable(False)
        area.setAlignment(Qt.AlignCenter)
        area.setFrameShape(QFrame.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        return area

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("sidebar")
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(360)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        top_buttons = QHBoxLayout()
        self.import_button = QPushButton("Import")
        self.export_button = QPushButton("Export")
        top_buttons.addWidget(self.import_button)
        top_buttons.addWidget(self.export_button)
        layout.addLayout(top_buttons)

        zoom_buttons = QHBoxLayout()
        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_reset_button = QPushButton("Reset Zoom")
        zoom_buttons.addWidget(self.zoom_in_button)
        zoom_buttons.addWidget(self.zoom_out_button)
        zoom_buttons.addWidget(self.zoom_reset_button)
        layout.addLayout(zoom_buttons)

        hint = QLabel("Ctrl+Shift+? for help")
        hint.setObjectName("hint")
        layout.addWidget(hint)

        logo = QLabel("DITHER-DAWG")
        logo.setObjectName("logo")
        logo.setAlignment(Qt.AlignCenter)
        logo.setMinimumHeight(60)
        layout.addWidget(logo)

        layout.addWidget(self._divider())

        layout.addWidget(self._section_title("Style"))
        self.style_combo = QComboBox()
        ordered_styles = [
            "Bayer 2x2",
            "Bayer 4x4",
            "Bayer 8x8",
            "Clustered Dot",
            "Halftone",
        ]
        error_styles = [
            "Floyd-Steinberg",
            "Jarvis-Judice-Ninke",
            "Stucki",
            "Burkes",
            "Sierra",
            "Sierra-2Row",
            "Sierra Lite",
            "Atkinson",
        ]
        self._add_style_header("Ordered")
        for item in ordered_styles:
            self.style_combo.addItem(item)
        self._add_style_header("Error Diffusion")
        for item in error_styles:
            self.style_combo.addItem(item)
        self._add_style_header("Modulation")
        self.style_combo.addItem("Modulated Diffuse Y")
        self.style_combo.addItem("Modulated Diffuse X")
        self.style_combo.addItem("Modulated Smooth Diffuse")
        self._add_style_header("Glitch")
        self.style_combo.addItem("Bayer 8x8 Quantize")
        self.style_combo.addItem("HSL Palette Dither")
        self.style_combo.addItem("Bayer 8x8 Palette Luma")
        self.style_combo.setCurrentText("Bayer 4x4")
        layout.addWidget(self.style_combo)

        layout.addWidget(self._section_title("Presets"))
        self.presets_combo = QComboBox()
        self.presets_combo.addItems(["None"])
        layout.addWidget(self.presets_combo)

        layout.addWidget(self._section_title("Scale"))
        self.scale_slider, self.scale_value = self._make_slider(1, 20, 11, "")
        self._add_slider(layout, self.scale_slider, self.scale_value)

        layout.addWidget(self._section_title("Line Scale"))
        self.line_scale_slider, self.line_scale_value = self._make_slider(1, 6, 1, "")
        self._add_slider(layout, self.line_scale_slider, self.line_scale_value)

        layout.addWidget(self._section_title("Smoothing Factor"))
        self.smoothing_slider, self.smoothing_value = self._make_slider(0, 100, 0, "")
        self._add_slider(layout, self.smoothing_slider, self.smoothing_value)

        layout.addWidget(self._section_title("Bleed Fraction"))
        self.bleed_slider, self.bleed_value = self._make_slider(0, 100, 0, "")
        self._add_slider(layout, self.bleed_slider, self.bleed_value)

        layout.addWidget(self._section_title("Palette Category"))
        self.palette_category_combo = QComboBox()
        if self.palette_categories:
            self.palette_category_combo.addItems(self.palette_categories)
        else:
            self.palette_category_combo.addItems(["Built-in"])
        layout.addWidget(self.palette_category_combo)

        layout.addWidget(self._section_title("Palette"))
        self.palette_combo = QComboBox()
        self._populate_palette_combo(self.palette_category_combo.currentText())
        layout.addWidget(self.palette_combo)

        self.palette_swatch_widget = QWidget()
        self.palette_swatch_layout = QGridLayout(self.palette_swatch_widget)
        self.palette_swatch_layout.setContentsMargins(0, 0, 0, 0)
        self.palette_swatch_layout.setHorizontalSpacing(4)
        self.palette_swatch_layout.setVerticalSpacing(4)
        layout.addWidget(self.palette_swatch_widget)

        self.import_palette_button = QPushButton("Import Palette")
        layout.addWidget(self.import_palette_button)

        layout.addWidget(self._section_title("Invert"))
        self.invert_check = QCheckBox("Enable Invert")
        layout.addWidget(self.invert_check)

        layout.addWidget(self._section_title("Contrast"))
        self.contrast_slider, self.contrast_value = self._make_slider(0, 100, 45, "")
        self._add_slider(layout, self.contrast_slider, self.contrast_value)

        layout.addWidget(self._section_title("Midtones"))
        self.midtones_slider, self.midtones_value = self._make_slider(0, 100, 50, "")
        self._add_slider(layout, self.midtones_slider, self.midtones_value)

        layout.addWidget(self._section_title("Highlights"))
        self.highlights_slider, self.highlights_value = self._make_slider(0, 100, 50, "")
        self._add_slider(layout, self.highlights_slider, self.highlights_value)

        layout.addWidget(self._section_title("Luminance Threshold"))
        self.luma_slider, self.luma_value = self._make_slider(0, 100, 50, "")
        self._add_slider(layout, self.luma_slider, self.luma_value)

        layout.addWidget(self._section_title("Blur"))
        self.blur_slider, self.blur_value = self._make_slider(0, 100, 0, "")
        self._add_slider(layout, self.blur_slider, self.blur_value)

        layout.addWidget(self._divider())

        bottom_buttons = QHBoxLayout()
        self.save_preset_button = QPushButton("Save Preset")
        self.reset_all_button = QPushButton("Reset All")
        bottom_buttons.addWidget(self.save_preset_button)
        bottom_buttons.addWidget(self.reset_all_button)
        layout.addLayout(bottom_buttons)
        layout.addStretch(1)

        self.import_button.clicked.connect(self.import_image)
        self.export_button.clicked.connect(self.export_image)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.zoom_reset_button.clicked.connect(self.reset_zoom)
        self.reset_all_button.clicked.connect(self.reset_controls)
        self.save_preset_button.clicked.connect(self.save_preset)
        self.import_palette_button.clicked.connect(self.import_palette)

        for slider in [
            self.scale_slider,
            self.line_scale_slider,
            self.smoothing_slider,
            self.bleed_slider,
            self.contrast_slider,
            self.midtones_slider,
            self.highlights_slider,
            self.luma_slider,
            self.blur_slider,
        ]:
            slider.valueChanged.connect(self.schedule_update)

        for combo in [
            self.style_combo,
            self.palette_category_combo,
            self.palette_combo,
        ]:
            combo.currentIndexChanged.connect(self.schedule_update)

        self.presets_combo.currentIndexChanged.connect(self.on_preset_changed)
        self.palette_category_combo.currentIndexChanged.connect(self.on_palette_category_changed)
        self.palette_combo.currentIndexChanged.connect(self._update_palette_swatches)
        self.invert_check.toggled.connect(self.schedule_update)
        if hasattr(self, "invert_action"):
            self.invert_action.setChecked(self.invert_check.isChecked())
            self.invert_check.toggled.connect(self.invert_action.setChecked)

        self._update_palette_swatches()
        return panel

    def _section_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    def _add_style_header(self, text: str) -> None:
        self.style_combo.addItem(text)
        header_index = self.style_combo.count() - 1
        item = self.style_combo.model().item(header_index)
        item.setFlags(Qt.ItemIsEnabled)
        item.setData(QBrush(QColor("#f2f2f2")), Qt.BackgroundRole)
        item.setData(QBrush(QColor("#1a1a1a")), Qt.ForegroundRole)
        item.setData(Qt.AlignCenter, Qt.TextAlignmentRole)

    def _divider(self) -> QFrame:
        line = QFrame()
        line.setObjectName("divider")
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Plain)
        return line

    def _sorted_palette_categories(self) -> list[str]:
        categories = list(self.palettes_by_category.keys())
        built_in = [name for name in categories if name.lower() == "built-in"]
        others = sorted([name for name in categories if name.lower() != "built-in"], key=str.lower)
        return built_in + others

    def _populate_palette_combo(self, category: str, keep_selection: bool = False) -> None:
        palettes = self.palettes_by_category.get(category, [])
        names = [palette.name for palette in palettes]
        if not names:
            names = ["None"]
        current = self.palette_combo.currentText() if keep_selection else None
        self.palette_combo.blockSignals(True)
        self.palette_combo.clear()
        self.palette_combo.addItems(names)
        if current and current in names:
            self.palette_combo.setCurrentText(current)
        self.palette_combo.blockSignals(False)
        if hasattr(self, "palette_swatch_layout"):
            self._update_palette_swatches()

    def _get_selected_palette_colors(self) -> list[tuple[int, int, int]]:
        category = self.palette_category_combo.currentText()
        name = self.palette_combo.currentText()
        for palette in self.palettes_by_category.get(category, []):
            if palette.name == name:
                return palette.colors
        return BUILTIN_PALETTES["Mono"]

    def _get_palette_data(self) -> tuple[np.ndarray, np.ndarray, int]:
        category = self.palette_category_combo.currentText()
        name = self.palette_combo.currentText()
        key = (category, name)
        cached = self._palette_cache.get(key)
        if cached:
            return cached
        colors = self._get_selected_palette_colors()
        data = _build_palette_data(colors)
        self._palette_cache[key] = data
        return data

    def _get_preview_source(self) -> tuple[Image.Image, float]:
        if self.original_image is None:
            raise ValueError("No source image loaded")
        viewport = self.preview_area.viewport().size()
        target_max = max(600, min(max(viewport.width(), viewport.height()) * 2, 1800))
        source = self.original_image
        scale = min(1.0, target_max / max(source.width, source.height))
        key = (id(source), source.width, source.height, int(target_max))
        if scale >= 1.0:
            return source, 1.0
        if self._preview_cache_key == key and self._preview_cache_image is not None:
            return self._preview_cache_image, scale
        new_size = (max(1, int(source.width * scale)), max(1, int(source.height * scale)))
        preview = source.resize(new_size, Image.Resampling.LANCZOS)
        self._preview_cache_key = key
        self._preview_cache_image = preview
        return preview, scale

    def _collect_settings(self) -> dict[str, object]:
        scale = self.scale_slider.value()
        line_scale = self.line_scale_slider.value()
        smoothing = self.smoothing_slider.value() / 100.0

        line_multiplier = 0.5 + (line_scale / 2.0)
        pixel_size = max(1, int(round(scale * line_multiplier)))

        mod_strength = 10.0 + (self.bleed_slider.value() / 100.0) * 55.0 + smoothing * 30.0
        mod_period = max(2, int(round(scale / max(1.0, line_scale * 1.2) / 3)))

        threshold = int(self.luma_slider.value() / 100.0 * 255)
        threshold += int((self.highlights_slider.value() - 50) * 2)
        threshold += int(self.bleed_slider.value())
        threshold = int(np.clip(threshold, 0, 255))

        brightness = 0.5 + self.midtones_slider.value() / 100.0
        contrast = 0.5 + self.contrast_slider.value() / 100.0

        blur = (self.blur_slider.value() / 20.0) + smoothing * 2.0
        sharpen = 0

        method = self.style_combo.currentText()

        return {
            "pixel_size": pixel_size,
            "threshold": threshold,
            "brightness": brightness,
            "contrast": contrast,
            "blur": blur,
            "sharpen": sharpen,
            "invert": self.invert_check.isChecked(),
            "blend": 1.0,
            "method": method,
            "mod_strength": mod_strength,
            "mod_period": mod_period,
        }

    def _run_dither(self, source: Image.Image, settings: dict[str, object], scale_factor: float) -> Image.Image:
        pixel_size = max(1, int(round(settings["pixel_size"] * scale_factor)))
        method = str(settings["method"])
        palette_data = self._mono_palette_data if method in MODULATION_METHODS else self._get_palette_data()
        return apply_dither_optimized(
            source,
            pixel_size=pixel_size,
            threshold=int(settings["threshold"]),
            palette_data=palette_data,
            method=method,
            brightness=float(settings["brightness"]),
            contrast=float(settings["contrast"]),
            blur=float(settings["blur"]),
            sharpen=int(settings["sharpen"]),
            invert=bool(settings["invert"]),
            blend=float(settings["blend"]),
            mod_strength=float(settings["mod_strength"]),
            mod_period=int(settings["mod_period"]),
        )

    def _reload_palettes(self, keep_selection: bool = True) -> None:
        self.palettes_by_category = load_palette_library(self.palette_dir)
        self.palette_categories = self._sorted_palette_categories()
        self._palette_cache.clear()
        current_category = self.palette_category_combo.currentText() if keep_selection else None
        self.palette_category_combo.blockSignals(True)
        self.palette_category_combo.clear()
        if self.palette_categories:
            self.palette_category_combo.addItems(self.palette_categories)
        else:
            self.palette_category_combo.addItems(["Built-in"])
        if current_category and current_category in self.palette_categories:
            self.palette_category_combo.setCurrentText(current_category)
        self.palette_category_combo.blockSignals(False)
        self._populate_palette_combo(self.palette_category_combo.currentText(), keep_selection=keep_selection)

    def on_palette_category_changed(self) -> None:
        self._populate_palette_combo(self.palette_category_combo.currentText())

    def _update_palette_swatches(self) -> None:
        if not hasattr(self, "palette_swatch_layout"):
            return
        colors = self._get_selected_palette_colors()
        max_colors = 32
        display_colors = colors[:max_colors]

        while self.palette_swatch_layout.count():
            item = self.palette_swatch_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        columns = 8
        for idx, color in enumerate(display_colors):
            row = idx // columns
            col = idx % columns
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); border: 1px solid #111;"
            )
            swatch.setToolTip(f"rgb({color[0]}, {color[1]}, {color[2]})")
            self.palette_swatch_layout.addWidget(swatch, row, col)

        if len(colors) > max_colors:
            label = QLabel(f"+{len(colors) - max_colors} more")
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.palette_swatch_layout.addWidget(label, (len(display_colors) // columns) + 1, 0, 1, columns)

    def import_palette(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Palette",
            "",
            "Palette Files (*.ase *.gpl *.hex *.txt)",
        )
        if not file_path:
            return
        source = Path(file_path)
        dest_dir = self.palette_dir / "Imported"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / source.name
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            counter = 1
            while dest.exists():
                dest = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        try:
            shutil.copy2(source, dest)
        except Exception as exc:
            QMessageBox.critical(self, "Import Failed", f"Could not import palette:\n{exc}")
            return

        self._reload_palettes(keep_selection=False)
        if "Imported" in self.palette_categories:
            self.palette_category_combo.setCurrentText("Imported")
            self._populate_palette_combo("Imported")
            self.palette_combo.setCurrentText(dest.stem)
        QMessageBox.information(self, "Palette Imported", f"Added palette: {dest.stem}")

    def _reset_video_state(self) -> None:
        if self.video_timer.isActive():
            self.video_timer.stop()
        if self._video_capture is not None:
            self._video_capture.release()
        self._video_mode = None
        self._video_frames = []
        self._video_durations = []
        self._video_index = 0
        self._video_capture = None
        self._video_path = None
        self._update_export_actions()

    def _load_gif(self, file_path: str) -> None:
        try:
            self._reset_video_state()
            image = Image.open(file_path)
            frames: list[Image.Image] = []
            durations: list[int] = []
            for frame in ImageSequence.Iterator(image):
                frame = ImageOps.exif_transpose(frame)
                frames.append(frame.convert("RGB"))
                durations.append(int(frame.info.get("duration", 80)))
            if not frames:
                raise ValueError("No frames found")
            self._video_frames = frames
            self._video_durations = durations
            self._video_mode = "gif"
            self._video_index = 0
            self._video_path = file_path
            self.original_image = frames[0]
            self._preview_cache_key = None
            self._preview_cache_image = None
            self.schedule_update()
            self._update_export_actions()
            self.video_timer.setInterval(max(20, durations[0]))
            self.video_timer.start()
        except Exception as exc:
            QMessageBox.critical(self, "Import Failed", f"Could not open GIF:\n{exc}")

    def _load_video(self, file_path: str) -> None:
        self._reset_video_state()
        capture = cv2.VideoCapture(file_path)
        if not capture.isOpened():
            QMessageBox.critical(self, "Import Failed", "Could not open video.")
            return
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 24.0
        self._video_interval_ms = int(1000 / min(fps, 30.0))
        self._video_capture = capture
        self._video_mode = "video"
        self._video_path = file_path
        self._advance_video_frame()
        self._update_export_actions()
        self.video_timer.start(self._video_interval_ms)

    def _advance_video_frame(self) -> None:
        if self._video_mode == "gif":
            if not self._video_frames:
                return
            frame = self._video_frames[self._video_index]
            self.original_image = frame
            self._preview_cache_key = None
            self._preview_cache_image = None
            self.schedule_update()
            duration = self._video_durations[self._video_index]
            self._video_index = (self._video_index + 1) % len(self._video_frames)
            self.video_timer.setInterval(max(20, duration))
            return

        if self._video_mode == "video" and self._video_capture is not None:
            ret, frame = self._video_capture.read()
            if not ret:
                self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._video_capture.read()
                if not ret:
                    return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.original_image = Image.fromarray(rgb)
            self._preview_cache_key = None
            self._preview_cache_image = None
            self.schedule_update()

    def _make_slider(self, minimum: int, maximum: int, value: int, suffix: str):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(value)
        slider.setSingleStep(1)
        slider.setPageStep(max(1, (maximum - minimum) // 10))
        value_label = QLabel()
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._update_value_label(value_label, value, suffix)
        slider.valueChanged.connect(lambda val: self._update_value_label(value_label, val, suffix))
        return slider, value_label

    def _add_slider(self, layout: QVBoxLayout, slider: QSlider, label: QLabel) -> None:
        row = QHBoxLayout()
        row.addWidget(slider, 1)
        row.addWidget(label, 0)
        layout.addLayout(row)

    def _update_value_label(self, label: QLabel, value: int, suffix: str) -> None:
        label.setText(f"{value}{suffix}")

    def _update_export_actions(self) -> None:
        has_image = self.original_image is not None
        if hasattr(self, "export_image_action"):
            self.export_image_action.setEnabled(has_image)
        if hasattr(self, "export_gif_action"):
            self.export_gif_action.setEnabled(self._video_mode == "gif" and bool(self._video_frames))
        if hasattr(self, "export_video_action"):
            self.export_video_action.setEnabled(self._video_mode == "video" and bool(self._video_path))
        if hasattr(self, "export_button"):
            self.export_button.setEnabled(has_image)

    def show_placeholder(self, feature: str) -> None:
        QMessageBox.information(self, "Coming Soon", f"{feature} is not implemented yet.")

    def open_folder(self, title: str, path: Path) -> None:
        url = QUrl.fromLocalFile(str(path))
        if not QDesktopServices.openUrl(url):
            QMessageBox.information(self, title, f"Could not open folder:\n{path}")

    def show_shortcuts(self) -> None:
        QMessageBox.information(
            self,
            "Shortcuts",
            "Zoom In: Ctrl+=\nZoom Out: Ctrl+-\nReset Zoom: Ctrl+0",
        )

    def toggle_invert_from_menu(self, checked: bool) -> None:
        if hasattr(self, "invert_check"):
            self.invert_check.setChecked(checked)

    def schedule_update(self) -> None:
        if self._update_scheduled:
            return
        self._update_scheduled = True
        QTimer.singleShot(0, self.update_preview)

    def import_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Image",
            "",
            "Images/Videos (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.gif *.mp4 *.mov *.avi *.mkv);;Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.gif);;Videos (*.mp4 *.mov *.avi *.mkv)",
        )
        if not file_path:
            return
        ext = Path(file_path).suffix.lower()
        if ext == ".gif":
            self._load_gif(file_path)
            return
        if ext in {".mp4", ".mov", ".avi", ".mkv"}:
            self._load_video(file_path)
            return
        try:
            self._reset_video_state()
            image = Image.open(file_path)
            image = ImageOps.exif_transpose(image)
            self.original_image = image.convert("RGB")
            self._preview_cache_key = None
            self._preview_cache_image = None
        except Exception as exc:
            QMessageBox.critical(self, "Import Failed", f"Could not open image:\n{exc}")
            return

        self._update_export_actions()
        self.schedule_update()

    def export_image(self) -> None:
        if self.original_image is None:
            QMessageBox.information(self, "Export", "No processed image to export yet.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Image",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff)",
        )
        if not file_path:
            return
        try:
            settings = self._last_settings or self._collect_settings()
            full_res = self._run_dither(self.original_image, settings, 1.0)
            full_res.save(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Could not save image:\n{exc}")

    def export_gif(self) -> None:
        if self._video_mode != "gif" or not self._video_frames:
            QMessageBox.information(self, "Export", "No GIF loaded in the viewport.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export GIF", "", "GIF (*.gif)")
        if not file_path:
            return
        if Path(file_path).suffix.lower() != ".gif":
            file_path = f"{file_path}.gif"
        try:
            settings = self._last_settings or self._collect_settings()
            frames_out: list[Image.Image] = []
            for frame in self._video_frames:
                dithered = self._run_dither(frame.convert("RGB"), settings, 1.0)
                frames_out.append(dithered)
            if not frames_out:
                raise ValueError("No frames to export.")
            durations = self._video_durations or [self._video_interval_ms] * len(frames_out)
            first = frames_out[0]
            first.save(
                file_path,
                save_all=True,
                append_images=frames_out[1:],
                duration=durations,
                loop=0,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Could not save GIF:\n{exc}")

    def export_video(self) -> None:
        if self._video_mode != "video" or not self._video_path:
            QMessageBox.information(self, "Export", "No video loaded in the viewport.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Video",
            "",
            "MP4 (*.mp4);;AVI (*.avi);;MOV (*.mov)",
        )
        if not file_path:
            return
        path_obj = Path(file_path)
        ext = path_obj.suffix.lower()
        if ext not in {".mp4", ".avi", ".mov"}:
            file_path = f"{file_path}.mp4"
            ext = ".mp4"
        try:
            capture = cv2.VideoCapture(self._video_path)
            if not capture.isOpened():
                raise ValueError("Could not reopen source video.")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 24.0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                raise ValueError("Invalid video dimensions.")
            if ext == ".avi":
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
            else:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise ValueError("Could not open output video for writing.")
            settings = self._last_settings or self._collect_settings()
            while True:
                ret, frame = capture.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb)
                dithered = self._run_dither(pil_frame, settings, 1.0)
                bgr = cv2.cvtColor(np.array(dithered), cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            writer.release()
            capture.release()
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Could not save video:\n{exc}")

    def _preset_sliders(self) -> list[tuple[str, QSlider]]:
        return [
            ("Scale", self.scale_slider),
            ("Line Scale", self.line_scale_slider),
            ("Smoothing Factor", self.smoothing_slider),
            ("Bleed Fraction", self.bleed_slider),
            ("Contrast", self.contrast_slider),
            ("Midtones", self.midtones_slider),
            ("Highlights", self.highlights_slider),
            ("Luminance Threshold", self.luma_slider),
            ("Blur", self.blur_slider),
        ]

    def _load_presets(self) -> None:
        self._preset_files.clear()
        if self.preset_dir.exists():
            for path in sorted(self.preset_dir.glob("*.dset")):
                self._preset_files[path.stem] = path

        current = self.presets_combo.currentText() if hasattr(self, "presets_combo") else "None"
        if hasattr(self, "presets_combo"):
            self.presets_combo.blockSignals(True)
            self.presets_combo.clear()
            self.presets_combo.addItem("None")
            for name in sorted(self._preset_files.keys(), key=str.casefold):
                self.presets_combo.addItem(name)
            if current in self._preset_files:
                self.presets_combo.setCurrentText(current)
            else:
                self.presets_combo.setCurrentIndex(0)
            self.presets_combo.blockSignals(False)

    def _write_preset_file(self, path: Path) -> None:
        lines: list[str] = []
        style = self.style_combo.currentText().strip()
        if style:
            lines.append(f"Style: {style}")
        for name, slider in self._preset_sliders():
            min_val = slider.minimum()
            max_val = slider.maximum()
            value = slider.value()
            if max_val == min_val:
                percent = 0
            else:
                percent = int(round((value - min_val) / float(max_val - min_val) * 100.0))
            lines.append(f"{name}: {percent}%")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _read_preset_file(self, path: Path) -> tuple[dict[str, int], str | None]:
        parsed: dict[str, int] = {}
        style_name: str | None = None
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return parsed, style_name
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            name, value = line.split(":", 1)
            name = name.strip()
            value = value.strip()
            if name.lower() == "style":
                style_name = value
                continue
            if value.endswith("%"):
                value = value[:-1].strip()
            try:
                percent = int(round(float(value)))
            except ValueError:
                continue
            parsed[name] = int(np.clip(percent, 0, 100))
        return parsed, style_name

    def _apply_preset(self, preset_values: dict[str, int], style_name: str | None) -> None:
        sliders = {name: slider for name, slider in self._preset_sliders()}
        for slider in sliders.values():
            slider.blockSignals(True)
        if style_name:
            self.style_combo.blockSignals(True)
            index = self.style_combo.findText(style_name, Qt.MatchFixedString)
            if index >= 0:
                self.style_combo.setCurrentIndex(index)
            self.style_combo.blockSignals(False)
        for name, percent in preset_values.items():
            slider = sliders.get(name)
            if slider is None:
                continue
            min_val = slider.minimum()
            max_val = slider.maximum()
            if max_val == min_val:
                value = min_val
            else:
                value = int(round(min_val + (max_val - min_val) * (percent / 100.0)))
            slider.setValue(int(np.clip(value, min_val, max_val)))
        for slider in sliders.values():
            slider.blockSignals(False)
        self.schedule_update()

    def import_preset(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Preset",
            str(self.preset_dir),
            "Dither Preset (*.dset)",
        )
        if not file_path:
            return
        source = Path(file_path)
        dest_dir = self.preset_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / source.name
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            counter = 1
            while dest.exists():
                dest = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        try:
            shutil.copy2(source, dest)
        except Exception as exc:
            QMessageBox.critical(self, "Presets", f"Could not import preset:\n{exc}")
            return
        self._preset_files[dest.stem] = dest
        self._load_presets()
        if hasattr(self, "presets_combo"):
            self.presets_combo.setCurrentText(dest.stem)

    def on_preset_changed(self) -> None:
        name = self.presets_combo.currentText()
        if not name or name == "None":
            return
        path = self._preset_files.get(name)
        if path is None or not path.exists():
            QMessageBox.information(self, "Presets", "Preset file not found.")
            return
        values, style_name = self._read_preset_file(path)
        if not values and not style_name:
            QMessageBox.information(self, "Presets", "Preset file is empty or invalid.")
            return
        self._apply_preset(values, style_name)

    def save_preset(self) -> None:
        default_name = "Preset"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Preset",
            str(self.preset_dir / f"{default_name}.dset"),
            "Dither Preset (*.dset)",
        )
        if not file_path:
            return
        path = Path(file_path)
        if path.suffix.lower() != ".dset":
            path = path.with_suffix(".dset")
        try:
            self._write_preset_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "Presets", f"Could not save preset:\n{exc}")
            return
        self._preset_files[path.stem] = path
        self._load_presets()
        if hasattr(self, "presets_combo"):
            self.presets_combo.setCurrentText(path.stem)

    def show_about(self) -> None:
        QMessageBox.information(self, "About", "Dither-Dawg\nPrototype UI")

    def reset_controls(self) -> None:
        self.style_combo.setCurrentIndex(0)
        self.presets_combo.setCurrentIndex(0)
        self.scale_slider.setValue(11)
        self.line_scale_slider.setValue(1)
        self.smoothing_slider.setValue(0)
        self.bleed_slider.setValue(0)
        self.palette_category_combo.setCurrentIndex(0)
        self._populate_palette_combo(self.palette_category_combo.currentText())
        self.palette_combo.setCurrentIndex(0)
        self.invert_check.setChecked(False)
        self.contrast_slider.setValue(45)
        self.midtones_slider.setValue(50)
        self.highlights_slider.setValue(50)
        self.luma_slider.setValue(50)
        self.blur_slider.setValue(0)
        self.reset_zoom()
        self.schedule_update()

    def zoom_in(self) -> None:
        self.zoom = min(4.0, self.zoom + 0.25)
        self.schedule_update()

    def zoom_out(self) -> None:
        self.zoom = max(0.25, self.zoom - 0.25)
        self.schedule_update()

    def reset_zoom(self) -> None:
        self.zoom = 1.0
        self.schedule_update()

    def _render_pixmap(self, image: Image.Image) -> None:
        qimage = pil_to_qimage(image)
        pixmap = QPixmap.fromImage(qimage)
        viewport = self.preview_area.viewport().size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            viewport = QSize(800, 600)
        base_size = pixmap.size().scaled(viewport, Qt.KeepAspectRatio)
        scaled_size = QSize(
            max(1, int(base_size.width() * self.zoom)),
            max(1, int(base_size.height() * self.zoom)),
        )
        scaled = pixmap.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.image_label.setFixedSize(scaled.size())
        self.image_label.setText("")

    def update_preview(self) -> None:
        self._update_scheduled = False
        if self.original_image is None:
            self.image_label.setText("Import an image to preview")
            self.image_label.setPixmap(QPixmap())
            self.image_label.adjustSize()
            return

        preview_source, scale_factor = self._get_preview_source()
        settings = self._collect_settings()
        self._last_settings = settings
        self.processed_image = self._run_dither(preview_source, settings, scale_factor)

        self._render_pixmap(self.processed_image)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._preview_cache_key = None
        self._preview_cache_image = None
        if self.processed_image is not None:
            self.schedule_update()


def main() -> int:
    app = QApplication(sys.argv)
    window = DitherWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
