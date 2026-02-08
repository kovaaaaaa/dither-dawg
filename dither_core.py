from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


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
    palette_array, palette_luma, levels = palette_data

    work = img.convert("RGB")
    if invert:
        work = ImageOps.invert(work)
    work = apply_image_adjustments(work, brightness, contrast, blur, sharpen)

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
            indices = _error_diffuse_1d(gray_mod, palette_luma, axis="y")
        elif method == "Modulated Diffuse X":
            gray_mod = apply_modulation(gray_array, "x", mod_amp * 1.1, mod_per, "sine", phase)
            indices = _error_diffuse_1d(gray_mod, palette_luma, axis="x")
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
            indices = _error_diffuse_1d(gray_mod, palette_luma, axis="x")
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
