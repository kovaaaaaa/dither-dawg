from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageSequence
from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QSize,
    Qt,
    QTimer,
    QUrl,
)
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
    QGraphicsOpacityEffect,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from dither_core import (
    BUILTIN_PALETTES,
    MODULATION_METHODS,
    Palette,
    _build_palette_data,
    apply_dither_optimized,
    load_palette_library,
)

def pil_to_qimage(image: Image.Image) -> QImage:
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    data = image.tobytes("raw", "RGBA")
    qimage = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
    return qimage.copy()


APP_NAME = "Dither-Dawg"
APP_VERSION = "1.0.0"
GITHUB_REPO = "owner/repo"
UPDATE_CHECK_TIMEOUT_SEC = 4.0
UPDATE_CHECK_DELAY_MS = 1200
UPDATE_CHECK_ENV = "DITHER_DAWG_DISABLE_UPDATE_CHECK"


@dataclass(frozen=True)
class UpdateInfo:
    tag: str
    url: str
    name: str = ""


def _updates_enabled() -> bool:
    value = os.environ.get(UPDATE_CHECK_ENV, "").strip().lower()
    return value not in {"1", "true", "yes", "on"}


def _parse_version(version: str) -> tuple[int, ...]:
    parts = [int(part) for part in re.findall(r"\d+", version)]
    return tuple(parts)


def _is_newer_version(candidate: str, current: str) -> bool:
    candidate_parts = _parse_version(candidate)
    current_parts = _parse_version(current)
    if not candidate_parts:
        return False
    length = max(len(candidate_parts), len(current_parts))
    candidate_parts += (0,) * (length - len(candidate_parts))
    current_parts += (0,) * (length - len(current_parts))
    return candidate_parts > current_parts


def _fetch_latest_release(repo: str, timeout: float) -> UpdateInfo | None:
    if not repo or "/" not in repo:
        return None
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    headers = {"User-Agent": f"{APP_NAME}/{APP_VERSION}"}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", errors="ignore"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None

    tag = str(payload.get("tag_name") or "").strip()
    html_url = str(payload.get("html_url") or "").strip()
    name = str(payload.get("name") or "").strip()
    if not tag or not html_url:
        return None
    return UpdateInfo(tag=tag, url=html_url, name=name)


class ZoomScrollArea(QScrollArea):
    def __init__(self, zoom_callback: Callable[[int], None], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._zoom_callback = zoom_callback

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta != 0 and self._zoom_callback is not None:
            self._zoom_callback(delta)
            event.accept()
            return
        super().wheelEvent(event)


class AnimatedComboDelegate(QStyledItemDelegate):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._start_time = 0.0
        self.duration_ms = 220
        self.stagger_ms = 22
        self.offset_px = 10

    def start(self) -> None:
        self._start_time = time.perf_counter()

    def paint(self, painter, option, index) -> None:
        if self._start_time <= 0:
            return super().paint(painter, option, index)

        elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
        delay = index.row() * self.stagger_ms
        progress = (elapsed_ms - delay) / float(self.duration_ms)
        if progress < 0.0:
            progress = 0.0
        elif progress > 1.0:
            progress = 1.0
        if progress <= 0.0:
            return

        painter.save()
        if progress < 1.0:
            painter.setOpacity(progress)
            painter.translate(int((1.0 - progress) * self.offset_px), 0)
        super().paint(painter, option, index)
        painter.restore()


class AnimatedComboBox(QComboBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._popup_anim: QPropertyAnimation | None = None
        self._popup_timer: QTimer | None = None

    def _ensure_delegate(self) -> AnimatedComboDelegate:
        view = self.view()
        delegate = getattr(view, "_animated_delegate", None)
        if delegate is None:
            delegate = AnimatedComboDelegate(view)
            view.setItemDelegate(delegate)
            view._animated_delegate = delegate
        return delegate

    def _start_item_animation(self) -> None:
        view = self.view()
        delegate = self._ensure_delegate()
        delegate.start()

        if self._popup_timer is not None:
            self._popup_timer.stop()
            self._popup_timer.deleteLater()

        timer = QTimer(view)
        timer.setInterval(16)

        def _tick() -> None:
            view.viewport().update()
            rows = view.model().rowCount() if view.model() is not None else 0
            total_ms = delegate.duration_ms + delegate.stagger_ms * max(0, rows - 1)
            elapsed_ms = (time.perf_counter() - delegate._start_time) * 1000.0
            if elapsed_ms >= total_ms:
                timer.stop()
                timer.deleteLater()
                self._popup_timer = None

        timer.timeout.connect(_tick)
        self._popup_timer = timer
        timer.start()

    def showPopup(self) -> None:
        view = self.view()
        view.setWindowOpacity(0.0)
        self._start_item_animation()
        super().showPopup()

        self._popup_anim = QPropertyAnimation(view, b"windowOpacity", view)
        self._popup_anim.setDuration(240)
        self._popup_anim.setStartValue(0.0)
        self._popup_anim.setEndValue(1.0)
        self._popup_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._popup_anim.start()




class DitherWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Dither-Dawg v1.0.0")
        self.resize(1280, 760)
        self.setStyleSheet(self._style_sheet())

        base_dir = Path(__file__).resolve().parent
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
        self._update_check_inflight = False
        self._resize_update_pending = False
        self._resize_update_delay_ms = 140
        self._sidebar_animated = False
        self._sidebar_min_width = 320
        self._sidebar_max_width = 380
        self._animate_preview_next = False

        self._build_menu()

        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        self.preview_area = self._build_preview()
        main_layout.addWidget(self.preview_area, 1)
        self._setup_preview_animation()

        controls = self._build_controls()
        self.control_scroll = QScrollArea()
        self.control_scroll.setWidget(controls)
        self.control_scroll.setWidgetResizable(True)
        self.control_scroll.setFrameShape(QFrame.NoFrame)
        self.control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.control_scroll.setMinimumWidth(0)
        self.control_scroll.setMaximumWidth(0)
        main_layout.addWidget(self.control_scroll, 0)

        self._load_presets()
        self._load_starter_image()
        self._schedule_update_check()
        self._schedule_sidebar_animation()

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
            self._animate_preview_next = True
            self.schedule_update()
        except Exception:
            return

    def _style_sheet(self) -> str:
        return """
        QMainWindow {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #0f1012, stop:0.6 #0b0c0f, stop:1 #0a0b0d);
        }
        QWidget { color: #d9dadb; font-family: "Bahnschrift", "Segoe UI"; font-size: 10.5pt; }
        #sidebar {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #181b1f, stop:1 #121418);
            border: 1px solid #2a2e34;
            border-radius: 12px;
        }
        QLabel#sectionTitle { color: #9aa6b2; font-size: 9pt; letter-spacing: 0.8px; }
        QLabel#logo { font-size: 19pt; font-weight: 700; letter-spacing: 2px; color: #f1f3f5; }
        QLabel#version { color: #8c97a3; font-size: 9pt; letter-spacing: 1.6px; }
        QLabel#hint { color: #8e9aa6; font-size: 9.5pt; }

        QPushButton {
            background: #1f2329;
            border: 1px solid #323842;
            padding: 7px 12px;
            border-radius: 8px;
        }
        QPushButton:hover { background: #262c34; border-color: #3a424e; }
        QPushButton:pressed { background: #1b1f25; }
        QPushButton:disabled { color: #6f7680; background: #171a1f; border-color: #242a33; }

        QComboBox {
            background: #1f2329;
            border: 1px solid #323842;
            padding: 5px 10px;
            border-radius: 8px;
        }
        QComboBox:hover { border-color: #3a424e; }
        QComboBox::drop-down { border: none; width: 18px; }
        QComboBox::down-arrow { image: none; border: 2px solid #9aa6b2; width: 6px; height: 6px;
            border-top: none; border-left: none; margin-right: 8px; }
        QComboBox QAbstractItemView {
            background: #14181d;
            border: 1px solid #2a2f36;
            padding: 6px;
            outline: 0;
            selection-background-color: #2b3440;
        }

        QSlider::groove:horizontal {
            height: 6px;
            background: #262b33;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            width: 16px;
            margin: -5px 0;
            background: #e6e9ed;
            border: 1px solid #2c323b;
            border-radius: 8px;
        }
        QSlider::sub-page:horizontal { background: #3a4a5f; border-radius: 3px; }

        QCheckBox { spacing: 8px; }
        QCheckBox::indicator {
            width: 16px; height: 16px;
            border-radius: 4px;
            border: 1px solid #3a4049;
            background: #171b20;
        }
        QCheckBox::indicator:checked {
            background: #3a4a5f;
            border-color: #4a5b73;
        }

        QScrollArea { background: #0b0c0f; border: 1px solid #232833; border-radius: 10px; }
        QScrollBar:vertical { background: #0b0c0f; width: 10px; margin: 6px 0; border-radius: 5px; }
        QScrollBar::handle:vertical { background: #2a313b; border-radius: 5px; min-height: 20px; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

        QMenuBar { background: #0f1114; color: #cfd4da; }
        QMenuBar::item:selected { background: #1f252d; }
        QMenu { background: #13171c; color: #d6dbe1; border: 1px solid #2a2f36; }
        QMenu::item:selected { background: #2b3440; }

        QFrame#divider { background: #2a2f36; max-height: 1px; }
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
        check_updates_action = QAction("Check for Updates...", self)

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
        check_updates_action.triggered.connect(lambda: self.check_for_updates(manual=True))
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
        help_menu.addAction(check_updates_action)
        self._update_export_actions()

    def _build_preview(self) -> QScrollArea:
        self.image_label = QLabel("Import an image to preview")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(500, 400)
        self.image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.image_label.setStyleSheet(
            "QLabel { background: #111114; color: #c0c0c0; border: 1px solid #242424; }"
        )

        area = ZoomScrollArea(self._handle_zoom_wheel)
        area.setWidget(self.image_label)
        area.setWidgetResizable(False)
        area.setAlignment(Qt.AlignCenter)
        area.setFrameShape(QFrame.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        return area

    def _setup_preview_animation(self) -> None:
        self._preview_opacity = QGraphicsOpacityEffect(self.image_label)
        self.image_label.setGraphicsEffect(self._preview_opacity)
        self._preview_opacity.setOpacity(1.0)
        self._preview_fade_anim = QPropertyAnimation(self._preview_opacity, b"opacity", self)
        self._preview_fade_anim.setDuration(180)
        self._preview_fade_anim.setEasingCurve(QEasingCurve.OutCubic)

    def _animate_preview_fade(self) -> None:
        if not hasattr(self, "_preview_fade_anim"):
            return
        self._preview_fade_anim.stop()
        self._preview_opacity.setOpacity(0.0)
        self._preview_fade_anim.setStartValue(0.0)
        self._preview_fade_anim.setEndValue(1.0)
        self._preview_fade_anim.start()

    def _schedule_sidebar_animation(self) -> None:
        if self._sidebar_animated:
            return
        self._sidebar_animated = True
        QTimer.singleShot(0, self._animate_sidebar_in)

    def _animate_sidebar_in(self) -> None:
        effect = QGraphicsOpacityEffect(self.control_scroll)
        self.control_scroll.setGraphicsEffect(effect)
        effect.setOpacity(0.0)

        width_anim = QPropertyAnimation(self.control_scroll, b"maximumWidth", self)
        width_anim.setStartValue(0)
        width_anim.setEndValue(self._sidebar_max_width)
        width_anim.setDuration(420)
        width_anim.setEasingCurve(QEasingCurve.OutCubic)

        opacity_anim = QPropertyAnimation(effect, b"opacity", self)
        opacity_anim.setStartValue(0.0)
        opacity_anim.setEndValue(1.0)
        opacity_anim.setDuration(420)
        opacity_anim.setEasingCurve(QEasingCurve.OutCubic)

        self._sidebar_width_anim = width_anim
        self._sidebar_opacity_anim = opacity_anim

        width_anim.start()
        opacity_anim.start()
        width_anim.finished.connect(self._lock_sidebar_width)

    def _lock_sidebar_width(self) -> None:
        self.control_scroll.setMinimumWidth(self._sidebar_min_width)
        self.control_scroll.setMaximumWidth(self._sidebar_max_width)

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

        hint = QLabel("Ctrl+Shift+? for help")
        hint.setObjectName("hint")
        layout.addWidget(hint)

        logo = QLabel("DITHER-DAWG")
        logo.setObjectName("logo")
        logo.setAlignment(Qt.AlignCenter)
        logo.setMinimumHeight(60)
        layout.addWidget(logo)
        version_label = QLabel(f"v{APP_VERSION}")
        version_label.setObjectName("version")
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)

        layout.addWidget(self._divider())

        layout.addWidget(self._section_title("Style"))
        self.style_combo = AnimatedComboBox()
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
        self.presets_combo = AnimatedComboBox()
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
        self.palette_category_combo = AnimatedComboBox()
        if self.palette_categories:
            self.palette_category_combo.addItems(self.palette_categories)
        else:
            self.palette_category_combo.addItems(["Built-in"])
        layout.addWidget(self.palette_category_combo)

        layout.addWidget(self._section_title("Palette"))
        self.palette_combo = AnimatedComboBox()
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
        palette_data = self._get_palette_data()
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
            self._animate_preview_next = True
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
        self._animate_preview_next = True
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
            "Zoom: Mouse Wheel (over preview)\nZoom In: Ctrl+=\nZoom Out: Ctrl+-\nReset Zoom: Ctrl+0",
        )

    def toggle_invert_from_menu(self, checked: bool) -> None:
        if hasattr(self, "invert_check"):
            self.invert_check.setChecked(checked)

    def schedule_update(self) -> None:
        if self._update_scheduled:
            return
        self._update_scheduled = True
        QTimer.singleShot(0, self.update_preview)

    def _schedule_resize_update(self) -> None:
        if self._resize_update_pending:
            return
        self._resize_update_pending = True
        QTimer.singleShot(self._resize_update_delay_ms, self._finish_resize_update)

    def _finish_resize_update(self) -> None:
        self._resize_update_pending = False
        self.schedule_update()

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
            self._animate_preview_next = True
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
        category = self.palette_category_combo.currentText().strip()
        palette = self.palette_combo.currentText().strip()
        if category:
            lines.append(f"Palette Category: {category}")
        if palette:
            lines.append(f"Palette: {palette}")
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

    def _read_preset_file(self, path: Path) -> tuple[dict[str, int], str | None, str | None, str | None]:
        parsed: dict[str, int] = {}
        style_name: str | None = None
        palette_category: str | None = None
        palette_name: str | None = None
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return parsed, style_name, palette_category, palette_name
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
            if name.lower() == "palette category":
                palette_category = value
                continue
            if name.lower() == "palette":
                palette_name = value
                continue
            if value.endswith("%"):
                value = value[:-1].strip()
            try:
                percent = int(round(float(value)))
            except ValueError:
                continue
            parsed[name] = int(np.clip(percent, 0, 100))
        return parsed, style_name, palette_category, palette_name

    def _apply_preset(
        self,
        preset_values: dict[str, int],
        style_name: str | None,
        palette_category: str | None,
        palette_name: str | None,
    ) -> None:
        sliders = {name: slider for name, slider in self._preset_sliders()}
        for slider in sliders.values():
            slider.blockSignals(True)
        if style_name:
            self.style_combo.blockSignals(True)
            index = self.style_combo.findText(style_name, Qt.MatchFixedString)
            if index >= 0:
                self.style_combo.setCurrentIndex(index)
            self.style_combo.blockSignals(False)
        if palette_category or palette_name:
            self.palette_category_combo.blockSignals(True)
            self.palette_combo.blockSignals(True)
            if palette_category:
                cat_index = self.palette_category_combo.findText(palette_category, Qt.MatchFixedString)
                if cat_index >= 0:
                    self.palette_category_combo.setCurrentIndex(cat_index)
                    self._populate_palette_combo(palette_category)
            if palette_name:
                pal_index = self.palette_combo.findText(palette_name, Qt.MatchFixedString)
                if pal_index >= 0:
                    self.palette_combo.setCurrentIndex(pal_index)
                else:
                    built_in_index = self.palette_category_combo.findText("Built-in", Qt.MatchFixedString)
                    if built_in_index >= 0:
                        self.palette_category_combo.setCurrentIndex(built_in_index)
                        self._populate_palette_combo("Built-in")
                        if self.palette_combo.count() > 0:
                            self.palette_combo.setCurrentIndex(0)
            self.palette_combo.blockSignals(False)
            self.palette_category_combo.blockSignals(False)
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
        self._update_palette_swatches()
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
        values, style_name, palette_category, palette_name = self._read_preset_file(path)
        if not values and not style_name and not palette_category and not palette_name:
            QMessageBox.information(self, "Presets", "Preset file is empty or invalid.")
            return
        self._apply_preset(values, style_name, palette_category, palette_name)

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
        QMessageBox.information(self, "About", f"Dither-Dawg\nVersion {APP_VERSION}")

    def _schedule_update_check(self) -> None:
        if not _updates_enabled():
            return
        if not GITHUB_REPO or GITHUB_REPO == "owner/repo":
            return
        QTimer.singleShot(UPDATE_CHECK_DELAY_MS, lambda: self.check_for_updates(manual=False))

    def check_for_updates(self, manual: bool = False) -> None:
        if self._update_check_inflight:
            return
        if not _updates_enabled():
            if manual:
                QMessageBox.information(self, "Updates", "Update checks are disabled.")
            return
        if not GITHUB_REPO or GITHUB_REPO == "owner/repo":
            if manual:
                QMessageBox.information(
                    self,
                    "Updates",
                    "Update checks are not configured.\nSet GITHUB_REPO in app.py to enable them.",
                )
            return

        self._update_check_inflight = True

        def worker() -> None:
            info = _fetch_latest_release(GITHUB_REPO, UPDATE_CHECK_TIMEOUT_SEC)
            QTimer.singleShot(0, lambda: self._finish_update_check(info, manual))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_update_check(self, info: UpdateInfo | None, manual: bool) -> None:
        self._update_check_inflight = False
        if info is None:
            if manual:
                QMessageBox.information(self, "Updates", "Could not check for updates right now.")
            return

        latest_tag = info.tag
        if _is_newer_version(latest_tag, APP_VERSION):
            label = f"{latest_tag}"
            if info.name and info.name != latest_tag:
                label = f"{latest_tag} ({info.name})"
            message = (
                f"A newer version is available.\nCurrent: {APP_VERSION}\nLatest: {label}\n\n"
                "Open the release page?"
            )
            if (
                QMessageBox.question(self, "Update available", message, QMessageBox.Yes | QMessageBox.No)
                == QMessageBox.Yes
            ):
                QDesktopServices.openUrl(QUrl(info.url))
        else:
            if manual:
                QMessageBox.information(self, "Updates", f"You're up to date.\nCurrent version: {APP_VERSION}")

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

    def _handle_zoom_wheel(self, delta: int) -> None:
        if self.original_image is None:
            return
        if delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

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
        transform_mode = Qt.SmoothTransformation
        if scaled_size.width() > pixmap.width() or scaled_size.height() > pixmap.height():
            transform_mode = Qt.FastTransformation
        scaled = pixmap.scaled(scaled_size, Qt.KeepAspectRatio, transform_mode)
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
        if self._animate_preview_next:
            self._animate_preview_next = False
            self._animate_preview_fade()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._preview_cache_key = None
        self._preview_cache_image = None
        if self.processed_image is not None:
            self._render_pixmap(self.processed_image)
        self._schedule_resize_update()


def main() -> int:
    app = QApplication(sys.argv)
    window = DitherWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
