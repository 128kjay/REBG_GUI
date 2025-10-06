"""
Background Remover with Auto Mask
Requires:
    pip install PyQt6 pillow rembg numpy opencv-python
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageQt
import cv2

from rembg import remove, new_session

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider,
    QSpinBox, QMessageBox, QCheckBox, QComboBox
)

# ------------------------------
# Utils
# ------------------------------

def pil_to_qimage(pil_img: Image.Image) -> QImage:
    if pil_img.mode in ("RGB", "RGBA"):
        data = pil_img.tobytes("raw", pil_img.mode)
        if pil_img.mode == "RGB":
            qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format.Format_RGB888)
        else:
            qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format.Format_RGBA8888)
        return qimg
    elif pil_img.mode == "L":
        data = pil_img.tobytes("raw", "L")
        return QImage(data, pil_img.width, pil_img.height, QImage.Format.Format_Grayscale8)
    pil_rgba = pil_img.convert("RGBA")
    data = pil_rgba.tobytes("raw", "RGBA")
    return QImage(data, pil_rgba.width, pil_rgba.height, QImage.Format.Format_RGBA8888)

def qimage_to_numpy(qimg: QImage) -> np.ndarray:
    qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    ptr = qimg.bits()
    ptr.setsize(qimg.width() * qimg.height() * 4)
    arr = np.frombuffer(ptr, np.uint8).reshape((qimg.height(), qimg.width(), 4))
    return arr.copy()

def pil_from_qimage(qimg: QImage) -> Image.Image:
    arr = qimage_to_numpy(qimg)
    return Image.fromarray(arr, mode="RGBA")

def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        qimg = QImage(arr.data, arr.shape[1], arr.shape[0], QImage.Format.Format_Grayscale8)
    elif arr.shape[2] == 3:
        qimg = QImage(arr.data, arr.shape[1], arr.shape[0], 3 * arr.shape[1], QImage.Format.Format_RGB888)
    else:
        qimg = QImage(arr.data, arr.shape[1], arr.shape[0], 4 * arr.shape[1], QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg)

# ------------------------------
# Graphics View
# ------------------------------

class ImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing |
                            QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setMouseTracking(True)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.base_item: QGraphicsPixmapItem | None = None     # original image
        self.mask_item: QGraphicsPixmapItem | None = None     # auto mask overlay (cyan)
        self.overlay_item: QGraphicsPixmapItem | None = None  # strokes overlay (red/green)
        self.preview_item: QGraphicsPixmapItem | None = None  # result preview

        self.image_size = None
        # Strokes: 0=unknown, 1=fg, 2=bg
        self.stroke_mask: np.ndarray | None = None
        # Auto mask (0..255); None until generated
        self.auto_mask: np.ndarray | None = None

        self.brush_size = 20
        self.paint_mode = 1     # 1=FG (left), 2=BG (right)
        self.overlay_alpha = 140
        self.mask_alpha = 120   # cyan overlay alpha

        self._panning = False
        self._last_pan = None

    def clear_all(self):
        self.scene.clear()
        self.base_item = None
        self.mask_item = None
        self.overlay_item = None
        self.preview_item = None
        self.image_size = None
        self.stroke_mask = None
        self.auto_mask = None

    def set_image(self, qimg: QImage):
        self.clear_all()
        self.base_item = QGraphicsPixmapItem(QPixmap.fromImage(qimg))
        self.scene.addItem(self.base_item)
        self.image_size = (qimg.height(), qimg.width())

        self.stroke_mask = np.zeros(self.image_size, dtype=np.uint8)
        self.auto_mask = None

        # Auto mask overlay (cyan, below strokes)
        mask_rgba = np.zeros((self.image_size[0], self.image_size[1], 4), dtype=np.uint8)
        self.mask_item = QGraphicsPixmapItem(numpy_to_qpixmap(mask_rgba))
        self.mask_item.setZValue(6)
        self.scene.addItem(self.mask_item)

        # Strokes overlay (green/red)
        overlay = np.zeros((self.image_size[0], self.image_size[1], 4), dtype=np.uint8)
        self.overlay_item = QGraphicsPixmapItem(numpy_to_qpixmap(overlay))
        self.overlay_item.setZValue(10)
        self.scene.addItem(self.overlay_item)

        self.preview_item = QGraphicsPixmapItem()
        self.preview_item.setZValue(5)
        self.scene.addItem(self.preview_item)

        self.setSceneRect(self.scene.itemsBoundingRect())
        self.fitInView(self.base_item, Qt.AspectRatioMode.KeepAspectRatio)

    def current_base_qimage(self) -> QImage | None:
        if not self.base_item:
            return None
        return self.base_item.pixmap().toImage()

    # -------- Painting / UI --------

    def set_brush_size(self, s: int):
        self.brush_size = max(1, int(s))

    def set_overlay_opacity(self, v: int):
        self.overlay_alpha = int(np.clip(v, 0, 255))
        self.redraw_stroke_overlay()

    def set_mask_opacity(self, v: int):
        self.mask_alpha = int(np.clip(v, 0, 255))
        self.redraw_auto_mask_overlay()

    def clear_strokes(self):
        if self.stroke_mask is not None:
            self.stroke_mask.fill(0)
        self.redraw_stroke_overlay()

    def clear_auto_mask(self):
        self.auto_mask = None
        self.redraw_auto_mask_overlay()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._last_pan = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if not self.base_item:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.paint_mode = 1
            self.apply_paint(event.position())
        elif event.button() == Qt.MouseButton.RightButton:
            self.paint_mode = 2
            self.apply_paint(event.position())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._panning and self._last_pan is not None:
            delta = event.position() - self._last_pan
            self._last_pan = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            return
        buttons = event.buttons()
        if self.base_item and (buttons & Qt.MouseButton.LeftButton or buttons & Qt.MouseButton.RightButton):
            self.apply_paint(event.position())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        if not self.base_item:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def view_to_image_coords(self, view_pos: QtCore.QPointF) -> tuple[int, int] | None:
        if not self.base_item:
            return None
        scene_pos = self.mapToScene(QtCore.QPoint(int(view_pos.x()), int(view_pos.y())))
        item_pos = self.base_item.mapFromScene(scene_pos)
        x, y = int(item_pos.x()), int(item_pos.y())
        if (0 <= x < self.image_size[1]) and (0 <= y < self.image_size[0]):
            return (x, y)
        return None

    def apply_paint(self, view_pos: QtCore.QPointF):
        coord = self.view_to_image_coords(view_pos)
        if coord is None:
            return
        x, y = coord
        r = self.brush_size // 2
        y0, y1 = max(0, y - r), min(self.image_size[0], y + r + 1)
        x0, x1 = max(0, x - r), min(self.image_size[1], x + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
        if self.paint_mode == 1:
            self.stroke_mask[y0:y1, x0:x1][circle] = 1
        else:
            self.stroke_mask[y0:y1, x0:x1][circle] = 2
        self.redraw_stroke_overlay()

    def redraw_stroke_overlay(self):
        if self.overlay_item is None or self.stroke_mask is None:
            return
        h, w = self.stroke_mask.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        fg = self.stroke_mask == 1
        bg = self.stroke_mask == 2
        overlay[fg, 1] = 255   # green
        overlay[bg, 0] = 255   # red
        overlay[fg | bg, 3] = self.overlay_alpha
        self.overlay_item.setPixmap(numpy_to_qpixmap(overlay))

    def redraw_auto_mask_overlay(self):
        if self.mask_item is None:
            return
        if self.auto_mask is None:
            h = self.image_size[0] if self.image_size else 1
            w = self.image_size[1] if self.image_size else 1
            blank = np.zeros((h, w, 4), dtype=np.uint8)
            self.mask_item.setPixmap(numpy_to_qpixmap(blank))
            return
        # Cyan overlay where auto mask > 0
        mask_bin = (self.auto_mask > 0).astype(np.uint8)
        h, w = mask_bin.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[mask_bin == 1, 1] = 255  # G
        rgba[mask_bin == 1, 2] = 255  # B
        rgba[mask_bin == 1, 3] = self.mask_alpha
        self.mask_item.setPixmap(numpy_to_qpixmap(rgba))

    def set_preview(self, qimg: QImage | None):
        if self.preview_item is None:
            return
        if qimg is None:
            self.preview_item.setPixmap(QPixmap())
        else:
            self.preview_item.setPixmap(QPixmap.fromImage(qimg))

# ------------------------------
# Main Window
# ------------------------------

@dataclass
class RemovalParams:
    model_name: str = "isnet-general-use"
    alpha_matting: bool = True
    fg_thresh: int = 240
    bg_thresh: int = 10
    erode_size: int = 0
    post_process_mask: bool = False
    dilate_kernel: int = 3
    feather_kernel: int = 5
    fg_priority: int = 255
    bg_priority: int = 0

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Background Remover")
        self.resize(1280, 820)

        self.view = ImageView(self)
        central = QWidget(self)
        self.setCentralWidget(central)

        # --- Top controls ---
        open_btn = QPushButton("Open")
        open_btn.clicked.connect(self.open_image)

        save_btn = QPushButton("Save PNG")
        save_btn.clicked.connect(self.save_result)

        clear_btn = QPushButton("Clear Strokes")
        clear_btn.clicked.connect(self.view.clear_strokes)

        clear_mask_btn = QPushButton("Clear Auto Mask")
        clear_mask_btn.clicked.connect(self.view.clear_auto_mask)

        # Brush size
        brush_label = QLabel("Brush")
        brush_spin = QSpinBox()
        brush_spin.setRange(1, 200)
        brush_spin.setValue(20)
        brush_spin.valueChanged.connect(self.view.set_brush_size)

        # Strokes opacity
        op_label = QLabel("Strokes α")
        op_slider = QSlider(Qt.Orientation.Horizontal)
        op_slider.setRange(0, 255)
        op_slider.setValue(140)
        op_slider.valueChanged.connect(self.view.set_overlay_opacity)

        # Auto mask opacity
        m_label = QLabel("Mask α")
        m_slider = QSlider(Qt.Orientation.Horizontal)
        m_slider.setRange(0, 255)
        m_slider.setValue(120)
        m_slider.valueChanged.connect(self.view.set_mask_opacity)

        # Engine select
        engine_label = QLabel("Auto Mask Engine")
        self.engine_combo = QComboBox()
        self.engine_combo.addItems([
            "rembg (isnet-general-use)",
            "rembg (u2net_human_seg)",
            "GrabCut (OpenCV)"
        ])

        auto_btn = QPushButton("Auto Mask")
        auto_btn.setStyleSheet("font-weight: bold;")
        auto_btn.clicked.connect(self.run_auto_mask)

        apply_btn = QPushButton("Apply Removal")
        apply_btn.setStyleSheet("font-weight: bold;")
        apply_btn.clicked.connect(self.apply_removal)

        self.chk_hide_overlay_on_preview = QCheckBox("Hide strokes on preview")
        self.chk_hide_overlay_on_preview.setChecked(False)

        # Layout
        tools = QHBoxLayout()
        tools.addWidget(open_btn)
        tools.addWidget(save_btn)
        tools.addSpacing(12)
        tools.addWidget(clear_btn)
        tools.addWidget(clear_mask_btn)
        tools.addSpacing(12)
        tools.addWidget(brush_label)
        tools.addWidget(brush_spin)
        tools.addWidget(op_label)
        tools.addWidget(op_slider)
        tools.addWidget(m_label)
        tools.addWidget(m_slider)
        tools.addStretch(1)
        tools.addWidget(engine_label)
        tools.addWidget(self.engine_combo)
        tools.addSpacing(8)
        tools.addWidget(auto_btn)
        tools.addSpacing(8)
        tools.addWidget(apply_btn)
        tools.addSpacing(10)
        tools.addWidget(self.chk_hide_overlay_on_preview)

        root = QVBoxLayout(central)
        root.addLayout(tools)
        root.addWidget(self.view, stretch=1)

        # Model session
        self.session = None
        self.params = RemovalParams()

        self._add_shortcuts()

    def _add_shortcuts(self):
        act_open = QAction(self)
        act_open.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self.open_image)
        self.addAction(act_open)

        act_save = QAction(self)
        act_save.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self.save_result)
        self.addAction(act_save)

        act_apply = QAction(self)
        act_apply.setShortcut("Ctrl+R")
        act_apply.triggered.connect(self.apply_removal)
        self.addAction(act_apply)

        act_clear = QAction(self)
        act_clear.setShortcut("C")
        act_clear.triggered.connect(self.view.clear_strokes)
        self.addAction(act_clear)

        act_auto = QAction(self)
        act_auto.setShortcut("A")
        act_auto.triggered.connect(self.run_auto_mask)
        self.addAction(act_auto)

    # ---------- File ops ----------

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return
        try:
            pil = Image.open(path).convert("RGBA")
        except Exception as e:
            QMessageBox.critical(self, "Open Failed", f"Could not open image:\n{e}")
            return
        qimg = pil_to_qimage(pil)
        self.view.set_image(qimg)
        self.view.set_preview(None)
        self.setWindowTitle(f"Guided Background Remover — Auto Mask — {os.path.basename(path)}")

    def save_result(self):
        if self.view.preview_item is None:
            return
        pix = self.view.preview_item.pixmap()
        if pix.isNull():
            QMessageBox.information(self, "Nothing to Save", "No preview result yet.")
            return
        qimg = pix.toImage()
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG with Alpha", "", "PNG Image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        if qimg.save(path, "PNG"):
            QMessageBox.information(self, "Saved", f"Saved:\n{path}")
        else:
            QMessageBox.critical(self, "Save Failed", "Could not save the image.")

    # ---------- Auto Mask ----------

    def ensure_session(self, model_name: str):
        if self.session is None or getattr(self, "_session_name", None) != model_name:
            try:
                self.session = new_session(model_name)
                self._session_name = model_name
            except Exception as e:
                QMessageBox.critical(self, "Model Load Failed", f"Could not load model:\n{e}")
                raise

    def run_auto_mask(self):
        base_qimg = self.view.current_base_qimage()
        if base_qimg is None:
            QMessageBox.information(self, "No Image", "Open an image first.")
            return

        engine = self.engine_combo.currentText()
        img_rgba = pil_from_qimage(base_qimg)
        img_rgb = img_rgba.convert("RGB")

        try:
            if engine.startswith("rembg"):
                model = "isnet-general-use" if "isnet" in engine else "u2net_human_seg"
                self.ensure_session(model)
                mask_img = remove(
                    img_rgb,
                    session=self.session,
                    alpha_matting=self.params.alpha_matting,
                    alpha_matting_foreground_threshold=self.params.fg_thresh,
                    alpha_matting_background_threshold=self.params.bg_thresh,
                    alpha_matting_erode_size=self.params.erode_size,
                    post_process_mask=self.params.post_process_mask,
                    only_mask=True
                )
                auto_mask = np.array(mask_img).astype(np.uint8)  # 0..255
            else:
                # GrabCut: initialize with a rectangle inset to exclude borders
                arr = np.array(img_rgb)  # HxWx3
                h, w = arr.shape[:2]
                mask = np.zeros((h, w), np.uint8)
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                inset = max(5, int(min(h, w) * 0.03))
                rect = (inset, inset, w - 2 * inset, h - 2 * inset)
                cv2.grabCut(arr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                # Convert GrabCut mask to 0/255
                mask_bin = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                auto_mask = mask_bin

            self.view.auto_mask = auto_mask
            self.view.redraw_auto_mask_overlay()

        except Exception as e:
            QMessageBox.critical(self, "Auto Mask Error", f"Failed to compute auto mask:\n{e}")
            return

    # ---------- Apply Removal ----------

    def apply_removal(self):
        base_qimg = self.view.current_base_qimage()
        if base_qimg is None:
            QMessageBox.information(self, "No Image", "Open an image first.")
            return

        img_rgba = pil_from_qimage(base_qimg)   # RGBA
        img_rgb = img_rgba.convert("RGB")
        h, w = img_rgb.size[1], img_rgb.size[0]

        # Start from auto_mask if present otherwise fallback to rembg
        base_mask = None
        if self.view.auto_mask is not None:
            base_mask = self.view.auto_mask
        else:
            # fallback: rembg mask using current default model
            try:
                self.ensure_session(self.params.model_name)
                base_mask_img = remove(
                    img_rgb,
                    session=self.session,
                    alpha_matting=self.params.alpha_matting,
                    alpha_matting_foreground_threshold=self.params.fg_thresh,
                    alpha_matting_background_threshold=self.params.bg_thresh,
                    alpha_matting_erode_size=self.params.erode_size,
                    post_process_mask=self.params.post_process_mask,
                    only_mask=True
                )
                base_mask = np.array(base_mask_img).astype(np.uint8)
            except Exception as e:
                QMessageBox.critical(self, "rembg Error", f"Background estimation failed:\n{e}")
                return

        # Fuse with strokes FG=255, BG=0
        fused = base_mask.copy()
        stroke = self.view.stroke_mask
        if stroke is not None and stroke.shape == fused.shape:
            fused[stroke == 1] = self.params.fg_priority
            fused[stroke == 2] = self.params.bg_priority

        # Light dilation and feather
        if self.params.dilate_kernel > 1:
            k = self.params.dilate_kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            fused = cv2.dilate(fused, kernel, iterations=1)

        if self.params.feather_kernel > 1 and self.params.feather_kernel % 2 == 1:
            fused = cv2.GaussianBlur(fused, (self.params.feather_kernel, self.params.feather_kernel), 0)

        rgb = np.array(img_rgb)
        rgba = np.dstack([rgb, fused]).astype(np.uint8)

        # Optionally hide strokes on preview
        if self.chk_hide_overlay_on_preview.isChecked():
            self.view.overlay_item.setVisible(False)
        else:
            self.view.overlay_item.setVisible(True)

        self.view.set_preview(numpy_to_qpixmap(rgba).toImage())

# ------------------------------
# Entry
# ------------------------------

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
