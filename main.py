"""PyQt5 GUI for real-time microplastic detection and counting using YOLOv8.

Features:
- Background inference thread (keeps UI responsive)
- Draws bounding boxes around detected microplastics
- Displays only the total count prominently in the UI
- Live controls for confidence and NMS IoU thresholds
- Uses camera index 1 by default (configurable)
- HiDPI enabled for crisp UI on high-DPI displays

Notes:
- Requires PyQt5 (pip install PyQt5) and ultralytics, opencv-python
- If you prefer to run without GUI, keep the previous CLI script as a fallback (not included here).
"""

import sys
import os
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtCore import pyqtSignal, Qt
except Exception as e:
    print('PyQt5 not installed. Please install with: pip install PyQt5')
    raise


# Default configuration
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), 'best.pt')
DEFAULT_CAMERA_INDEX = 1
# Detection thresholds (change these in code only)
CONF_THRESHOLD = 0.03
IOU_THRESHOLD = 0.30


class InferenceThread(QtCore.QThread):
    # emits a dict: {'frame': np.ndarray (RGB), 'boxes': list of [x1,y1,x2,y2], 'count': int}
    frame_data = pyqtSignal(object)

    def __init__(self, weights, camera_index=1, device=None, parent=None):
        super().__init__(parent)
        self.weights = weights
        self.camera_index = camera_index
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._run_flag = False
        self.conf = 0.25
        self.iou = 0.45
        self.model = None

    def run(self):
        # Load model once in this thread to avoid blocking UI
        try:
            self.model = YOLO(self.weights)
        except Exception as e:
            print('Failed to load model:', e)
            return

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f'Cannot open camera index {self.camera_index}')
            return

        self._run_flag = True

        prev_time = time.time()
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # Run inference
            try:
                results = self.model(frame, conf=self.conf, iou=self.iou, device=self.device)
            except Exception as e:
                print('Inference error:', e)
                results = []

            # Prepare boxes and count (no drawing here)
            count = 0
            boxes_list = []
            if len(results) > 0:
                res = results[0]
                try:
                    xy = res.boxes.xyxy
                    if hasattr(xy, 'cpu'):
                        boxes = xy.cpu().numpy()
                    else:
                        boxes = np.array(xy)
                    if boxes.size != 0:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box[:4])
                            boxes_list.append([x1, y1, x2, y2])
                        count = len(boxes_list)
                except Exception:
                    # fallback older api
                    try:
                        xyxy = res.boxes.xyxy
                        for b in xyxy:
                            x1, y1, x2, y2 = map(int, b[:4])
                            boxes_list.append([x1, y1, x2, y2])
                        count = len(boxes_list)
                    except Exception:
                        count = 0

            # Convert to RGB and emit raw frame and boxes
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_data.emit({'frame': rgb_image.copy(), 'boxes': boxes_list, 'count': count})

            # small sleep to yield
            time.sleep(0.01)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class VideoWidget(QtWidgets.QWidget):
    """Widget that displays frames, supports pan & zoom and draws boxes scaled to view."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pix = None
        self._boxes = []
        self._scale = 1.0
        self._offset = QtCore.QPointF(0, 0)
        self._dragging = False
        self.setMouseTracking(True)

    def set_frame_and_boxes(self, rgb_frame, boxes):
        # rgb_frame: numpy array HxWx3
        h, w = rgb_frame.shape[:2]
        image = QtGui.QImage(rgb_frame.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        self._pix = QtGui.QPixmap.fromImage(image)
        self._boxes = boxes
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor('#0b0b0b'))
        if self._pix is None:
            return

        # compute scaled pixmap size
        sw = self._pix.width() * self._scale
        sh = self._pix.height() * self._scale
        # center
        cx = (self.width() - sw) / 2 + self._offset.x()
        cy = (self.height() - sh) / 2 + self._offset.y()

        # draw the pixmap using integer rects to match PyQt5 overloads
        tx = int(cx)
        ty = int(cy)
        tw = max(1, int(sw))
        th = max(1, int(sh))
        painter.drawPixmap(tx, ty, tw, th, self._pix)

        # draw boxes scaled
        pen = QtGui.QPen(QtGui.QColor(0, 255, 128), max(2, int(2 * self._scale)))
        painter.setPen(pen)
        for b in self._boxes:
            x1, y1, x2, y2 = b
            rx1 = cx + x1 * self._scale
            ry1 = cy + y1 * self._scale
            rx2 = cx + x2 * self._scale
            ry2 = cy + y2 * self._scale
            painter.drawRoundedRect(QtCore.QRectF(rx1, ry1, rx2 - rx1, ry2 - ry1), 4, 4)

    def wheelEvent(self, event):
        # zoom centered on cursor
        delta = event.angleDelta().y()
        factor = 1.0 + (0.001 * delta)
        self._scale = max(0.1, min(5.0, self._scale * factor))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if getattr(self, '_dragging', False):
            delta = event.pos() - self._last_pos
            self._offset += delta
            self._last_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        self._dragging = False


class DraggableLabel(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.label = QtWidgets.QLabel('Microplastic\n0', self)
        self.label.setAlignment(Qt.AlignCenter)
        font = QtGui.QFont('Helvetica Neue', 14)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setStyleSheet('color: white;')
        self.setStyleSheet('background: rgba(20,20,20,0.85); border-radius: 10px;')
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)
        self._drag_active = False

    def setText(self, text):
        self.label.setText(text)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_active = True
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag_active:
            self.move(event.globalPos() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_active = False


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, weights=DEFAULT_WEIGHTS, camera_index=DEFAULT_CAMERA_INDEX, device=None):
        super().__init__()
        self.setWindowTitle('Microplastic Detector (YOLOv8)')
        self.weights = weights
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Central widget
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Video display
        self.video_widget = VideoWidget()
        self.video_widget.setMinimumSize(640, 480)
        layout.addWidget(self.video_widget)

        # Count display card (hidden; we use overlay)
        count_card = QtWidgets.QFrame()
        count_card.setObjectName('card')
        count_layout = QtWidgets.QVBoxLayout(count_card)
        count_layout.setContentsMargins(20, 10, 20, 10)
        self.count_label = QtWidgets.QLabel('0')
        count_font = QtGui.QFont('Helvetica Neue', 24)
        count_font.setBold(True)
        self.count_label.setFont(count_font)
        self.count_label.setAlignment(Qt.AlignCenter)
        count_layout.addWidget(self.count_label)
        sub_label = QtWidgets.QLabel('Microplastic count')
        sub_font = QtGui.QFont('Helvetica Neue', 10)
        sub_label.setFont(sub_font)
        sub_label.setAlignment(Qt.AlignCenter)
        count_layout.addWidget(sub_label)
        count_card.setFixedHeight(80)
        # layout.addWidget(count_card)

        # Controls: simple Start/Stop buttons only
        ctrl_layout = QtWidgets.QHBoxLayout()
        ctrl_layout.setSpacing(10)
        self.start_btn = QtWidgets.QPushButton('Start')
        self.start_btn.setFixedHeight(36)
        self.start_btn.setCursor(Qt.PointingHandCursor)
        ctrl_layout.addWidget(self.start_btn)
        self.stop_btn = QtWidgets.QPushButton('Stop')
        self.stop_btn.setFixedHeight(36)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        ctrl_layout.addWidget(self.stop_btn)
        layout.addLayout(ctrl_layout)

        # Show count below the controls as well (numeric readout)
        layout.addWidget(count_card)

        # Status bar
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        # Inference thread
        self.thread = None
        self.camera_index = camera_index

        # Connections
        self.start_btn.clicked.connect(self.start_inference)
        self.stop_btn.clicked.connect(self.stop_inference)

        # HiDPI support
        self.device_pixel_ratio = QtWidgets.QApplication.primaryScreen().devicePixelRatio()

        # Apply dark theme stylesheet
        self.apply_styles()

        # Draggable overlay for count
        self.overlay = DraggableLabel(parent=self)
        self.overlay.setFixedSize(180, 70)
        self.overlay.move(30, 30)
        self.overlay.show()

    def closeEvent(self, event):
        # ensure thread stopped on exit
        if self.thread is not None:
            self.thread.stop()
        event.accept()

    def apply_styles(self):
        stylesheet = """
        QMainWindow { background: #0b0b0b; }
        QFrame#card { background: #1f1f1f; border-radius: 10px; border: 1px solid rgba(255,255,255,0.04); }
        QPushButton { background: #2d6cdf; color: white; border-radius: 8px; padding: 6px 12px; font-size: 14px; }
        QPushButton:disabled { background: #3b3b3b; color: #777; }
        QLabel { color: #eaeaea; }
        """
        self.setStyleSheet(stylesheet)

    def start_inference(self):
        if not os.path.exists(self.weights):
            QtWidgets.QMessageBox.critical(self, 'Weights not found', f'Weights file not found:\n{self.weights}')
            return

        # Create and start thread
        self.thread = InferenceThread(self.weights, camera_index=self.camera_index, device=self.device)
        # thresholds are code-only constants
        self.thread.conf = float(CONF_THRESHOLD)
        self.thread.iou = float(IOU_THRESHOLD)
        self.thread.frame_data.connect(self.on_frame_data)
        self.thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.showMessage('Inference started')

    def stop_inference(self):
        if self.thread is not None:
            self.thread.stop()
            self.thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.showMessage('Inference stopped')

    def on_frame_data(self, data):
        # data: {'frame':rgb_np, 'boxes':[[x1,y1,x2,y2],...], 'count':int}
        frame = data['frame']
        boxes = data['boxes']
        count = data['count']
        # update video widget with frame and boxes
        self.video_widget.set_frame_and_boxes(frame, boxes)
        # update overlay count text (overlay is draggable so position independent)
        self.overlay.setText(f'Microplastic\n{count}')
        # also update the count label shown under Start/Stop
        try:
            self.count_label.setText(str(count))
        except Exception:
            pass


def main():
    # Enable High DPI scaling
    QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    # Allow passing weights and camera via CLI args for convenience
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=DEFAULT_WEIGHTS)
    parser.add_argument('--camera', default=DEFAULT_CAMERA_INDEX, type=int)
    parser.add_argument('--device', default=('cuda' if torch.cuda.is_available() else 'cpu'))
    args = parser.parse_args()

    win = MainWindow(weights=args.weights, camera_index=args.camera, device=args.device)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
