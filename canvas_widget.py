from typing import List, Tuple

import cv2
from PySide6.QtCore import QPoint, Slot
from PySide6.QtGui import QPaintEvent, QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget
import numpy as np

from edge_snapping import local_snapping, EdgeSnappingConfig
from video import Video


class CanvasWidget(QWidget):
    def __init__(self, parent):
        super(CanvasWidget, self).__init__(parent)
        # set mouse track on
        self.setMouseTracking(True)

        self.test_video = Video("config/test_video_init.yaml")

        # indicator for mouse pressed
        self.is_mouse_pressed: bool = False

        # factors for controlling mouse sampling length
        self.min_manhattan: int = 3

        # indicator for current frame index
        self.index_current_frame: int = 0

        # indicator for current insert order number on each frame
        self.index_insert_on_each_frame = np.zeros(self.test_video.frame_num, dtype=int)

        # indicator for drawing order on current frame
        self.index_drawing_order_on_each_frame = np.zeros(self.test_video.frame_num, dtype=int)

        # storage for curves (Div1: frame idx<list>, Div2: drawing idx<list>, Div3: point idx<list>, Div4: QPoint)
        self.curves_on_each_frame: List[List[List[QPoint]]] = [[] for _ in range(self.test_video.frame_num)]
        self.curve_temp: List[QPoint] = []

        self.weights_history: List[np.ndarray] = []

        # TODO: Other variables for future implementation of bezier curve

    def reDraw(self):
        self.update()

    # ==================================================
    # Event Handle Function
    # ==================================================
    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # draw frame
        painter.drawPixmap(0, 0, self.test_video.qPixmap_format[self.index_current_frame])

        # draw curves
        pen = QPen()
        pen.setColor(QColor(0, 0, 0))
        pen.setWidth(3)
        painter.setPen(pen)  # black

        for curve in self.curves_on_each_frame[self.index_current_frame]:
            painter.drawPolyline(curve)

        if len(self.curve_temp) > 1:
            painter.drawPolyline(self.curve_temp)
        elif len(self.curve_temp) == 1:
            painter.drawPoint(self.curve_temp[0])

        # other works
        QWidget.paintEvent(self, event)

    def mousePressEvent(self, event):
        # set to pressed
        self.is_mouse_pressed = True

        # get point xy
        point = event.position().toPoint()

        # store point xy
        self.curve_temp.append(point)

        # other works
        self.reDraw()
        QWidget.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        if self.is_mouse_pressed:
            current_pos = event.position().toPoint()

            if len(self.curve_temp) == 0 or \
                    (current_pos - self.curve_temp[-1]).manhattanLength() > self.min_manhattan:
                # store point xy
                self.curve_temp.append(current_pos)

                # update
                self.reDraw()

        # other works
        QWidget.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        # set to unpressed
        self.is_mouse_pressed = False
        print(f"\n[Sketch] Drawing Info of Latest Curve: Total {len(self.curve_temp)} Point(s)")

        # if len(self.curve_temp) == 0:
        #     print("Temporary curve is empty.")

        # TODO: edge snapping curve here
        # only current curve needs to be done local snapping
        current_stroke_np_yx = np.array([[p.y(), p.x()] for p in self.curve_temp], dtype=np.float32)
        candidates_yx = self.test_video.candidate_kd_trees.query_batch(
            frame_idx=self.index_current_frame,
            centers_yx=current_stroke_np_yx,
            radius=EdgeSnappingConfig.r_s
        )

        # get snapped stroke in np xy form
        local_snapped_stroke_np_xy = local_snapping(stroke_np_yx=current_stroke_np_yx,
                       image_tensor_rgb=self.test_video.tensor_format[self.index_current_frame],
                       candidate_points_yx=candidates_yx)

        # convert it to List [QPoint]
        # print(local_snapped_stroke_np_xy.shape)
        fitted_stroke_xy = [QPoint(int(xy[0]), int(xy[1])) for xy in local_snapped_stroke_np_xy]

        # test_image = np.zeros_like(cv2.imread("videos/soapbox/00000.jpg", flags=cv2.IMREAD_GRAYSCALE), dtype=np.uint8)
        # for candidate_group in stroke_point_idx_to_candidates:
        #     for (y, x) in candidate_group:
        #         test_image[int(y), int(x)] = 255
        # cv2.imshow("test_image", test_image)

        # move temp curve to curves
        self.curves_on_each_frame[self.index_current_frame].append(fitted_stroke_xy)
        self.curve_temp = []
        # debug
        print(f"[Sketch] {len(self.curves_on_each_frame[self.index_current_frame])} curve(s) on Frame {self.index_current_frame}")

        # other works
        self.reDraw()
        QWidget.mouseReleaseEvent(self, event)
