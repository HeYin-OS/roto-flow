from typing import List, Tuple

from PySide6.QtCore import QPoint, Slot
from PySide6.QtGui import QPaintEvent, QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget
import numpy as np

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
        self.min_manhattan = 3

        # indicator for current frame index
        self.index_current_frame: int = 0

        # indicator for current insert order number on each frame
        self.index_insert_on_each_frame = np.zeros(self.test_video.frame_num, dtype=int)

        # indicator for drawing order on current frame
        self.index_drawing_order_on_each_frame = np.zeros(self.test_video.frame_num, dtype=int)

        # storage for curves (Div1: frame num<list>, Div2: Drawing order<list>, Div3: Point idx<idx>, Div4: QPointF)
        self.curves_on_each_frame: List[List[List[QPoint]]] = [[] for _ in range(self.test_video.frame_num)]
        self.curve_temp:List[QPoint] = []

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
        painter.setPen(pen) # black

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

        # move temp curve to curves
        print(f"\n[Sketch] Drawing Info of Latest Curve: Total {len(self.curve_temp)} Point(s)")
        self.curves_on_each_frame[self.index_current_frame].append(self.curve_temp)
        self.curve_temp = []

        # debug
        print(f"[Sketch] {len(self.curves_on_each_frame[self.index_current_frame])} curve(s) on Frame {self.index_current_frame}")
        # if len(self.curve_temp) == 0:
        #     print("Temporary curve is empty.")

        # other works
        self.reDraw()
        QWidget.mouseReleaseEvent(self, event)
