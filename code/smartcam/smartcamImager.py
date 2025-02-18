import numpy as np
try:
    from PySide2 import QtCore, QtWidgets
except ImportError:
    from PySide6 import QtCore, QtWidgets

import pyqtgraph as pg
import pyqtgraph.dockarea
pg.setConfigOptions(imageAxisOrder="row-major")
pg.setConfigOptions(antialias=True)


class TimeSlider(QtWidgets.QSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mousePressEvent(self, evt):
        self.setValue(QtWidgets.QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), evt.x(), self.width()))

    def mouseMoveEvent(self, evt):
        self.setValue(QtWidgets.QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), evt.x(), self.width()))


class SmartCamImagingWidget(QtWidgets.QWidget):

    sliderChanged = QtCore.Signal(int)

    def __init__(self, title=None, num_cams=2, parent=None):
        super().__init__(parent)

        self.num_cameras = num_cams
        self.received = 0
        self.recorded = 0
        self.isRecording = False
        self.showPoints = True
        self.visible_shape = (720, 720, 3)
        self.thermal_shape = (256, 320)
        self.rotation = 0
        self.autolevels = True
        self.mouse_pos = None
        self.tooltip_enable = False
        self.tooltip = QtWidgets.QToolTip()
        self.total_frames = 0
        self.total_time = 0
        self.setup_gui(title, num_cams)


    def setup_gui(self, title, num_cams):
        self.mainLayout = QtWidgets.QHBoxLayout()
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)
        self.setLayout(self.mainLayout)
        if title is None:
            self.framePanel = QtWidgets.QWidget()
            layout = QtWidgets.QGridLayout()
            layout.setContentsMargins(0, 0, 0, 0)
        else:
            self.framePanel = QtWidgets.QGroupBox(title)
            layout = QtWidgets.QGridLayout()
        self.framePanel.setLayout(layout)

        self.receivedLabel = QtWidgets.QLabel("Received Frames: 0")
        layout.addWidget(self.receivedLabel, 0, 0, 1, 1)
        self.recordedLabel = QtWidgets.QLabel("Recorded Frames: 0")
        if num_cams == 1:
            layout.addWidget(self.recordedLabel, 1, 0, 1, 1)
        else:
            layout.addWidget(self.recordedLabel, 0, 1, 1, 1)
        
        self.dockarea = pg.dockarea.DockArea()
        if num_cams == 1:
            layout.addWidget(self.dockarea, 2, 0, 1, 1)
        else:
            layout.addWidget(self.dockarea, 1, 0, 1, 2)

        # visible dock 
        self.visible_widget = pg.GraphicsLayoutWidget()
        self.visible_widget.ci.layout.setSpacing(0)
        self.visible_widget.ci.layout.setContentsMargins(0, 0, 0, 0)
        #self.visible_widget.setBackground(background=None)    # added
        frame_vb = pg.ViewBox()
        self.visible_ii = pg.ImageItem()
        frame_vb.addItem(self.visible_ii)
        frame_vb.setAspectLocked(True)
        frame_vb.invertY(True)
        self.visible_points = pg.ScatterPlotItem(brush='y')
        frame_vb.addItem(self.visible_points, ignoreBounds=True)
        self.visible_widget.addItem(frame_vb)
        self.visible_lut = pg.HistogramLUTItem()
        #self.visible_lut.vb.setMaximumWidth(25)     # added
        #self.visible_lut.vb.setMinimumWidth(15)     # added
        #self.visible_lut.gradient.rectSize = 15     # added
        self.visible_lut.setImageItem(self.visible_ii)
        self.visible_lut.plot.setData([])
        self.visible_widget.addItem(self.visible_lut)
        self.visibleDock = pg.dockarea.Dock("Visible")
        self.visibleDock.addWidget(self.visible_widget)
        self.dockarea.addDock(self.visibleDock, "left")

        # thermal dock 
        self.thermal_widget = pg.GraphicsLayoutWidget()
        self.thermal_widget.ci.layout.setSpacing(0)
        self.thermal_widget.ci.layout.setContentsMargins(0, 0, 0, 0)
        frame_vb = pg.ViewBox()
        self.thermal_ii = pg.ImageItem()
        self.thermal_ii.hoverEvent = self.on_hover
        frame_vb.addItem(self.thermal_ii)
        frame_vb.setAspectLocked(True)
        frame_vb.invertY(True)
        self.thermal_points = pg.ScatterPlotItem(brush='y')
        frame_vb.addItem(self.thermal_points, ignoreBounds=True)
        self.thermal_widget.addItem(frame_vb)
        self.thermal_lut = pg.HistogramLUTItem()
        #self.thermal_lut.vb.setMaximumWidth(25)     # added
        #self.thermal_lut.vb.setMinimumWidth(15)     # added
        #self.thermal_lut.gradient.rectSize = 15     # added
        self.thermal_lut.setImageItem(self.thermal_ii)
        self.thermal_lut.plot.setData([])
        self.thermal_widget.addItem(self.thermal_lut)
        self.thermalDock = pg.dockarea.Dock("Thermal")
        self.thermalDock.addWidget(self.thermal_widget)
        self.dockarea.addDock(self.thermalDock, "left")

        self.mainLayout.addWidget(self.framePanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.setSizePolicy(sizePolicy)
        self.thermal_ii.scene().sigMouseMoved.connect(self.on_mouse)
        #self.visible_ii.scene().sigMouseMoved.connect(self.on_mouse)

        self.slider = TimeSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setSingleStep(1)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.on_slider)
        self.sliderPos = 1
        if num_cams == 1:
            layout.addWidget(self.slider, 3, 0, 1, 1)
        else:
            layout.addWidget(self.slider, 2, 0, 1, 2)

        # hide some elements
        self.receivedLabel.setVisible(False)
        self.recordedLabel.setVisible(False)
        self.slider.setVisible(False)
        layout.setRowStretch(0, 0)
        if num_cams == 1:
            layout.setRowStretch(1, 0)
            layout.setRowStretch(2, 20)
        else:
            layout.setRowStretch(1, 20)        
        if num_cams == 1:
            self.visibleDock.setVisible(False)
        self.reset_frame()


    def on_hover(self, evt):
        self.tooltip_enable = not evt.isExit()


    def on_mouse(self, pos):
        self.tooltip.hideText()
        if not self.tooltip_enable:
            return
        self.mouse_pos = pos
        #if self.visibleImg is None or self.mouse_pos is None:
        #    return
        #cursorPos = self.visible_ii.mapFromScene(pos)
        if self.thermalImg is None or self.mouse_pos is None:
            return
        cursorPos = self.thermal_ii.mapFromScene(pos)
        x = int(cursorPos.x())
        y = int(cursorPos.y())
        if (y<0 or y>=self.thermal_shape[0]):
            return
        if (x<0 or x>=self.thermal_shape[1]):
            return
        val = self.thermalImg[y, x]
        #if (y<0 or y>=self.visible_shape[0]):
        #    return
        #if (x<0 or x>=self.visible_shape[1]):
        #    return
        #val = self.visibleImg[y, x]
        if isinstance(val, float):
            tip = f"({x} {y}) {val:.2f}"
        else:
            tip = f"({x} {y}) {val}"
        pt = QtCore.QPointF.toPoint(pos)
        pt = self.thermal_widget.mapToGlobal(pt)
        #pt = self.visible_widget.mapToGlobal(pt)
        self.tooltip.showText(pt, tip) #, msecShowTime = 100)


    def set_rotation(self, rot):
        while rot < 0:
            rot += 360
        self.rotation = int(rot) % 360
        self.show_visible_image()


    def rotate_left(self):
        self.rotation = (self.rotation+90) % 360
        self.show_visible_image()


    def rotate_right(self):
        self.rotation = (self.rotation+270) % 360
        self.show_visible_image()


    def view_slider(self, enable):
        self.slider.setVisible(enable)


    def enable_slider(self, enable):
        self.slider.setEnabled(enable)


    def slider_set_max(self, total_frames, total_time):
        self.total_frames = total_frames
        self.total_time = total_time
        if total_time > 0:
            self.slider.setMaximum(total_time)
            self.slider.setTickInterval(total_time // 100)


    def on_slider(self):
        val = self.slider.value()
        if val != self.sliderPos:
            self.sliderPos = val
            self.sliderChanged.emit(val)
            #print(f"Slider position: {val}")


    def show_visible_image(self):
        if self.visibleImg is not None:
            if self.rotation == 0:
                img = self.visibleImg
            elif self.rotation == 90:
                img = np.rot90(self.visibleImg)
            elif self.rotation == 270:
                img = np.rot90(self.visibleImg, axes=(1,0))
            elif abs(self.rotation) == 180:
                img = np.rot90(self.visibleImg, k=2)

            self.visibleDock.setVisible(True)
            self.visible_ii.setImage(img, autoLevels=False)
        else:
            self.visibleDock.setVisible(False)

        if self.visibleImg is not None and self.showPoints and self.score>0:
            if self.visiblePts is not None:
                if self.visibleVis is None:
                    pts = self.visiblePts
                else:
                    pts = self.visiblePts[self.visibleVis]

                if self.rotation == 90:
                    t = pts[:, 1].copy()
                    pts[:, 1] = img.shape[0]-pts[:, 0]
                    pts[:, 0] = t[:]
                if self.rotation == 270:
                    t = pts[:, 0].copy()
                    pts[:, 0] = img.shape[1]-pts[:, 1]
                    pts[:, 1] = t[:]
                if abs(self.rotation) == 180:
                    pts[:, 0] = img.shape[1]-pts[:, 0]
                    pts[:, 1] = img.shape[0]-pts[:, 1]

                self.visible_points.setData(pos=pts)
            else:
                self.visible_points.setData([])
        else:
            self.visible_points.setData([])


    def show_thermal_image(self):
        if self.thermalImg is not None:
            self.thermalDock.setVisible(True)
            self.thermal_ii.setImage(self.thermalImg, autoLevels=self.autolevels)
            self.thermal_shape = self.thermalImg.shape
        else:
            self.thermalDock.setVisible(False)
        if self.thermalImg is not None and self.showPoints and self.score>0:
            if self.thermalPts is not None:
                if self.thermalVis is None:
                    pts = self.thermalPts
                else:
                    pts = self.thermalPts[self.thermalVis]
                self.thermal_points.setData(pos=pts)
            else:
                self.thermal_points.setData([])
        else:
            self.thermal_points.setData([])


    def reset_frame(self):
        self.received = 0
        self.recorded = 0
        self.total_frames = 0
        self.total_time = 0
        self.sliderPos = 0
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        img = np.zeros(self.visible_shape).astype(int)
        self.visible_ii.setImage(img)
        img = np.zeros(self.thermal_shape).astype(int)
        self.thermal_ii.setImage(img, autoLevels=False)
        self.thermal_points.setData([])
        self.visible_points.setData([])
        self.visibleImg = None
        self.visiblePts = None
        self.visibleVis = None
        self.thermalImg = None
        self.thermalPts = None
        self.thermalVis = None
        self.score = 0
        self.autolevels = True


    def show_frame(self, num, vImg, vPts, vVis, tImg, tPts, tVis, score):
        self.visibleImg = vImg
        self.visiblePts = vPts
        self.visibleVis = vVis
        self.thermalImg = tImg
        self.thermalPts = tPts
        self.thermalVis = tVis
        self.score = score

        if self.slider.isVisible() and self.total_frames!=0:
            pos = int(self.total_time*num / self.total_frames)
            self.sliderPos = pos
            self.slider.setValue(pos)

        self.received += 1
        if self.receivedLabel.isVisible():
            text = f"Received Frames: {self.received:d}"
            self.receivedLabel.setText(text)
        if self.received == 1:
            self.autolevels = True
        else:
            self.autolevels = False
        if self.isRecording and self.recordedLabel.isVisible():
            self.recorded += 1
            text = f"Recorded Frames: {self.recorded:d}"
            self.recordedLabel.setText(text)

        self.show_visible_image()
        self.show_thermal_image()


    def flip_points(self):
        self.showPoints = not self.showPoints


    def toggleRecording(self):
        self.isRecording = not self.isRecording
        if self.isRecording:
            self.recorded = 0

