import numpy as np
try:
    from PySide2 import QtCore, QtWidgets, QtGui
except ImportError:
    from PySide6 import QtCore, QtWidgets, QtGui
import time
import sys
import os

from smartcamDecoder import SmartCamDecoder
from smartcamImager import SmartCamImagingWidget

import pyqtgraph as pg
pg.setConfigOptions(imageAxisOrder="row-major")
pg.setConfigOptions(antialias=True)

DEFAULT_PATH = os.path.join("..", "recs")

class SmartCamPlayingWidget(QtWidgets.QWidget):

    fileOpening = QtCore.Signal(bool, int, int, float)
    newFrameset = QtCore.Signal(object)
    newRotation = QtCore.Signal()
    showPoints = QtCore.Signal(bool)
    isPlaying = QtCore.Signal(bool)

    def __init__(self, do_gui=False, title=None, parent=None):
        super().__init__(parent)

        if do_gui:
            self.setup_gui(title)

        self.decoder = None
        self.frame_size = 0
        self.frame_index = 0
        self.num_frames = 0
        self.total_time = 0
        self.video_rate = 0
        self.video_speed = 1
        self.playing = False
        self.videoname = None
        self.calib = None
        self.tt0 = 0
        self.path = DEFAULT_PATH


    def setup_gui(self, title):

        self.mainLayout = QtWidgets.QHBoxLayout()
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.mainLayout)
        if title is None:
            self.commandPanel = QtWidgets.QWidget()
        else:
            self.commandPanel = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout()
        self.commandPanel.setLayout(layout)

        self.btnOpen = QtWidgets.QPushButton('Open')
        self.btnOpen.setEnabled(True)
        self.btnOpen.clicked.connect(self.on_open)
        layout.addWidget(self.btnOpen)
        self.checkPoints = QtWidgets.QCheckBox()
        self.checkPoints.setText("Show Points")
        self.checkPoints.setEnabled(True)
        self.checkPoints.setChecked(True)
        self.checkPoints.stateChanged.connect(self.check_points_changed)
        layout.addWidget(self.checkPoints)
        layout.addStretch()

        self.totalFrameLabel = QtWidgets.QLabel("Total Frames: 0")
        layout.addWidget(self.totalFrameLabel)
        self.totalTimeLabel = QtWidgets.QLabel("Total Time: 0:00:00")
        layout.addWidget(self.totalTimeLabel)
        self.videoRateLabel = QtWidgets.QLabel("Video Rate: 0")
        layout.addWidget(self.videoRateLabel)
        layout.addStretch()

        self.btnRotate = QtWidgets.QPushButton('Rotate')
        self.btnRotate.setEnabled(False)
        self.btnRotate.clicked.connect(self.on_rotate)
        layout.addWidget(self.btnRotate)
        self.btnPlay = QtWidgets.QPushButton('Play')
        self.btnPlay.setEnabled(False)
        self.btnPlay.clicked.connect(self.on_play)
        layout.addWidget(self.btnPlay)
        widget = QtWidgets.QWidget()
        widget_layout = QtWidgets.QHBoxLayout()
        widget_layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(widget_layout)
        speedLabel = QtWidgets.QLabel("Video Speed: ")
        widget_layout.addWidget(speedLabel, alignment=QtCore.Qt.AlignLeft)
        self.speedBox = QtWidgets.QComboBox()
        self.speedBox.addItem("• 8")
        self.speedBox.addItem("• 4")
        self.speedBox.addItem("• 2")
        self.speedBox.addItem("  1")
        self.speedBox.addItem("/ 2")
        self.speedBox.addItem("/ 4")
        self.speedBox.addItem("/ 8")
        self.speedBox.setCurrentIndex(3)
        self.speedBox.setEnabled(False)
        self.speedBox.currentIndexChanged.connect(self.speed_changed)
        widget_layout.addWidget(self.speedBox, alignment=QtCore.Qt.AlignRight)
        layout.addWidget(widget)
        
        self.btnFirst = QtWidgets.QPushButton('First Frame')
        self.btnFirst.setEnabled(False)
        self.btnFirst.clicked.connect(self.on_first)
        layout.addWidget(self.btnFirst)
        self.btnPrev = QtWidgets.QPushButton('Previous Frame')
        self.btnPrev.setEnabled(False)
        self.btnPrev.clicked.connect(self.on_prev)
        layout.addWidget(self.btnPrev)
        self.btnNext = QtWidgets.QPushButton('Next Frame')
        self.btnNext.setEnabled(False)
        self.btnNext.clicked.connect(self.on_next)
        layout.addWidget(self.btnNext)
        self.btnLast = QtWidgets.QPushButton('Last Frame')
        self.btnLast.setEnabled(False)
        self.btnLast.clicked.connect(self.on_last)
        layout.addWidget(self.btnLast)
        self.btnGoto = QtWidgets.QPushButton('Go to...')
        self.btnGoto.setEnabled(False)
        self.btnGoto.clicked.connect(self.on_goto)
        layout.addWidget(self.btnGoto)
        layout.addStretch()

        self.currentFrameLabel = QtWidgets.QLabel("Current Frame: 0")
        layout.addWidget(self.currentFrameLabel)
        currentTimeLabel = QtWidgets.QLabel("Current Time:")
        layout.addWidget(currentTimeLabel)
        self.currentTimeEdit = QtWidgets.QLineEdit()
        self.currentTimeEdit.setText("0:00:00 (0)")
        self.currentTimeEdit.setEnabled(False)
        self.playedTimer = QtCore.QTimer(self)
        self.playedTimer.timeout.connect(self.play_timeout)
        layout.addWidget(self.currentTimeEdit)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.commandPanel.setSizePolicy(sizePolicy)
        self.mainLayout.addWidget(self.commandPanel)


    def do_open(self, fullname):
        if self.videoname is not None:
            if self.playing:
                self.on_play()
            self.decoder.on_file(None)
            self.videoname = None
            self.btnOpen.setText("Open")
            self.decoder = None
            self.update_file_info()
            self.fileOpening.emit(False, 0, 0, 0)
            self.calib = None
            return
        
        print(f"Opening {fullname}...")
        self.path = os.path.dirname(fullname)
        self.decoder = SmartCamDecoder()
        self.num_frames, self.total_time, self.video_rate = self.decoder.on_file(fullname)
        if self.num_frames > 0:
            self.videoname = fullname
            self.fileOpening.emit(True, self.num_frames, self.total_time, self.video_rate)
            self.on_first()
            self.update_file_info()
        else:
            self.videoname = None
            self.decoder.on_file(None)
            self.decoder = None
            print("Not a valid smartcam video...")


    def on_open(self):
        fullname = None
        if self.videoname is None:
            fullname, ok = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select recording...", self.path, "*.zsc *.smc"
            )
            if not ok:
                return
        self.do_open(fullname)
        self.update_buttons()


    def read_frame(self):
        if self.frame_index>=0 and self.frame_index<self.num_frames:
            self.decoder.read_frame(self.frame_index)
            self.frame_index += 1
            return True
        return False


    def update_file_info(self):
        if self.videoname is None:
            self.totalFrameLabel.setText("Total Frames: 0")
            self.totalTimeLabel.setText("Total Time: 0:00:00")
            self.videoRateLabel.setText("Video Rate: 0")
            self.currentFrameLabel.setText("Current Frame: 0")
            self.currentTimeEdit.setText("0:00:00 (0.0)")
            return
        
        text = f"Total Frames: {self.num_frames:d}"
        self.totalFrameLabel.setText(text)
        totalTime = int(self.total_time)
        hours, mins, secs = self.decompose(totalTime)
        text = f"Total Time: {hours:d}:{mins:02d}:{secs:02d}"
        self.totalTimeLabel.setText(text)
        text = f"Video Rate: {self.video_rate:.2f}"
        self.videoRateLabel.setText(text)


    def update_buttons(self):
        if self.videoname is not None:
            self.btnOpen.setText("Close")
        else:
            self.btnOpen.setText("Open")        
        self.btnPlay.setEnabled(self.videoname is not None)
        self.speedBox.setEnabled(self.videoname is not None)
        self.btnFirst.setEnabled(self.videoname is not None)
        self.btnPrev.setEnabled(self.videoname is not None)
        self.btnNext.setEnabled(self.videoname is not None)
        self.btnLast.setEnabled(self.videoname is not None)
        self.btnGoto.setEnabled(self.videoname is not None)


    def speed_changed(self):
        speeds = [8, 4, 2, 1, 0.5, 0.25, 0.125]
        index = self.speedBox.currentIndex()
        self.video_speed = speeds[index]
        if self.playing:
            self.playedTimer.stop()
            self.playedTimer.start(1000/(self.video_rate*self.video_speed))


    def on_rotate(self):
        self.newRotation.emit()


    def on_play(self):
        if self.playing:
            self.playedTimer.stop()
            self.btnPlay.setText("Play")
        else:
            self.btnPlay.setText("Pause")
            self.playedTimer.start(1000/(self.video_rate*self.video_speed))
        self.playing = not self.playing
        self.isPlaying.emit(self.playing)


    def on_first(self):
        self.frame_index = 0
        if self.read_frame():
            self.update_gui()


    def on_prev(self):
        self.frame_index -= 2
        if self.read_frame():
            self.update_gui()


    def play_timeout(self):
        if self.videoname is None:
            return
        if self.read_frame():
            self.update_gui()
        else:
            self.on_play()


    def on_next(self):
        if self.read_frame():
            self.update_gui()


    def on_last(self):
        self.frame_index = self.num_frames-1
        if self.read_frame():
            self.update_gui()


    def on_goto(self):
        frame, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Go to frame ...",
            "Enter the desired frame:",
            value=self.frame_index,
            minValue=1,
            maxValue=self.num_frames
        )
        if not ok:
            return
        self.do_goto(frame)


    def do_goto(self, frame):
        frame -= 1
        if self.playing:
            self.on_play()
        self.frame_index = frame
        if self.read_frame():
            self.update_gui()


    def check_points_changed(self):
        status = self.checkPoints.isChecked()
        self.showPoints.emit(status)


    def update_gui(self):
        vImg, vt, tImg, tt = self.decoder.get_images()
        points_tuple = self.decoder.get_points()
        vPts, vVis, tPts, tVis, score, lands = points_tuple
        if self.frame_index == 1:
            self.tt0 = tt if tImg is not None else vt 

        text = f"Current Frame: {self.frame_index:d}"
        self.currentFrameLabel.setText(text)
        if tImg is not None:
            currentTime = (tt-self.tt0) / 1000000
        else:
            currentTime = (vt-self.tt0) / 1000000
        hours, mins, secs = self.decompose(int(currentTime))
        text = f"{hours:d}:{mins:02d}:{secs:02d} ({currentTime:.1f})"
        self.currentTimeEdit.setText(text)

        num = self.frame_index
        frameInfo = (num, vImg, vPts, vVis, vt, tImg, tPts, tVis, tt, score, lands)
        self.newFrameset.emit(frameInfo)


    def decompose(self, totalSecs):
        secs = totalSecs % 60
        mins = (totalSecs // 60) % 60
        hours = (totalSecs // 60) // 60
        return hours, mins, secs


class SmartCamPlayer(QtWidgets.QMainWindow):
    def __init__(self, argv, parent=None):
        super().__init__(parent)

        self.playingWidget = SmartCamPlayingWidget(True, "Command Panel", self)
        self.imagingWidget = SmartCamImagingWidget("Frame Panel", self)
        self.setup_gui()

        self.playingWidget.fileOpening.connect(self.fileOpening)
        self.playingWidget.newFrameset.connect(self.newFrameset)
        self.playingWidget.newRotation.connect(self.newRotation)
        self.playingWidget.showPoints.connect(self.show_points_changed)
        self.imagingWidget.slider.setVisible(True)
        if len(argv) > 1:
            self.playingWidget.do_open(argv[1])


    def setup_gui(self):
        self.setWindowTitle("Next2U SmartCam Player")
        self.centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.centralWidget)
        self.mainLayout = QtWidgets.QGridLayout()
        self.centralWidget.setLayout(self.mainLayout)
        self.mainLayout.addWidget(self.playingWidget, 0, 0)
        self.mainLayout.addWidget(self.imagingWidget, 0, 1)
        self.mainLayout.setColumnStretch(0, 0)
        self.mainLayout.setColumnStretch(1, 1)


    def show_points_changed(self):
        self.imagingWidget.flip_points()
        self.playingWidget.update_gui()


    def fileOpening(self, isOpened, num_frames, total_time, video_rate):
        if isOpened:
            self.imagingWidget.slider_set_max(self.playingWidget.num_frames, self.playingWidget.total_time)
            self.imagingWidget.sliderChanged.connect(self.on_slider)
            self.playingWidget.isPlaying.connect(self.isPlaying)
        else:
            self.imagingWidget.reset_frame()
            self.playingWidget.isPlaying.disconnect()
            self.imagingWidget.sliderChanged.disconnect()
        self.imagingWidget.enable_slider(isOpened)
        self.playingWidget.btnRotate.setEnabled(isOpened)


    def newRotation(self):
        self.imagingWidget.rotate_left()


    def isPlaying(self, status):
        self.imagingWidget.enable_slider(not status)


    def on_slider(self, val):
        frame = int(val*self.playingWidget.video_rate)+1
        self.playingWidget.do_goto(frame)


    def newFrameset(self, frameInfo):
        num, vImg, vPts, vVis, vt, tImg, tPts, tVis, tt, score, _ = frameInfo
        self.imagingWidget.show_frame(num, vImg, vPts, vVis, tImg, tPts, tVis, score)





if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('logo.png'))

    mw = SmartCamPlayer(sys.argv)
    #mw.setMinimumSize(1500, 550)
    mw.show()

    sys.exit(app.exec_())


