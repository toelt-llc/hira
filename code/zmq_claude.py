import numpy as np
try:
    from PySide2 import QtCore, QtWidgets, QtGui
except ImportError:
    from PySide6 import QtCore, QtWidgets, QtGui
import struct
import zmq
import sys
import os
import threading
import queue
import time
import cv2
import logging

from smartcamDecoder import SmartCamDecoder
from smartcamImager import SmartCamImagingWidget
from inference_gen import ImageUpscaler  # Import your upscaler model
from inference_frame import inference_realesrgan_frame  # Import the inference function

import pyqtgraph as pg
pg.setConfigOptions(imageAxisOrder="row-major")
pg.setConfigOptions(antialias=True)

DEFAULT_PATH = os.path.join("..", "recs")

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

class UpscalerThread(QtCore.QThread):
    """Thread for handling image upscaling without blocking the UI"""
    upscaled_image = QtCore.Signal(object)
    status_update = QtCore.Signal(str)  # Signal for status updates

    def __init__(self, parent=None):
        super().__init__(parent)
        self.queue_in = queue.Queue(maxsize=10)
        self.running = True
        try:
            self.status_update.emit("Initializing upscaler model...")
            self.upscaler = ImageUpscaler(device=None)
            logging.info("Upscaler initialized")
            self.status_update.emit("Upscaler ready")
            self.model_loaded = True
        except Exception as e:
            self.status_update.emit(f"Error initializing upscaler: {e}")
            logging.error(f"Error initializing upscaler: {e}")
            self.model_loaded = False

    def run(self):
        while self.running:
            try:
                # Get frame from queue
                thermal_img = self.queue_in.get(timeout=0.1)
                if thermal_img is not None and self.model_loaded:
                    # Process the frame
                    self.status_update.emit("Processing frame...")
                    result = self.upscaler.upscale_single_frame(thermal_img)
                    # result = inference_realesrgan_frame(thermal_img, outscale=2, model_name='realesr-general-x4v3', denoise_strength=0, fp16=False)
                    print(result.shape)
                    if result is not None:
                        # Make sure result is properly scaled
                        if result.dtype != np.uint8:
                            if result.max() > 1.0:  # Assuming float in range beyond 0-1
                                result = np.clip(result, 0, 255).astype(np.uint8)
                            else:  # Assuming float in range 0-1
                                result = (result * 255).astype(np.uint8)
                        # Emit the result
                        self.upscaled_image.emit(result)
                        self.status_update.emit("Frame processed")
                    else:
                        self.status_update.emit("Upscaling returned None")
            except queue.Empty:
                continue
            except Exception as e:
                self.status_update.emit(f"Error: {str(e)}")
                logging.error(f"Error in upscaler thread: {e}")

    def stop(self):
        self.running = False
        self.wait()

    def process_image(self, img):
        if not self.queue_in.full() and img is not None:
            # Preprocess the thermal image similar to read_binary function
            # Normalize and scale to 0-255 range
            try:
                frame_np = np.array(img)
                if frame_np.max() > frame_np.min():  # Avoid division by zero
                    frame_array_scaled_255 = ((frame_np - frame_np.min()) / (frame_np.max() - frame_np.min()) * 255).astype(np.uint8)
                    self.queue_in.put(frame_array_scaled_255)
                else:
                    logging.warning("Skipping frame with no contrast (min == max)")
            except Exception as e:
                logging.error(f"Error preprocessing thermal image: {e}")


class ThermalImageWidget(QtWidgets.QGroupBox):
    """Widget for displaying the upscaled thermal image"""

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setup_gui()
        self.image = None
        self.upscaled = None
        self.fps_counter = 0
        self.last_time = time.time()
        self.fps = 0
        # Set default colormap - can be changed to other cv2 colormaps like COLORMAP_JET, COLORMAP_HOT, etc.
        self.colormap = cv2.COLORMAP_INFERNO  # Good default for thermal

    def setup_gui(self):
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        # Add upscaler status info
        self.infoLabel = QtWidgets.QLabel("Upscaler: Ready")
        layout.addWidget(self.infoLabel)

        # Add image view
        self.imageView = pg.ImageView()
        layout.addWidget(self.imageView)

        # Add controls for colormap selection
        colormap_layout = QtWidgets.QHBoxLayout()
        colormap_label = QtWidgets.QLabel("Colormap:")
        self.colormapCombo = QtWidgets.QComboBox()
        self.colormapCombo.addItems(["INFERNO", "JET", "HOT", "VIRIDIS", "PLASMA", "TURBO", "PARULA"])
        self.colormapCombo.currentIndexChanged.connect(self.change_colormap)
        colormap_layout.addWidget(colormap_label)
        colormap_layout.addWidget(self.colormapCombo)
        layout.addLayout(colormap_layout)

        # Add histogram stretch controls
        stretch_layout = QtWidgets.QHBoxLayout()
        self.autoLevelsCheck = QtWidgets.QCheckBox("Auto Contrast")
        self.autoLevelsCheck.setChecked(True)
        stretch_layout.addWidget(self.autoLevelsCheck)
        layout.addLayout(stretch_layout)

        # Add FPS counter
        self.fpsLabel = QtWidgets.QLabel("FPS: 0.0")
        layout.addWidget(self.fpsLabel)
        
    def change_colormap(self, index):
        """Change the colormap based on combo box selection"""
        colormap_dict = {
            0: cv2.COLORMAP_INFERNO,
            1: cv2.COLORMAP_JET,
            2: cv2.COLORMAP_HOT,
            3: cv2.COLORMAP_VIRIDIS, 
            4: cv2.COLORMAP_PLASMA,
            5: cv2.COLORMAP_TURBO,
            6: cv2.COLORMAP_PARULA
        }
        self.colormap = colormap_dict.get(index, cv2.COLORMAP_INFERNO)
        # If we already have an image, update it with the new colormap
        if self.upscaled is not None:
            self.show_upscaled(self.upscaled)

    def show_upscaled(self, upscaled_image):
        """Display the upscaled thermal image"""
        self.upscaled = upscaled_image
        
        # Update FPS counter
        self.fps_counter += 1
        if self.fps_counter % 10 == 0:
            now = time.time()
            self.fps = 10 / (now - self.last_time)
            self.last_time = now
            self.fpsLabel.setText(f"FPS: {self.fps:.2f}")
        
        # Display the upscaled image with a thermal colormap
        if self.upscaled is not None:
            # Apply colormap if the image is grayscale
            if len(self.upscaled.shape) == 2:
                # Apply colormap (converts to BGR)
                colored = cv2.applyColorMap(self.upscaled, self.colormap)
                # Convert BGR to RGB for PyQtGraph
                colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
                self.imageView.setImage(colored_rgb, autoLevels=False)
            else:
                # If already has channels, just display it
                self.imageView.setImage(self.upscaled, autoLevels=True)
                
            self.infoLabel.setText(f"Upscaler: Active - Size: {self.upscaled.shape}")


class SmartCamPlayingWidget(QtWidgets.QWidget):

    fileOpening = QtCore.Signal(bool, int, int, float)
    newFrameset = QtCore.Signal(object)
    newRotation = QtCore.Signal()
    showPoints = QtCore.Signal(bool)
    isPlaying = QtCore.Signal(bool)
    newThermalImage = QtCore.Signal(object)  # Signal for thermal image processing

    def __init__(self, do_gui=False, title=None, parent=None):
        super().__init__(parent)

        if do_gui:
            self.setup_gui(title)

        self.decoder = None
        self.zctx = zmq.Context()
        self.zpub = self.zctx.socket(zmq.PUB)
        self.zpub.bind("tcp://0.0.0.0:9999")
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


    def do_close(self):
        if self.videoname is not None:
            self.do_open(None)
        url = self.zpub.LAST_ENDPOINT
        self.zpub.unbind(url)


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
            fullname, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select recording...", self.path, "*.zsc *.smc"
            )
            if not fullname:
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

        # zmq is sending data
        if tImg is not None:
            h, w = tImg.shape[:2]
            b = struct.pack("HH", h, w)
            self.zpub.send(b, zmq.SNDMORE)
            b = tImg.tobytes()
            self.zpub.send(b, 0)
            
            # Emit thermal image for upscaling
            self.newThermalImage.emit(tImg)
        #
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

        # Create widgets
        self.playingWidget = SmartCamPlayingWidget(True, "Command Panel", self)
        self.imagingWidget = SmartCamImagingWidget("Frame Panel", self)
        self.thermalWidget = ThermalImageWidget("Upscaled Thermal", self)
        
        # Initialize upscaler thread
        self.upscalerThread = UpscalerThread(self)
        self.upscalerThread.status_update.connect(self.update_upscaler_status)
        self.upscalerThread.start()
        
        self.setup_gui()

        # Connect signals
        self.playingWidget.fileOpening.connect(self.fileOpening)
        self.playingWidget.newFrameset.connect(self.newFrameset)
        self.playingWidget.newRotation.connect(self.newRotation)
        self.playingWidget.showPoints.connect(self.show_points_changed)
        self.playingWidget.newThermalImage.connect(self.process_thermal)
        self.upscalerThread.upscaled_image.connect(self.thermalWidget.show_upscaled)
        self.imagingWidget.slider.setVisible(True)
        
        # Open file if provided as argument
        if len(argv) > 1:
            self.playingWidget.do_open(argv[1])
            
    def update_upscaler_status(self, status):
        """Update upscaler status display"""
        self.thermalWidget.infoLabel.setText(f"Upscaler: {status}")


    def setup_gui(self):
        self.setWindowTitle("Next2U SmartCam Player with Upscaler")
        self.centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.centralWidget)
        self.mainLayout = QtWidgets.QGridLayout()
        self.centralWidget.setLayout(self.mainLayout)
        
        # Add widgets to layout
        self.mainLayout.addWidget(self.playingWidget, 0, 0)  # Command panel
        self.mainLayout.addWidget(self.imagingWidget, 0, 1)  # Original frames
        self.mainLayout.addWidget(self.thermalWidget, 0, 2)  # Upscaled thermal
        
        # Set column stretch factors
        self.mainLayout.setColumnStretch(0, 0)  # Command panel doesn't stretch
        self.mainLayout.setColumnStretch(1, 1)  # Original frames stretch
        self.mainLayout.setColumnStretch(2, 1)  # Upscaled thermal stretches
        
        # Set reasonable minimum size
        self.setMinimumSize(1600, 600)


    def closeEvent(self, evt):
        self.playingWidget.do_close()
        self.upscalerThread.stop()
        evt.accept()


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


    def process_thermal(self, thermal_img):
        """Send thermal image to upscaler thread for processing"""
        if thermal_img is not None:
            # Display original thermal in a small preview in the upscaler panel
            try:
                frame_np = np.array(thermal_img)
                if frame_np.max() > frame_np.min():  # Avoid division by zero
                    # Save original size for display in status
                    orig_size = frame_np.shape
                    self.thermalWidget.infoLabel.setText(f"Upscaler: Processing... (Original: {orig_size})")
                    # Process the image through the upscaler thread
                    self.upscalerThread.process_image(thermal_img)
            except Exception as e:
                logging.error(f"Error sending thermal image to upscaler: {e}")


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('logo.png'))

    mw = SmartCamPlayer(sys.argv)
    mw.show()

    sys.exit(app.exec_())