import numpy as np

from PySide6 import QtWidgets, QtCore, QtGui
import sys,  os
from glob import glob

import pyqtgraph as pg
import pyqtgraph.dockarea

pg.setConfigOptions(imageAxisOrder='row-major')

frame_dtype = np.dtype(
    [
        ("imgLepton", np.uint16, (60, 80)),
        ("imgBoson", np.uint16, (256, 320)),
    ]
)

def load_frame(vid,index):
    video_file = open(vid, 'rb')
    #print(frame_dtype.itemsize)	
    video_file.seek(index*frame_dtype.itemsize)
    b = video_file.read(frame_dtype.itemsize)
    frame = np.ndarray(1, dtype=frame_dtype, buffer=b)
    lepton_im, boson_im = frame.item()
    return lepton_im, boson_im


# getting the frame size for each of the videos from the other script
#nframes_lb1 = 832
#nframes_lb2 = 1403
#nframes_lb3 = 1337
nframes = {"lb1": 832, "lb2": 1403, "lb3":1337}

#for k in range():
#    low_image, high_image = load_video(k)

