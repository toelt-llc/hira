import numpy as np
try:
    from PySide2 import QtCore, QtWidgets, QtNetwork
except ImportError:
    from PySide6 import QtCore, QtWidgets, QtNetwork
import struct
import time
import warnings
import pims
import zmq
import os

from .smartcamLandmarker import SmartCamLandmarker
from .smartcamFramer import *


DOUBLE_CAMERA_DTYPES = [frameADM_dtype, frameVTB_dtype, frameA4M_dtype, 
                        frameBosonRotate_dtype, frameL2Rotate_dtype, frameL3Rotate_dtype]
SINGLE_CAMERA_DTYPES = [frameA6x5_dtype, frameA400_dtype, frameBoson_dtype,
                        frameLepton2_dtype, frameLepton3_dtype, frameMerc_dtype, 
                        frameCrop_dtype, frameDal16_dtype]
VISIBLE_CAMERA_DTYPES = [frameMerc_dtype, frameCrop_dtype]
VISIBLE_MONO_CAMERA_DTYPES = [frameMerc_dtype, frameCrop_dtype]
THERMAL_CAMERA_DTYPES = [frameA6x5_dtype, frameA400_dtype, frameDal16_dtype, frameBoson_dtype,
                         frameLepton2_dtype, frameLepton3_dtype]

class SmartCamDecoder():
    def __init__(self, zctx=None, pubAddr=None, tirTrkAddr=None, rgbTrkAddr=None):
        self.thermal_header = None
        self.thermal_time = None
        self.thermal_serial = None
        self.thermal_fn = None
        self.thermal_image = None
        self.thermal_pts = None
        self.thermal_vis = None
        self.thermal_par = None # thermal pims av video reader

        self.visible_header = None
        self.visible_time = None
        self.visible_serial = None
        self.visible_fn = None
        self.visible_image = None
        self.visible_pts = None
        self.visible_vis = None
        self.visible_par = None # visible pims av video reader

        self.distance = 50
        self.calib = None
        self.score = -1
        self.landmarker = None
        self.frame_dtype = None
        self.frame = None
        self.zframe = None
        self.frame_size = 0
        self.num_frames = 0

        self.landmarks = None
        self.video_name = None
        self.video_file = None
        self.zctx = zctx
        self.thermal_address = tirTrkAddr
        self.thermal_pub = None
        self.visible_address = rgbTrkAddr
        self.visible_pub = None
        if self.thermal_address is not None:
            assert zctx is not None
            self.thermal_pub = self.zctx.socket(zmq.PUB)
            self.thermal_pub.bind(self.thermal_address)
        if self.visible_address is not None:
            assert zctx is not None
            self.visible_pub = self.zctx.socket(zmq.PUB)
            self.visible_pub.bind(self.visible_address)


    def close(self):
        if self.thermal_pub is not None:
            url = self.thermal_pub.LAST_ENDPOINT
            self.thermal_pub.unbind(url)
        if self.visible_pub is not None:
            url = self.visible_pub.LAST_ENDPOINT
            self.visible_pub.unbind(url)


    def get_serials(self):
        return self.visible_serial, self.thermal_serial


    def get_images(self):
        img = self.visible_image
        if img is not None:
            if len(img.shape)==2:
                if type(img)==np.ndarray and img.dtype=="uint8":
                    img = np.stack((img, img, img), axis=-1)
            elif type(img) != np.ndarray:
                imglist = np.dsplit(img, 3)
                img = np.dstack((imglist[0], imglist[1], imglist[2]))
        return img, self.visible_time, self.thermal_image, self.thermal_time


    def get_points(self):
        visible_tuple = (self.visible_pts, self.visible_vis)
        thermal_tuple = (self.thermal_pts, self.thermal_vis)
        points_tuple  = (self.score, self.landmarks)
        return visible_tuple + thermal_tuple + points_tuple


    def get_calib(self):
        return self.calib


    def on_file(self, name):
        num_frames = 0
        if self.video_file is None:
            if name.endswith(".smc"):
                self.video_name = name
                self.video_file = open(name, "rb")
            else:
                if not name.endswith(".zsc"):
                    name = name + ".zsc"
                self.video_name = name
                self.video_file = open(name, "rb")
                basename = name.rsplit('.', 1)[0]
                while basename.endswith("-trimmed"):
                    basename = basename.rsplit('-', 1)[0]
                if os.path.exists(basename+"_rgb.mkv"):
                    self.visible_par = pims.PyAVReaderIndexed(basename+"_rgb.mkv")
                if os.path.exists(basename+"_tir.mkv"):
                    self.thermal_par = pims.PyAVReaderIndexed(basename+"_tir.mkv")
                if self.thermal_par is not None and self.visible_par is not None:
                    if len(self.thermal_par) != len(self.visible_par):
                        num_frames = min(len(self.thermal_par), len(self.visible_par))
                        self.thermal_par = self.thermal_par[:num_frames]
                        self.visible_par = self.visible_par[:num_frames]
                    assert len(self.thermal_par) == len(self.visible_par)
            self.num_frames = num_frames
        else:
            self.video_name = None
            self.video_file.close()
            self.video_file = None
            self.visible_par = None
            self.thermal_par = None
            self.frame_size = 0
            self.num_frames = 0
        return self.get_info(self.video_file, par_frames=num_frames)


    def read_frame(self, index):
        if self.visible_par is not None:
            self.visible_image = self.visible_par[index]
        if self.thermal_par is not None:
            self.thermal_image = self.rgb_to_gray(self.thermal_par[index])
        pos = index*self.frame_size
        self.video_file.seek(pos, os.SEEK_SET)
        b = self.video_file.read(self.frame_size)
        self.decode_frame(b)
        if len(b) == self.frame_dtype.itemsize:
            return b, self.thermal_time
        return self.frame.tobytes(), self.thermal_time


    def rgb_to_gray(self, img_rgb):
        # pixel type is uint16 -> two channels
        img_msb = img_rgb[:, :, 1].astype(np.uint16) << 8
        img_lsb = img_rgb[:, :, 0].astype(np.uint16)
        return img_msb+img_lsb


    ##########################################################################  READER part


    def get_info(self, fid, par_frames=0, log=False):
        if fid is None:
            return 0, 0, 0

        # TODO: nel primo pezzo di codice che segue si cerca di capire quale/i sia/siano
        # la/le camera/e incluse nel file. Alcune camere (termiche) infatti possono essere
        # registrate sia da sole che come master di una coppia, per cui occorre fare dei
        # check annidati (capita qual e' la prima camera, verificare se ce n'e' una diversa
        # al seguito). In generale, se sono da sole per i punti si assume che si usi dlib,
        # mentre viene utilizzato mediapipe per i casi in cui c'e' anche la visible.
        # A cio', ora si aggiunge la complicazione che l'immagine non e' presente quando
        # si usa il formato compresso. Bisogna verificare che tutti i check eseguiti siano
        # corretti (di sicuro non lo sono)...

        # determine the smatcam type
        fid.seek(0, os.SEEK_SET)
        b = fid.read(HEADER_MAX_SIZE)
        if b[:len(CROP_HEADER)] == CROP_HEADER:
            # check whether we have a boson after the venus
            if self.video_name.endswith(".smc"):
                frame_size = frameCrop_dtype.itemsize
            else:
                frame_size = frameMSC_dtype.itemsize
            fid.seek(0, os.SEEK_SET)
            b = fid.read(frame_size)
            b = fid.read(HEADER_MAX_SIZE)
            if b[:len(CROP_HEADER)] == CROP_HEADER:
                self.frame_dtype = frameCrop_dtype
            else:
                self.frame_dtype = frameVTB_dtype
        elif b[:len(A6x5_HEADER)] == A6x5_HEADER:
            # check whether we have a mercury after the a6x5
            if self.video_name.endswith(".smc"):
                frame_size = frameADM_dtype.itemsize
            else:
                frame_size = frameDSC_dtype.itemsize
            fid.seek(0, os.SEEK_SET)
            b = fid.read(frame_size)
            b = fid.read(HEADER_MAX_SIZE)
            if b[:len(A6x5_HEADER)] == A6x5_HEADER:
                self.frame_dtype = frameA6x5_dtype
            else: 
                self.frame_dtype = frameADM_dtype
        elif b[:len(A400_HEADER)] == A400_HEADER:
            # check whether we have a mercury after the a400
            if self.video_name.endswith(".smc"):
                frame_size = frameA400_dtype.itemsize
            else:
                frame_size = frameDSC_dtype.itemsize
            fid.seek(0, os.SEEK_SET)
            b = fid.read(frame_size)
            b = fid.read(HEADER_MAX_SIZE)
            if b[:len(A400_HEADER)] == A400_HEADER:
                self.frame_dtype = frameA400_dtype
            else: 
                self.frame_dtype = frameA4M_dtype
        elif b[:len(DAL16_HEADER)] == DAL16_HEADER:
            self.frame_dtype = frameDal16_dtype
        elif b[:len(BOSON_HEADER)] == BOSON_HEADER:
            if self.video_name.endswith(".smc"):
                frame_size = frameBoson_dtype.itemsize
            else:
                frame_size = frameDSC_dtype.itemsize
            fid.seek(0, os.SEEK_SET)
            b = fid.read(frame_size)
            b = fid.read(HEADER_MAX_SIZE)
            if b[:len(BOSON_HEADER)] == BOSON_HEADER:
                self.frame_dtype = frameBoson_dtype
            else:
                self.frame_dtype = frameBosonRotate_dtype
        elif b[:len(LEPTON2_HEADER)] == LEPTON2_HEADER:
            if self.video_name.endswith(".smc"):
                frame_size = frameLepton2_dtype.itemsize
            else:
                frame_size = frameDSC_dtype.itemsize
            fid.seek(0, os.SEEK_SET)
            b = fid.read(frame_size)
            b = fid.read(HEADER_MAX_SIZE)
            if b[:len(LEPTON2_HEADER)] == LEPTON2_HEADER:
                self.frame_dtype = frameLepton2_dtype
            else:
                self.frame_dtype = frameL2Rotate_dtype
        elif b[:len(LEPTON3_HEADER)] == LEPTON3_HEADER:
            if self.video_name.endswith(".smc"):
                frame_size = frameLepton3_dtype.itemsize
            else:
                frame_size = frameDSC_dtype.itemsize
            fid.seek(0, os.SEEK_SET)
            b = fid.read(frame_size)
            b = fid.read(HEADER_MAX_SIZE)
            if b[:len(LEPTON3_HEADER)] == LEPTON3_HEADER:
                self.frame_dtype = frameLepton3_dtype
            else:
                self.frame_dtype = frameL3Rotate_dtype
        elif b[:len(MERC_HEADER)] == MERC_HEADER:
            self.frame_dtype = frameMerc_dtype
        else:
            return 0, 0, 0

        # get the calibration parameters
        if self.video_name.endswith(".smc"):
            frame_dtype = self.frame_dtype
            self.frame_size = self.frame_dtype.itemsize
        else:
            if self.frame_dtype in DOUBLE_CAMERA_DTYPES:
                self.frame_size = frameZSC_dtype.itemsize
                frame_dtype = frameZSC_dtype
            elif self.frame_dtype in VISIBLE_CAMERA_DTYPES:
                self.frame_size = frameMSC_dtype.itemsize
                frame_dtype = frameMSC_dtype
            else:
                self.frame_size = frameDSC_dtype.itemsize
                frame_dtype = frameDSC_dtype

        fid.seek(0, os.SEEK_SET)
        b = fid.read(self.frame_size)
        frame = np.ndarray(1, dtype=frame_dtype, buffer=b)
        dtype = self.frame_dtype
        if dtype in DOUBLE_CAMERA_DTYPES:
            if frame_dtype == frameZSC_dtype:
                self.calib = frame.item()[11]
            else:
                self.calib = frame.item()[13]
            self.landmarker = SmartCamLandmarker(self.calib)
        elif dtype in SINGLE_CAMERA_DTYPES:
            pass
        else:
            raise NotImplementedError

        # make sure we are getting the frames correctly
        b = fid.read(self.frame_size)
        if self.frame_dtype==frameADM_dtype and b[:len(A6x5_HEADER)]!=A6x5_HEADER:
                print("Frame size mismatch!")
                raise ValueError
        elif self.frame_dtype==frameVTB_dtype and b[:len(CROP_HEADER)]!=CROP_HEADER:
            print("Frame size mismatch!")
            raise ValueError
        elif self.frame_dtype==frameA4M_dtype and b[:len(A400_HEADER)]!=A400_HEADER:
            print("Frame size mismatch!")
            raise ValueError
        elif self.frame_dtype==frameA6x5_dtype and b[:len(A6x5_HEADER)]!=A6x5_HEADER:
            print("Frame size mismatch!")
            raise ValueError
        elif self.frame_dtype==frameA400_dtype and b[:len(A400_HEADER)]!=A400_HEADER:
            print("Frame size mismatch!")
            raise ValueError
        elif self.frame_dtype==frameBoson_dtype and b[:len(BOSON_HEADER)]!=BOSON_HEADER:
            print("Frame size mismatch!")
            raise ValueError
        elif self.frame_dtype==frameBosonRotate_dtype and b[:len(BOSON_HEADER)]!=BOSON_HEADER:
            print("Frame size mismatch!")
            raise ValueError
        elif self.frame_dtype==frameLepton2_dtype and b[:len(LEPTON2_HEADER)]!=LEPTON2_HEADER:
            print("Frame size mismatch!")
            raise ValueError
        elif self.frame_dtype==frameL2Rotate_dtype and b[:len(LEPTON2_HEADER)]!=LEPTON2_HEADER:
            print("Frame size mismatch!")
            raise ValueError
        elif self.frame_dtype==frameLepton3_dtype and b[:len(LEPTON3_HEADER)]!=LEPTON3_HEADER:
            print("Frame size mismatch!")
            raise ValueError
        elif self.frame_dtype==frameL3Rotate_dtype and b[:len(LEPTON3_HEADER)]!=LEPTON3_HEADER:
            print("Frame size mismatch!")
            raise ValueError
        elif self.frame_dtype==frameMerc_dtype and b[:len(MERC_HEADER)]!=MERC_HEADER:
            print("Frame size mismatch!")
            raise ValueError
        #else:
        #    print("Unknown frame type!")
        #    raise NotImplementedError

        if log:
            print("Counting frames...", end=' ')
        text = ""
        if par_frames != 0:
            text = "meta-"
        fid.seek(0, os.SEEK_END)
        total_size = fid.tell()
        num_frames = total_size // self.frame_size
        if total_size % self.frame_size != 0:
            text += "file NOT correctly closed"
            # last frames may be corrupted...
            last_frame = num_frames-30 if num_frames >= 50 else num_frames // 2
            num_frames = last_frame + 1
        else:
            text += "file correctly closed"
            last_frame = num_frames-1
        if par_frames != 0:
            if par_frames < num_frames:
                text += ", img-files do NOT appear to be ok"
            else:
                text += ", img-files appear to be ok"
            num_frames = min(num_frames, par_frames)
            last_frame = num_frames-1
        self.num_frames = num_frames
        if log:
            print(f"...{num_frames:d} found  ({text})")

        if log:
            print("Computing frame rate...", end=' ')
        fid.seek(0, os.SEEK_SET)
        b = fid.read(self.frame_size)
        frame = np.ndarray(1, dtype=frame_dtype, buffer=b)
        ti = frame.item()[1]

        fid.seek(last_frame*self.frame_size, os.SEEK_SET)
        b = fid.read(self.frame_size)
        frame = np.ndarray(1, dtype=frame_dtype, buffer=b)
        tf = frame.item()[1]

        total_time = (tf-ti)/1000000
        video_rate = last_frame/total_time
        if log:
            print(f"...video was recorded with average fps = {video_rate:.2f}")
        fid.seek(0, os.SEEK_SET)

        return num_frames, total_time, video_rate


    def decode_frame(self, b):
        frame_type = self.frame_dtype
        if frame_type in VISIBLE_CAMERA_DTYPES:
            if len(b)<frameMSC_dtype.itemsize:
                # the compressed format is the smallest one, so this is not a valid frame
                return
            if len(b)!=frameMSC_dtype.itemsize and len(b)!=self.frame_dtype.itemsize:
                # not the compressed frame, but not even an uncompressed one
                return
        elif frame_type in SINGLE_CAMERA_DTYPES:
            if len(b)<frameDSC_dtype.itemsize:
                # the compressed format is the smallest one, so this is not a valid frame
                return
            if len(b)!=frameDSC_dtype.itemsize and len(b)!=self.frame_dtype.itemsize:
                # not the compressed frame, but not even an uncompressed one
                return
        elif frame_type in DOUBLE_CAMERA_DTYPES:
            if len(b)<frameZSC_dtype.itemsize:
                # the compressed format is the smallest one, so this is not a valid frame
                return
            if len(b)!=frameZSC_dtype.itemsize and len(b)!=self.frame_dtype.itemsize:
                # not the compressed frame, but not even an uncompressed one
                return
        else:
            assert False
        if frame_type in DOUBLE_CAMERA_DTYPES:
            self.decode_frame_combined(b)
        elif frame_type in SINGLE_CAMERA_DTYPES:
            self.decode_frame_single(b)
        else:
            assert False


    def decode_frame_combined(self, b):
        frame_size = len(b)
        if frame_size == frameZSC_dtype.itemsize:
            zframe = np.ndarray(1, dtype=frameZSC_dtype, buffer=b)
            hdr1, t1, ser1, n1, hdr2, t2, ser2, n2, ghdr, gt, geom, calib, score = zframe.item()
            self.zframe = zframe
        else:
            frame = np.ndarray(1, dtype=self.frame_dtype, buffer=b)
            hdr1, t1, ser1, n1, img1, hdr2, t2, ser2, n2, img2, ghdr, gt, geom, calib, score = frame.item()
            zframe = (hdr1, t1, ser1, n1, hdr2, t2, ser2, n2, ghdr, gt, geom, calib, score)
            self.zframe = np.array([zframe], dtype=frameZSC_dtype)
            self.frame = frame
        geom = np.frombuffer(geom, dtype="3f4")
        if self.frame_dtype == frameADM_dtype:
            if frame_size == self.frame_dtype.itemsize:
                self.visible_image = img2
                self.thermal_image = img1
            else:
                # frames in the video are rgb, but true images are greyscale
                self.visible_image = self.visible_image[:, :, 0]
                img1 = self.thermal_image
                img2 = self.visible_image
                full_frame = (hdr1, t1, ser1, n1, img1, hdr2, t2, ser2, n2, img2, ghdr, gt, geom, calib, score)
                self.frame = np.array([full_frame], dtype=self.frame_dtype)
            self.visible_time = t2
            self.thermal_time = t1
            self.visible_serial = ser2
            self.thermal_serial = ser1
        elif self.frame_dtype == frameVTB_dtype:
            if frame_size == self.frame_dtype.itemsize:
                self.visible_image = img1
                self.thermal_image = img2
            else:
                # frames in the video are rgb, but true images are greyscale
                self.visible_image = self.visible_image[:, :, 0]
                img1 = self.visible_image
                img2 = self.thermal_image
                full_frame = (hdr1, t1, ser1, n1, img1, hdr2, t2, ser2, n2, img2, ghdr, gt, geom, calib, score)
                self.frame = np.array([full_frame], dtype=self.frame_dtype)
            self.visible_time = t1
            self.thermal_time = t2
            self.visible_serial = ser1
            self.thermal_serial = ser2
        elif self.frame_dtype == frameA4M_dtype:
            if frame_size == self.frame_dtype.itemsize:
                self.visible_image = img2
                self.thermal_image = img1
            else:
                # frames in the video are rgb, but true images are greyscale
                self.visible_image = self.visible_image[:, :, 0]
                img1 = self.thermal_image
                img2 = self.visible_image
                full_frame = (hdr1, t1, ser1, n1, img1, hdr2, t2, ser2, n2, img2, ghdr, gt, geom, calib, score)
                self.frame = np.array([full_frame], dtype=self.frame_dtype)
            self.visible_time = t2
            self.thermal_time = t1
            self.visible_serial = ser2
            self.thermal_serial = ser1
        elif self.frame_dtype == frameBosonRotate_dtype:
            if frame_size == self.frame_dtype.itemsize:
                self.visible_image = img2
                self.thermal_image = img1
            else:
                img1 = self.thermal_image
                img2 = self.visible_image[:, :, 0]
                full_frame = (hdr1, t1, ser1, n1, img1, hdr2, t2, ser2, n2, img2, ghdr, gt, geom, calib, score)
                self.frame = np.array([full_frame], dtype=self.frame_dtype)
            self.visible_time = t2
            self.thermal_time = t1
            self.visible_serial = ser2
            self.thermal_serial = ser1
        elif self.frame_dtype == frameL2Rotate_dtype:
            if frame_size == self.frame_dtype.itemsize:
                self.visible_image = img2
                self.thermal_image = img1
            else:
                img1 = self.thermal_image
                img2 = self.visible_image[:, :, 0]
                full_frame = (hdr1, t1, ser1, n1, img1, hdr2, t2, ser2, n2, img2, ghdr, gt, geom, calib, score)
                self.frame = np.array([full_frame], dtype=self.frame_dtype)
            self.visible_time = t2
            self.thermal_time = t1
            self.visible_serial = ser2
            self.thermal_serial = ser1
        elif self.frame_dtype == frameL3Rotate_dtype:
            if frame_size == self.frame_dtype.itemsize:
                self.visible_image = img2
                self.thermal_image = img1
            else:
                img1 = self.thermal_image
                img2 = self.visible_image[:, :, 0]
                full_frame = (hdr1, t1, ser1, n1, img1, hdr2, t2, ser2, n2, img2, ghdr, gt, geom, calib, score)
                self.frame = np.array([full_frame], dtype=self.frame_dtype)
            self.visible_time = t2
            self.thermal_time = t1
            self.visible_serial = ser2
            self.thermal_serial = ser1
        else:
            raise NotImplementedError

        self.score = score
        if score>0 and self.landmarker is not None:
            vPts, vVis, tPts, tVis = self.landmarker.extract_points(geom)
            self.visible_pts = vPts
            self.visible_vis = vVis
            self.thermal_pts = tPts
            self.thermal_vis = tVis
            self.landmarks = geom
        else:
            self.visible_pts = None
            self.visible_vis = None
            self.thermal_pts = None
            self.thermal_vis = None
            self.landmarks = None


    def decode_frame_single(self, b):
        frame_size = len(b)
        if frame_size == frameMSC_dtype.itemsize:
            zframe = np.ndarray(1, dtype=frameMSC_dtype, buffer=b)
            hdr, t, ser, n, ghdr, gt, geom, calib, score = zframe.item()
            self.zframe = zframe
            img = self.visible_image
            if self.frame_dtype in VISIBLE_MONO_CAMERA_DTYPES:
                img = img[:, :, 0]
            full_frame =  (hdr, t, ser, n, img, ghdr, gt, geom, calib, score)
            self.frame = np.array([full_frame], dtype=self.frame_dtype)
        elif frame_size == frameDSC_dtype.itemsize:
            zframe = np.ndarray(1, dtype=frameDSC_dtype, buffer=b)
            hdr, t, ser, n, phdr, pt, pts, calib, score = zframe.item()
            self.zframe = zframe
            img = self.thermal_image
            full_frame =  (hdr, t, ser, n, img, phdr, pt, pts, calib, score)
            self.frame = np.array([full_frame], dtype=self.frame_dtype)
        else:
            frame = np.ndarray(1, dtype=self.frame_dtype, buffer=b)
            if self.frame_dtype in VISIBLE_CAMERA_DTYPES:
                hdr, t, ser, n, img, ghdr, gt, geom, calib, score = frame.item()
                zframe = (hdr, t, ser, n, ghdr, gt, geom, calib, score)
                self.zframe = np.array([zframe], dtype=frameMSC_dtype)
            else:
                hdr, t, ser, n, img, phdr, pt, pts, calib, score = frame.item()
                zframe = (hdr, t, ser, n, phdr, pt, pts, calib, score)
                self.zframe = np.array([zframe], dtype=frameDSC_dtype)
            self.frame = frame

        self.score = score
        if self.frame_dtype in VISIBLE_CAMERA_DTYPES:
            if self.landmarker is None:
                self.landmarker = SmartCamLandmarker(calib.copy())
            self.thermal_serial = None
            self.thermal_image = None
            self.thermal_time = None
            self.thermal_pts = None
            self.thermal_vis = None
            self.visible_serial = ser
            self.visible_image = img
            self.visible_time = t
            if score > 0:
                points, visibility = self.landmarker.extract_rgb_points(geom)
                self.visible_pts = points
                self.visible_vis = visibility
                self.landmarks = geom
            else:
                self.visible_pts = None
                self.visible_vis = None
                self.landmarks = None
        else:
            self.landmarks = None
            self.visible_serial = None
            self.visible_image = None
            self.visible_time = None
            self.visible_pts = None
            self.visible_vis = None
            self.thermal_serial = ser
            self.thermal_image = img
            self.thermal_time = t
            if score > 0:
                self.thermal_pts = pts
                vis = [pts[k, 0]!=0 or pts[k, 1]!=0 for k in range(pts.shape[0])]
                self.thermal_vis = np.array(vis, dtype=bool)
            else:
                self.thermal_pts = None
                self.thermal_vis = np.zeros(68, dtype=bool)[:pts.shape[0]]







