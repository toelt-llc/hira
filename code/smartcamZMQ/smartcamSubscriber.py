import numpy as np
import struct
import zmq
import cv2

zctx = zmq.Context()
zsub = zctx.socket(zmq.SUB)
zsub.connect("tcp://localhost:9999")
zsub.subscribe(b"")

while True:
    mp = zsub.recv_multipart()
    h, w = struct.unpack("HH", mp[0])
    img = np.frombuffer(mp[1], dtype=np.uint16).reshape(h, w)

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow("image", img)
    cv2.pollKey()



