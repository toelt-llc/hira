import numpy as np
import struct
import zmq
import time
import cv2

context = zmq.Context()
zpub = context.socket(zmq.PUB)
zpub.bind("tcp://*:9999")

# Wait briefly to allow subscriber to connect
time.sleep(1)

zctx = zmq.Context()
zsub = zctx.socket(zmq.SUB)
zsub.connect("tcp://localhost:9999")
zsub.subscribe(b"")

while True:
    # Generate a dummy grayscale image (e.g., 480x640)
    height, width = 480, 640
    img = np.random.randint(0, 65535, size=(height, width), dtype=np.uint16)
    # Pack header with image shape
    header = struct.pack("HH", height, width)
    # Send multipart message
    zpub.send_multipart([header, img.tobytes()])

    time.sleep(0.1)  # ~10 FPS


    mp = zsub.recv_multipart()
    h, w = struct.unpack("HH", mp[0])
    img = np.frombuffer(mp[1], dtype=np.uint16).reshape(h, w)

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow("image", img)
    cv2.pollKey()

