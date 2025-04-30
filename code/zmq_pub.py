import numpy as np
import struct
import zmq
import time

# Setup ZeroMQ publisher
context = zmq.Context()
zpub = context.socket(zmq.PUB)
zpub.bind("tcp://*:9999")

# Wait briefly to allow subscriber to connect
time.sleep(1)

while True:
    # Generate a dummy grayscale image (e.g., 480x640)
    height, width = 480, 640
    img = np.random.randint(0, 65535, size=(height, width), dtype=np.uint16)

    # Pack header with image shape
    header = struct.pack("HH", height, width)

    # Send multipart message
    zpub.send_multipart([header, img.tobytes()])

    time.sleep(0.1)  # ~10 FPS
