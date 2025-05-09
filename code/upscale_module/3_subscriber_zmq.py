import zmq
import struct
import numpy as np

# ZMQ subscriber output reader - to be run after the upscaler
# This script reads the frames from the upscaling model. 
# Options to convert to numpy array or display using OpenCV.

# adress in here is the adress out of the upscaler
address_in = 10000

context = zmq.Context()
subscriber = context.socket(zmq.SUB)
subscriber.connect(f"tcp://localhost:{address_in}")
subscriber.setsockopt_string(zmq.SUBSCRIBE, '')

while True:
    try:
        # metadata for frame reconstruction
        metadata = subscriber.recv()
        h, w, channels = struct.unpack("III", metadata)

        # receive BINARY frame
        frame_bytes = subscriber.recv()

        # To NUMPY and reshape 
        if channels == 1:
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((h, w))
        else:
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((h, w, channels))
        print(f"Received frame: {frame.shape}")

    except Exception as e:
        print(f"Error receiving frame: {e}")
        break