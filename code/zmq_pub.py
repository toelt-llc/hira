# zmq_publisher_test.py
import zmq
import time
import numpy as np

context = zmq.Context()
publisher = context.socket(zmq.PUB)
publisher.bind("tcp://*:5555")

time.sleep(1)  # Give subscriber time to connect

H, W = 480, 640
for i in range(10):
    frame = np.random.randint(0, 256, (H, W), dtype=np.uint8)
    publisher.send(frame.tobytes())
    print("Published frame")
    time.sleep(0.05)
