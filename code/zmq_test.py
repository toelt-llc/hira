import threading
import queue
import time
import numpy as np
import zmq
import cv2

# Simulated model import
from inference_gen import ImageUpscaler  # Replace with actual import

queue_in = queue.Queue(maxsize=50)
queue_out = queue.Queue(maxsize=50)

shutdown_event = threading.Event()

def main_thread():
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://localhost:5555")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, '')

    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://*:5556")

    while not shutdown_event.is_set():
        try:
            frame_bytes = subscriber.recv(flags=zmq.NOBLOCK)
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((H, W))  # Example shape

            if not queue_in.full():
                queue_in.put(frame)

            if not queue_out.empty():
                processed = queue_out.get()
                publisher.send(processed.tobytes())
                cv2.imshow("Upscaled Frame", processed.astype(np.uint8))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    shutdown_event.set()
                    break

        except zmq.Again:
            time.sleep(0.001)
            continue

def secondary_thread():
    upscaler = ImageUpscaler(device='cuda')  # or 'cuda:0'
    
    while not shutdown_event.is_set():
        try:
            frame = queue_in.get(timeout=0.1)
            result = upscaler.upscale_single_frame(frame)
            if result is not None:
                queue_out.put(result)
        except queue.Empty:
            continue

# Start threads
main = threading.Thread(target=main_thread)
secondary = threading.Thread(target=secondary_thread)

main.start()
secondary.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    shutdown_event.set()
    main.join()
    secondary.join()
