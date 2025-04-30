import threading
import queue
import time
import numpy as np
import zmq
import cv2
import logging

from inference_gen import ImageUpscaler  # Replace with your model

H, W = 480, 640

queue_in = queue.Queue(maxsize=50)
queue_out = queue.Queue(maxsize=50)

shutdown_event = threading.Event()

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

def main_thread():
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://localhost:5555")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, '')

    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://*:5556")

    frame_count = 0
    last_time = time.time()

    cv2.namedWindow("Original vs Upscaled", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original vs Upscaled", 1280, 480)

    while not shutdown_event.is_set():
        try:
            frame_bytes = subscriber.recv(flags=zmq.NOBLOCK)
            logging.info(f"Received {len(frame_bytes)} bytes")
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((H, W))

            if not queue_in.full():
                queue_in.put(frame)
                logging.info("Put frame into input queue")

            if not queue_out.empty():
                processed = queue_out.get()
                publisher.send(processed.tobytes())
                logging.info("Sent processed frame")

                # Convert grayscale to BGR for visualization
                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                processed_color = cv2.cvtColor(processed.astype(np.uint8), cv2.COLOR_GRAY2BGR)

                # FPS counter
                frame_count += 1
                if frame_count % 10 == 0:
                    now = time.time()
                    fps = 10 / (now - last_time)
                    last_time = now
                else:
                    fps = None

                if fps:
                    cv2.putText(processed_color, f"FPS: {fps:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                combined = np.hstack((frame_color, processed_color))
                cv2.imshow("Original vs Upscaled", combined)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    shutdown_event.set()
                    break

        except zmq.Again:
            time.sleep(0.001)
        except Exception as e:
            logging.error(f"Error in main_thread: {e}")
            time.sleep(0.1)

    cv2.destroyAllWindows()

def secondary_thread():
    upscaler = ImageUpscaler(device=None)
    logging.info("Upscaler initialized")

    while not shutdown_event.is_set():
        try:
            frame = queue_in.get(timeout=0.1)
            logging.info("Got frame from input queue")
            result = upscaler.upscale_single_frame(frame)
            if result is not None:
                queue_out.put(result)
                logging.info("Put processed frame into output queue")
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Error in secondary_thread: {e}")
            time.sleep(0.1)

if __name__ == "__main__":
    secondary = threading.Thread(target=secondary_thread)
    secondary.start()

    try:
        main_thread()
    except KeyboardInterrupt:
        shutdown_event.set()
    finally:
        secondary.join()
