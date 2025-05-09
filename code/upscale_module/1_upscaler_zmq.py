import zmq
import sys
import time
import queue
import struct
import logging
import numpy as np

from inference_gen import ImageUpscaler

# ZMQ reader upscaler version
# This script receives frames as NUMPY ARRAY from a ZMQ socket, processes them using an upscaling model,
# and sends the upscaled BINARY frames back out through another ZMQ socket.

ADDRESS_IN = 9999
ADDRESS_OUT = 10000
# change to 'cpu' if not using GPU
DEVICE = 'cuda:0' # or 'cuda' 

# logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('upscaler.log')
    ]
)

class ZMQUpscaler:
    def __init__(self):
        self.running = True
        self.model_loaded = False
        self.queue_in = queue.Queue()

        # input ZMQ (receiving frames from ADDRESS_IN)
        self.input_ctx = zmq.Context()
        self.input_socket = self.input_ctx.socket(zmq.SUB)
        self.input_socket.connect(f"tcp://localhost:{ADDRESS_IN}")
        self.input_socket.setsockopt_string(zmq.SUBSCRIBE, '')

        # output ZMQ (sending upscaled frames to ADDRESS_OUT)
        self.output_ctx = zmq.Context()
        self.output_socket = self.output_ctx.socket(zmq.PUB)
        self.output_socket.bind(f"tcp://*:{ADDRESS_OUT}")
        logging.info(f"Listening for frames on tcp://localhost:{ADDRESS_IN} and sending to tcp://*:{ADDRESS_OUT}")

        self.frame_count = 0
        self.start_time = time.time()

        self.init_model()

    def init_model(self):
        """Initialize upscaler model"""
        try:
            logging.info("Initializing upscaler model...")
            self.upscaler = ImageUpscaler(device=f'{DEVICE}')
            self.model_loaded = True
            logging.info("Upscaler ready")
        except Exception as e:
            logging.error(f"Error initializing upscaler: {e}")
            self.model_loaded = False

    def process_frame(self, frame):
        """Process NUMPY frame using the upscaler model"""
        if not self.model_loaded:
            return None
        try:
            # Upscale the frame
            result = self.upscaler.upscale_single_frame(frame)

            # Convert to uint8 if needed
            if result.dtype != np.uint8:
                if result.max() > 1.0:
                    result = np.clip(result, 0, 255).astype(np.uint8)
                else:
                    result = (result * 255).astype(np.uint8)

            # Test - Save the stream in a continuous binary format
            # print(result.shape)
            self.save_stream_continuous(result)

            return result
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return None

    def output_frame(self, frame):
        """Output frame in BINARY format: [height, width, channels], pixel_data"""
        if len(frame.shape) == 2:  # grayscale
            h, w = frame.shape
            channels = 1
        else:  # color (unexpected)
            h, w, channels = frame.shape

        try:
            # send metadata first (height, width, channels)
            metadata = struct.pack("III", h, w, channels)
            self.output_socket.send(metadata, zmq.SNDMORE)

            # frame data
            self.output_socket.send(frame.tobytes())
        except Exception as e:
            logging.error(f"Error sending frame: {e}")

    def save_stream_continuous(self, frame, log_file="stream.binlog"):
        """Test - Save the stream in a continuous binary format. Expects a numpy array frame"""
        if len(frame.shape) == 2:
            h, w = frame.shape
            channels = 1
        else:
            h, w, channels = frame.shape
        # write frame header (h,w,c) + data
        binary_data = struct.pack("III", h, w, channels) + frame.tobytes()
        # append frames to binary log file
        with open(log_file, 'ab') as f:
            f.write(struct.pack("III", h, w, channels))
            f.write(frame.tobytes())
            f.flush()  

    def print_stats(self):
        """Test - print processing stats"""
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        sys.stdout.write(f"\rFrames processed: {self.frame_count} | FPS: {fps:.2f} | Running for: {elapsed:.1f}s")
        sys.stdout.flush()

    def run(self):
        logging.info("Starting upscaler")

        # Start the queue worker thread
        self.queue_worker = QueueWorker(self)
        self.queue_worker.start()

        try:
            while self.running:
                try:
                    # ZMQ Receive frame
                    hw_bytes = self.input_socket.recv(flags=zmq.NOBLOCK)
                    frame_bytes = self.input_socket.recv(flags=zmq.NOBLOCK)
                    h, w = struct.unpack("HH", hw_bytes)
                    frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((h, w))

                    # upscale frame
                    processed = self.process_frame(frame)
                    if processed is not None:
                        self.output_frame(processed)
                        self.frame_count += 1
                        # print stats every 10 frames
                        if self.frame_count % 10 == 0:
                            self.print_stats()
                # optionally sleep
                except zmq.Again:
                    time.sleep(0.001)
                except KeyboardInterrupt:
                    self.running = False
                    print("\nShutting down.")
                except Exception as e:
                    logging.error(f"Error in main loop: {e}")
                    time.sleep(0.1)
        finally:
            self.shutdown()

    def shutdown(self):
        logging.info("Shutting down.")
        self.input_socket.close()
        self.output_socket.close()
        self.input_ctx.term()
        self.output_ctx.term()
        logging.info(f"Total frames processed: {self.frame_count}")

if __name__ == "__main__":
    upscaler = ZMQUpscaler()
    upscaler.run()