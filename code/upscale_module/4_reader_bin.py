import cv2
import struct
import numpy as np

# ZMQ subscriber output reader 
# to be run after the upscaler is completed

def read_binlog(binlog_file="stream.binlog"):
    """Test - Generator to yields frames from the binary log"""
    with open(binlog_file, 'rb') as f:
        while True:
            # header
            header = f.read(12)
            if not header:
                break 
            h, w, channels = struct.unpack("III", header)
            frame_size = h * w * channels

            # Read frame data
            frame_data = f.read(frame_size)
            if len(frame_data) != frame_size:
                raise ValueError("Corrupted frame data")

            # Reconstruct frame
            if channels == 1:
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((h, w))
            else:
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((h, w, channels))

            yield frame

# Example usage, uncomment for display:
for i, frame in enumerate(read_binlog()):
    print(f"Frame {i}: Shape={frame.shape}, dtype={frame.dtype}")
    # cv2.imshow("Frame", frame)
    # if cv2.waitKey(30) == 27:  # ESC to exit
    #     break