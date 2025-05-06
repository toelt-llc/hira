import zmq
import time
import numpy as np
import struct
from inference_binary import read_binary, inference_realesrgan
from inference_frame import inference_realesrgan_frame
import matplotlib.pyplot as plt

video = './inputs/S002-20250109/Rec-20250109-131827_lep2.zsc'
frames_array, num_frames, total_time, video_rate = read_binary(video)

context = zmq.Context()
publisher = context.socket(zmq.PUB)
publisher.bind("tcp://*:9999")

H, W = 60, 80  # Must match your application's expected dimensions

# Warm-up period for subscribers to connect
time.sleep(1)

for i in range(1000):
    # Create frame as uint8 instead of uint16
    # frame = np.random.randint(0, 256, (H, W), dtype=np.uint8)
    # frame 
    
    # Send frame dimensions first
    # print(frames_array.shape)
    plt.imshow(frames_array[i])
    #plt.show()
    publisher.send(struct.pack("HH", H, W), zmq.SNDMORE)
    # Then send frame data
    publisher.send(frames_array[i].tobytes())
    
    print(f"Published frame {i+1}")
    time.sleep(0.05)