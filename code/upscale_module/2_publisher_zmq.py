import zmq
import time
import struct
import matplotlib.pyplot as plt

from inference_binary import read_binary

# ZMQ publisher - to be run before the upscaler or while it is running
# This script reads a binary video file and sends the frames over ZMQ publisher at selected adress.
# ! Currently loads binary here and convert to numpy array before publishing.

adress_in = 9999

video = './inputs/S002-20250109/Rec-20250109-131827_lep2.zsc'
frames_array, num_frames, total_time, video_rate = read_binary(video)
# print(frames_array.shape)

context = zmq.Context()
publisher = context.socket(zmq.PUB)
publisher.bind(f"tcp://*:{adress_in}")

H, W = 60, 80  # Must match your application's expected dimensions

# Warm-up period for subscribers to connect
time.sleep(1)

for i in range(len(frames_array)):
    # plot frame 
    # plt.imshow(frames_array[i])
    # plt.show()

    # send frame dimensions first
    publisher.send(struct.pack("HH", H, W), zmq.SNDMORE)
    # then send frame data
    publisher.send(frames_array[i].tobytes())

    print(f"Published frame {i+1}")
    # processing speed tests
    time.sleep(0.005)