from file_loader import load_frame, nframes, frame_dtype

import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui
import sys,  os
from glob import glob
import tensorflow as tf

#import pyqtgraph as pg
#import pyqtgraph.dockarea

#print(nframes['lb1'])
#path = '../initial_superres_videos/*.raw'
path = 'tfrecord/vid3/*.raw'

file_list = glob(path)
file_dict = {}
# create separate dictionaries for high and low rez
high_dict = {}
low_dict = {}

high_list = []
low_list = []

# run file loader for every video
for file in file_list:
    basename = os.path.basename(file)[:-4]
    file_nframes = nframes[basename]
    for fr in range(file_nframes):
        low_rez, high_rez = load_frame(file,fr)
        file_dict[basename+'_l_'+str(fr)] = low_rez
        file_dict[basename+'_h_'+str(fr)] = high_rez

        low_dict[basename+str(fr)] = low_rez
        high_dict[basename+str(fr)] = high_rez
        
        low_list.append(low_rez)
        high_list.append(high_rez)



print(file_dict.keys())
#print(type(file_dict['lb1_h_831']))
#print(file_dict['lb1_h_831'])
#print(file_dict['lb1_h_831'].shape)
#print(file_dict['lb1_l_831'].shape)

#print(file_dict.values())


# create tensors from lists
low_tensor = tf.constant(low_list)
high_tensor = tf.constant(high_list)

rgb_low_tensor = tf.expand_dims(low_tensor, axis=-1)
rgb_low_tensor = tf.concat([rgb_low_tensor, rgb_low_tensor, rgb_low_tensor], axis=-1)

rgb_high_tensor = tf.expand_dims(high_tensor, axis=-1)
rgb_high_tensor = tf.concat([rgb_high_tensor, rgb_high_tensor, rgb_high_tensor], axis=-1)

print(low_tensor.shape)
print(rgb_low_tensor.shape)
print(high_tensor.shape)
print(rgb_high_tensor.shape)

# "example" is of type tf.train.Example.
x2 = tf.io.serialize_tensor(rgb_low_tensor)
print("low tensor serialized")
with tf.io.TFRecordWriter('srlowdata3.tfrecord') as writer:
  writer.write(x2.numpy())
print("low tensor saved")

x3 = tf.io.serialize_tensor(rgb_high_tensor)
print("high tensor serialized")
with tf.io.TFRecordWriter('srhighdata3.tfrecord') as writer:
  writer.write(x3.numpy())

print("high tensor saved")
