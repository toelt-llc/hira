#!/home/arnaud/miniconda3/bin/python
from file_loader import load_frame, nframes
from PIL import Image
import numpy as np
import os
from glob import glob
import matplotlib as mpl
from matplotlib import cm
from tqdm import tqdm


PATH = '/home/arnaud/projects/super_res/HIRA/data/raw/*.raw'
file_list = tqdm(glob(PATH))

# create png dirs & mpl norm functions
os.makedirs('./data/images_l', exist_ok=True)
os.makedirs('./data/images_h', exist_ok=True)
cmap = cm.viridis
norm = mpl.colors.Normalize()


def normalize_array(original_array, axis=None):
    min_vals = np.min(original_array, axis=axis, keepdims=True)
    max_vals = np.max(original_array, axis=axis, keepdims=True)

    normalized_array = (original_array - min_vals) * 40 / (max_vals - min_vals)

    return normalized_array.astype(np.uint8)


for file in file_list:
    # run file loader for every video
    basename = os.path.basename(file)[:-4]
    file_nframes = nframes[basename]
    for fr in tqdm(range(file_nframes)):
        low_rez, high_rez = load_frame(file, fr)

        im_h = Image.fromarray(np.uint8(cmap(norm(high_rez[16:, :]))*255))
        im_h.convert('RGB').save(f'tfrecord/images_h/{basename}_h_{str(fr)}.png')

# norm and loop again because of Normalize behaviour
# might need better normalization and python loop
# https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Normalize.html
norm = mpl.colors.Normalize()

for file in file_list:
    basename = os.path.basename(file)[:-4]
    file_nframes = nframes[basename]
    for fr in tqdm(range(file_nframes)):
        low_rez, high_rez = load_frame(file, fr)

        im_l = Image.fromarray(np.uint8(cmap(norm(low_rez))*255))
        im_l.convert('RGB').save(f'tfrecord/images_l/{basename}_l_{str(fr)}.png')
