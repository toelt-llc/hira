import os
import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from basicsr.utils.download_util import load_file_from_url
from loguru import logger
from matplotlib import cm
# from pims import Frame
from PIL import Image
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import smartcamDecoder
from tqdm import tqdm

## Same function as in inference_binary.py, but without the command line args

def inference_realesrgan_frame(frame_array, outscale=2, model_name='realesr-general-x4v3', denoise_strength=0, fp16=False):#, video_save_path):
    # ---------------------- determine models according to model names ---------------------- #
    if model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
    else:
        logger.error(f'Error: model name {model_name} is not supported.')

    # ---------------------- determine model paths, auto-download ---------------------- #
    model_path = os.path.join('weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # use dni to control the denoise strength, base model only
    dni_weight = None
    if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1 - denoise_strength]

    # restorer, simplest form, no padding 
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        half=fp16 # default to False
    )

    if len(frame_array.shape) == 2:
        frame_array = np.expand_dims(frame_array, axis=0)
    height, width = frame_array.shape if len(frame_array.shape) == 2 else frame_array[0].shape

    outscale = int(outscale) # default x2 upscale
    logger.info(f'Upscaling to resolution: {height*outscale}x{width*outscale}, computing {frame_array.shape[0]} frames.')
    pbar = tqdm(total=frame_array.shape[0], unit='frame', desc='inference')
    i = 0
    up_frames_np = np.empty((frame_array.shape[0], height*outscale, width*outscale), dtype=np.uint16) # Pre-allocate
    while i < frame_array.shape[0]:
        frame = frame_array[i]
        if frame is None:
            break

        try:
            output, _ = upsampler.enhance(frame, outscale=outscale)
        except RuntimeError as error:
            logger.error('Error', error)
        else:
            up_frames_np[i] = output

        i += 1
        pbar.update(1)
    return up_frames_np