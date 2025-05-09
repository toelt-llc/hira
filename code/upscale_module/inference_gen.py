import numpy as np
import os

from loguru import logger
from PIL import Image
from tqdm import tqdm

from basicsr.utils.download_util import load_file_from_url
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer

import smartcamDecoder

## Same function as in inference_binary.py, but without the command line args
## Same function as in inference_frame.py, but with some generator logic

class ImageUpscaler:

    def __init__(self, model_name='realesr-general-x4v3', outscale=2, denoise_strength=0, fp16=False, frame_array=None, device=1):
        self.model_name = model_name
        self.outscale = outscale
        self.denoise_strength = denoise_strength
        self.fp16 = fp16
        self.frame_array = frame_array
        self.device = device
        self.dni_weight = None
        self.upsampler = None
        self.height = None
        self.width = None
        self.pbar = None
        self.sr_model = self._build_sr_model(model_name, fp16)

    def _build_sr_model(self, model_name, fp16=False):

        if model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu') #upscale at 4 for weights fit
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
        if model_name == 'realesr-general-x4v3' and self.denoise_strength != 1:
            wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            model_path = [model_path, wdn_model_path]
            self.dni_weight = [self.denoise_strength, 1 - self.denoise_strength]

        upsampler_model = RealESRGANer(
            device=self.device,
            scale=4,
            model_path=model_path,
            dni_weight=self.dni_weight,
            model=model,
            half=self.fp16 # default to False
        )

        return upsampler_model

    def _build_input_frames(self, frame_array):
        # handles single or multiple frames
        if len(frame_array.shape) == 2:
            frame_array = np.expand_dims(frame_array, axis=0)

        # return frame_array
        for frame in frame_array:
            yield frame


        # while True:
        #     for frame in frame_array:
        #         if frame is None:
        #             break
        #         yield frame

    def upscale_frames(self, frame_array):
        """ Takes 1 or more input frames and upscales them using Real-ESRGAN."""
        frame_gen = self._build_input_frames(frame_array)
        first_frame = next(frame_gen)
        height, width = first_frame.shape
        frames = [first_frame]

        for frame in frame_gen:
            frames.append(frame)

        logger.info(f'Upscaling to resolution: {height*self.outscale}x{width*self.outscale}, computing {len(frames)} frames.')
        pbar = tqdm(total=len(frames), unit='frame', desc='inference')
        up_frames_np = np.empty((len(frames), height*self.outscale, width*self.outscale), dtype=np.uint16)

        for i, frame in enumerate(frames):
            if frame is None:
                break
            try:
                output, _ = self.sr_model.enhance(frame, outscale=self.outscale)
            except RuntimeError as error:
                logger.error('Error', error)
            else:
                up_frames_np[i] = output
            pbar.update(1)

        return up_frames_np

    def upscale_single_frame(self, frame):
        try:
            height, width = frame.shape
            up_frames_np = np.empty((1, height*self.outscale, width*self.outscale), dtype=np.uint16)
            output, _ = self.sr_model.enhance(frame, outscale=self.outscale)
            up_frames_np[0] = output
            return up_frames_np
        except RuntimeError as error:
            logger.error('Error during inference: {}'.format(error))
            return None

def read_binary(input_path):
    """ Read binary file and return the number of frames, total time and video rate.
    """
    decoder = smartcamDecoder.SmartCamDecoder()
    num_frames, total_time, video_rate = decoder.on_file(input_path)
    # input for demo
    num_frames = 2000

    decoder.read_frame(0)
    vImg, vt, tImg, tt = decoder.get_images()
    height, width = tImg.shape
    logger.info(f'Input resolution: {height}x{width}, fps: {int(video_rate)}, frames: {num_frames}, length: {int(total_time)}s')

    frames_np = np.empty((num_frames, height, width), dtype=np.uint8)  # Preallocate
    for i in tqdm(range(num_frames), unit='frame'):
        # convert
        decoder.read_frame(i)
        vImg, vt, tImg, tt = decoder.get_images()
        frame_np = np.array(tImg)
        frame_array_scaled_255 = ((frame_np - frame_np.min()) / (frame_np.max() - frame_np.min()) * 255)
        frames_np[i] = frame_array_scaled_255  # Assign 

    return frames_np, num_frames, total_time, video_rate

# shape = (2000, 60, 80)
# array = np.random.rand(*shape)
# upscaler = ImageUpscaler(frame_array=array)
# print(upscaler.upscale_frames(array).shape)