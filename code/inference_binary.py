import os
import argparse

import matplotlib.pyplot as plt
import numpy as np

from basicsr.utils.download_util import load_file_from_url
from loguru import logger
from matplotlib import cm
from pims import Frame
from PIL import Image
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from smartcam import smartcamDecoder
from tqdm import tqdm

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

def create_video_gif(input_frames, output_name, fps=100):
    """ Create a gif from a list of frames (expects NumPy).
    """
    imgs = [Image.fromarray(cm.viridis(frame, bytes=True)) for frame in input_frames]
    imgs[0].save(f"{output_name}.gif", save_all=True, append_images=imgs[1:], duration=1/fps, loop=0)

def inference_realesrgan(args, frame_array):#, video_save_path):
    # ---------------------- determine models according to model names ---------------------- #
    args.model_name = args.model_name.split('.pth')[0]
    if args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
    else:
        logger.error(f'Error: model name {args.model_name} is not supported.')

    # ---------------------- determine model paths, auto-download ---------------------- #
    model_path = os.path.join('weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # use dni to control the denoise strength, base model only
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # restorer, simplest form, no padding 
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        half=args.fp16 # default to False
    )

    height, width = frame_array.shape if len(frame_array.shape) == 2 else frame_array[0].shape

    logger.info(f'Upscaling to resolution: {height*args.outscale}x{width*args.outscale}, computing {frame_array.shape[0]} frames.')
    pbar = tqdm(total=frame_array.shape[0], unit='frame', desc='inference')
    i = 0
    up_frames_np = np.empty((frame_array.shape[0], height*args.outscale, width*args.outscale), dtype=np.uint16) # Pre-allocate
    while i < frame_array.shape[0]:
        frame = frame_array[i]
        if frame is None:
            break

        try:
            # numpy rescale
            # frame_np = np.array(frame, dtype=np.float32)
            # frame_scaled_256 = ((frame - frame_np.max()) / (frame_np.max() - frame_np.min()) * 255).astype(np.uint8)
            # print(frame_scaled_256.shape)
            output, _ = upsampler.enhance(frame, outscale=args.outscale)
            # return output
        except RuntimeError as error:
            logger.error('Error', error)
        else:
            # writer.write_frame(output)
            up_frames_np[i] = output

        i += 1
        pbar.update(1)
    return up_frames_np

# @gpu.gpu_cpu_util
def main():
    """ Inference demo.
        Script simplified and customized from Real-ESRGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input video, image or folder')
    parser.add_argument( '-n', '--model_name', type=str, default='realesr-general-x4v3',
        help=('Model names: realesr-general-x4v3 (default)| finetuned-realesr'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-dn', '--denoise_strength', type=float, default=0,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    parser.add_argument('-s', '--outscale', type=float, default=2, help='The final upsampling scale of the image. Default 2')
    parser.add_argument('--fp16', action='store_true', help='Use fp16 precision. Default: fp32 (max precision).') #Stores False default
    parser.add_argument('--fps', type=float, default=None, help='FPS of the output video')
    parser.add_argument('--debug', action='store_true', help='For debugging')
    args = parser.parse_args()

    args.input = args.input.rstrip('/').rstrip('\\')
    print(args.input)
    if args.debug:
        print("start")
        frames_np, num_frames, total_time, video_rate = read_binary(args.input)
        create_video_gif(frames_np[:1000], 'test') # save only the first 1000 frames
        print("end")
    else:
        logger.info(f"Processing video file :  {args.input}")
        frames_np, num_frames, total_time, video_rate = read_binary(args.input)
        create_video_gif(frames_np[:1000], 'low') # save only the first 1000 frames
        up_frames_np = inference_realesrgan(args, frames_np)
        create_video_gif(up_frames_np[:1000], 'high') # save only the first 1000 frames

        # Read lep 3 for comp
        lep3 = args.input.replace("lep2", "lep3")
        logger.info(f"Reading video file :  {lep3}")
        frames_np_lep3, num_frames, total_time, video_rate = read_binary(lep3)
        create_video_gif(frames_np_lep3[:1000], 'lep3')

        print(up_frames_np.shape)
        fig, axs = plt.subplots(1, 3, figsize=(17, 6))
        axs[0].imshow(frames_np[0], cmap='gray')
        axs[0].set_title('Thermal Image Lep2')
        axs[0].axis('off')

        axs[1].imshow(up_frames_np[0], cmap='gray')
        axs[1].set_title('Thermal Image Lep2 x2')
        axs[1].axis('off')

        axs[2].imshow(up_frames_np[0], cmap='gray')
        axs[2].set_title('Thermal Image Lep3 ')
        axs[2].axis('off')

        plt.savefig(f"first_frame_{args.input[21:-4]}.png")
        plt.show()

if __name__ == '__main__':
    main()
