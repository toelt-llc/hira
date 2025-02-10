import argparse
import ffmpeg
import glob
#import gpu
import mimetypes
import numpy as np
import os

from basicsr.utils.download_util import load_file_from_url
from loguru import logger
from os import path as osp
from tqdm import tqdm

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

def get_video_meta_info(video_path):
    """ Extracts metadata from a video file using ffmpeg 
    """
    # logger.info(f"Extracting metadata from {video_path}")
    ret = {}
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
    ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    ret['nb_frames'] = int(video_streams[0]['nb_frames'])
    ret['duration'] = float(video_streams[0]['duration'])
    return ret

def split_video(video_path, output_folder):
    """ Splits video > 2 minutes in subparts and saves them in the output folder. 
        Currently unused.
    """
    duration = get_video_meta_info(video_path)['duration']
    if duration is None:
        logger.error(f"Failed to get duration for video: {video_path}")
        return
    parts = []
    start_time = 0
    while start_time < duration:
        end_time = min(start_time + 120, duration)
        part_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_part_{int(start_time // 60)}.mp4")
        os.system(f'ffmpeg -i {video_path} -ss {start_time} -to {end_time} -c copy {part_path}')
        parts.append(part_path)
        start_time += 120
    return parts

class Reader:

    def __init__(self, args, video_path):
        self.args = args
        self.input_type = mimetypes.guess_type(video_path)[0]
        self.audio = None
        self.input_fps = None
        if self.input_type.startswith('video'):
            self.stream_reader = (
                ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='bgr24',
                                                loglevel='error').run_async(
                                                    pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
            meta = get_video_meta_info(video_path)
            self.width = meta['width']
            self.height = meta['height']
            self.input_fps = meta['fps']
            self.audio = meta['audio']
            self.nb_frames = meta['nb_frames']
        else:
            logger.error("Error - Not a video file")

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None:
            return self.input_fps
        return 10

    def get_audio(self):
        return self.audio

    def get_len(self):
        return self.nb_frames

    def __fps__(self):
        return self.input_fps

    def get_frame_from_stream(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # 3 bytes for one pixel
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img

    def close(self):
        if self.input_type.startswith('video'):
            self.stream_reader.stdin.close()
            self.stream_reader.wait()

class Writer:

    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.outscale), int(height * args.outscale)
        if out_height > 2160:
            logger.warning('You are generating video that is larger than 4K, which will be very slow due to IO speed.',
                  'We highly recommend to decrease the outscale(aka, -s).')

        if audio is not None:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 audio,
                                 video_save_path,
                                 pix_fmt='yuv420p',
                                 vcodec='libx264',
                                 loglevel='error',
                                 acodec='copy').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 video_save_path, pix_fmt='yuv420p', vcodec='libx264',
                                 loglevel='error').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))

    def write_frame(self, frame):
        frame = frame.astype(np.uint8).tobytes()
        self.stream_writer.stdin.write(frame)

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()

def inference_video(args, file, video_save_path):
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

    # ---------------------- determine model paths ---------------------- #
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

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=args.fp16 # default to False
    )

    reader = Reader(args, file)
    audio = reader.get_audio()
    height, width = reader.get_resolution()
    fps = reader.get_fps()
    in_fps, in_len = reader.__fps__(), reader.get_len()
    writer = Writer(args, audio, height, width, video_save_path, fps)
    logger.info(f'Input video: {height}x{width}, fps: {in_fps}, frames: {in_len}, length: {in_len / in_fps:.2f}s')

    pbar = tqdm(total=in_len, unit='frame', desc='inference')
    while True:
        img = reader.get_frame_from_stream()
        if img is None:
            break

        try:
            output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            logger.error('Error', error)
            logger.error('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            writer.write_frame(output)

        # torch.cuda.synchronize(device)
        pbar.update(1)

    reader.close()
    writer.close()

def run(args, file):
    logger.info(f"Processing video file {file}: ")

    args.video_name = osp.splitext(os.path.basename(file))[0]
    video_save_path = osp.join(args.output, f'{args.video_name}_{args.suffix}.mp4')
    inference_video(args, file, video_save_path)

    # check output video info
    probe = ffmpeg.probe(video_save_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    height_out = video_streams[0]['height']
    width_out = video_streams[0]['width']
    fps_out = eval(video_streams[0]['avg_frame_rate'])
    frames_out = int(video_streams[0]['nb_frames'])
    logger.info(f'Output video: {height_out}x{width_out}, fps: {fps_out}, frames: {frames_out}, length: {frames_out / fps_out:.2f}s')
    logger.info(f"Upscaled video saved to {video_save_path} \n")

    return

# @gpu.gpu_cpu_util
def main():
    """ Inference demo.
        Script simplified and customized from Real-ESRGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input video, image or folder')
    parser.add_argument( '-n', '--model_name', type=str, default='realesr-general-x4v3',
        help=('Model names: realesr-general-x4v3 (default)| realesr-animevideov3 '))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-dn', '--denoise_strength', type=float, default=0.5,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    parser.add_argument('-s', '--outscale', type=float, default=2, help='The final upsampling scale of the image. Default 2')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored video name')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing (default)')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding. Default 10.')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--fp16', action='store_true', help='Use fp16 precision. Default: fp32 (max precision).') #Stores False default
    parser.add_argument('--fps', type=float, default=None, help='FPS of the output video')
    parser.add_argument('--ffmpeg_bin', type=str, default='ffmpeg', help='The path to ffmpeg')

    args = parser.parse_args()

    args.input = args.input.rstrip('/').rstrip('\\')
    os.makedirs(args.output, exist_ok=True)

    # single file
    if mimetypes.guess_type(args.input)[0] is not None and mimetypes.guess_type(args.input)[0].startswith('video'):
        # python case test, process flv and mkv 
        match args.input.lower():
            case str(path) if path.endswith('.flv'):
                mp4_path = path.replace('.flv', '.mp4')
            case str(path) if path.endswith('.mkv'):
                mp4_path = path.replace('.mkv', '.mp4')

        if mp4_path:
            os.system(f'ffmpeg -i {args.input} -codec copy {mp4_path}')
            args.input = mp4_path

        run(args, args.input)

    # folder
    elif os.path.isdir(args.input):
        logger.info(f"Processing all videos from {args.input} folder :")
        print([os.path.basename(i) for i in glob.glob(os.path.join(args.input, '*')) if i.lower().endswith(('.mp4', '.flv', '.mkv'))], "\n")
        for file in glob.glob(os.path.join(args.input, '*')):
            if file.lower().endswith(('.mp4')):
                run(args, file)
            # convert first
            elif file.lower().endswith(('.flv', '.mkv')):
                logger.info(f"Converting video file {os.path.basename(file)} to mp4")
                mp4_path = file.replace('.flv', '.mp4') if file.lower().endswith('.flv') else file.replace('.mkv', '.mp4')
                result = os.system(f'ffmpeg -i {file} -c:v libx264 -c:a aac {mp4_path}')
                if result == 0:
                    run(args, mp4_path)
                else:
                    logger.error(f"Failed to convert file: {file}")
                    continue
            else:
                logger.warning(f"skipping non-video file: {file} \n")

if __name__ == '__main__':
    main()
