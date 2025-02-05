# Hira Upscale Video Inference

## Overview

This repository contains a script to perform video inference using several models. The script supports various video formats and can handle large videos by splitting them into manageable parts. It upscales videos and save the results.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Arguments](#arguments)
- [Examples](#examples)
- [Models Evaluated](#models-evaluated)
- [Data](#data)
- [Implementation](#implementation)
- [Partners](#partners)
- [Contributing](#contributing)
- [License](#license)

## Installation

**Prerequisites**

- Python 3.11
- Conda (for environment management)

### Steps

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/hira.git
    cd hira
    ```

2. **Create a Conda environment:**

    ```sh
    conda create -n hira python=3.11
    conda activate hira
    ```

3. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Note on model weights:**

    The script will automatically download the necessary default model weights if they are not already present in the `weights` directory.

## Usage

### Basic Usage

By default, the command looks for files in the *code/inputs/* folder. Put your samples in this folder.
The upscaled results outputs are in *code/results*. Default upscales x2.

Command options:

```sh
cd code
python3 inference.py -h
```

To run the inference script on a single video file:

```sh
python3 inference.py -i path/to/input/video.mp4 -o path/to/output/directory
```
### Folder Input

To process all videos in a directory:

```sh
python3 inference.py -i path/to/input/directory -o path/to/output/directory
```

For a reference pipeline of the model training and setup of the workflow, refer to the [notebook](./notebook.ipynb)

### Splitting Long Videos

If a video is longer than 2 minutes, the script will automatically split it into parts of 2 minutes each and process them individually. The processed parts will be saved in subfolders named after the input video.

## Arguments

- `-i`: Path to the input video or directory containing videos.
- `-o`: Output directory location to save the processed videos.
- `-s`: The final upsampling scale of the image (default: 2).
- `--fps`: FPS of the output video. Default to input FPS.
- `-n`: Model name to use for inference. Options include `realesr-general-x4v3` (default) and a custom trained model (TBS)
- `-dn`: Denoise strength for the default model, varying results based on the input. (0 for weak denoise, 1 for strong denoise).

*additional arguments*
- `--suffix`: Suffix of the restored video name (default: `out`).
- `--fp16`: Use fp16 precision (default: fp32).
Options to upscale subtiles of the input (useful for larger resolutions.)
- `-t`: Tile size for processing large images (default: 0).
- `--tile_pad`: Tile padding (default: 10).
- `--pre_pad`: Pre padding size at each border (default: 0).

## Examples
**Single Video**

```sh
python inference.py -i inputs/video.mp4 -o results -n realesr-general-x4v3 -s 4
```
**Directory of Videos**

```sh
python inference.py -i inputs -o results -n realesr-animevideov3 -s 2
```

## Models Evaluated

LIST of models implementations:  
[x] [RealESRGAN](https://github.com/xinntao/Real-ESRGAN)  
[x] [keras base model](hira/models/keras)  
[~] [hugginface models](hira/models/hf_models):  
[ ] [Stable SR](https://github.com/IceClear/StableSR)  

<!-- ## Partners

The partner institutions from the HIRA project are the following:

* Next2U (ITA)
* SUPSI IDSIA (CHE)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. -->