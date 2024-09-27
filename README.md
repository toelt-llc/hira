# HIRA PROJECT -- Hybrid InfraRed System for Affective Computing

## Documentation
The HIRA project is involved with performing super-resolution on infrared images 

### Data 
* Grayscale pictures
* Lepton camera (60x80)
* Boson camera (256x320)
* 3 different sets of frames for a total of circa 3000k pictures
* 16-bit images
* Image range values between \[27476-23431\] (boson), \[31267-29707\] (lepton)

### Models
#### Current best model
[RealESRGAN](https://github.com/xinntao/Real-ESRGAN)

#### Models evaluated
* [RealESRGAN](https://github.com/xinntao/Real-ESRGAN)
* [Stable SR](https://github.com/IceClear/StableSR)
* [keras base model](hira/models/keras)
* [hugginface models](hira/models/hf_models):
    - Latent Diffusion
    - GigaGan (1 implementation)

### Implementation
For a reference pipeline and setup of the workflow, refer to the [notebook](./notebook.ipynb)

___
## Partners
The partner institutions from the HIRA project are the following:
* Next2U (ITA)
* SUPSI IDSIA (CHE)
