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
#### Current model
https://github.com/xinntao/Real-ESRGAN

#### model to try
* Unet (needs to be implemented, current models are 4+ years old, e.g., https://github.com/cerniello/Super_Resolution_DNN?tab=readme-ov-file)
* https://github.com/IceClear/StableSR

#### losses to try
* Selective SSIM
* Charbonnier loss

## Partners
The partner institutions from the HIRA project are the following:
* Next2U (ITA)
* SUPSI IDSIA (CHE)



## bugs
if `pytorch>2.1.1`, it is possible that `basicsr` stops running because this library hasn't been updated since 2022. For that it is necessary to modify `basicsr/data/degradation.py` line 8

```from torchvision.transforms.functional_tensor import rgb_to_grayscale```

to

```from torchvision.transforms.functional import rgb_to_grayscale```
