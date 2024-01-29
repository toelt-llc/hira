# Fine-tuning real ESRGAN

## 1. Dataset

Copy the dataset into dataset/hira/.  With the two subfolders images_h/ images_l/
Then:

```bash
python3 scripts/generate_meta_info_pairdata.py --input datasets/hira/images_h datasets/hira/images_l --meta_info datasets/hira/metainfo_paired.txt
```
Will generate the metainfo_paired.txt necessary for training.

Modify `options/finetune_realesrgan_x4plus_pairdata.yml`

## 2. Download pre-trained models

- *RealESRGAN_x4plus.pth*
    ```bash
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
    ```

- *RealESRGAN_x4plus_netD.pth*
    ```bash
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth -P experiments/pretrained_models
    ```

Downloads into [experiments/pretrained_models](experiments/pretrained_models).

## 3. Finetune

Modify [options/finetune_realesrgan_x4plus_pairdata.yml](options/finetune_realesrgan_x4plus_pairdata.yml) accordingly, especially the `datasets` part at line 16:

```yml
datasets:
  train:
    name: HIRA
    type: RealESRGANPairedDataset
    dataroot_gt: datasets/hira
    dataroot_lq: datasets/hira
    meta_info: datasets/hira/metainfo_paired.txt
    io_backend:
      type: disk

    gt_size: 240
```

And run  :
```bash
python realesrgan/train.py -opt options/finetune_realesrgan_x4plus_pairdata.yml --auto_resume
```

## 4. Evaluate

```bash
python inference_realesrgan.py -n net_g_latest  -i inputs --face_enhance
```
