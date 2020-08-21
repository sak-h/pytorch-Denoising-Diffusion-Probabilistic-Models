# Unofficial PyTorch implementation of Denoising Diffusion Probabilistic Models

Unofficial PyTorch implementation of [Denoising Diffusion Probabilistic Models](https://hojonathanho.github.io/diffusion).

## Reference
```
Denoising Diffusion Probabilistic Models
Jonathan Ho, Ajay Jain, Pieter Abbeel
Paper: https://arxiv.org/abs/2006.11239
Website: https://hojonathanho.github.io/diffusion
Citation:
@article{ho2020denoising,
    title={Denoising Diffusion Probabilistic Models},
    author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
    year={2020},
    journal={arXiv preprint arxiv:2006.11239}
}
```


This code requires PyTorch 1.3 and Python 3.6, and these dependencies (see `requirements.txt`):
```
pip3 install pillow
pip3 install tensorboard
```

## Dataset
You need to prepare your own dataset to use. The training images need to be in `DATAROOT/train` like:
```
cifer/
    train/
        image001.png
        image002.png
        :
``` 

## Model training
```
python train.py --name PROJECT_NAME --dataroot DATAROOT --checkpoints_dir CHECK_POINTS_DIR --log_dir LOG_DIR --batch_size BATCH_SIZE
```
See `utils/config.py` and `train.py` to find other options.
Instead, you can set options by specifying a config file (, which is generated as `CHECK_POINTS_DIR/NAME/train_option.yaml` when the script run).
```
python train.py --config_file CHECK_POINTS_DIR/NAME/train_option.yaml
```


## Sampling images
```
python sample.py --name PROJECT_NAME --dataroot DATAROOT --checkpoints_dir CHECK_POINTS_DIR --results_dir RESULTS_DIR --batch_size BATCH_SIZE
```
See `utils/config.py` and `sample.py` to find other options.


## Interpolating images
The interpolation script outputs images as `RESULTS_DIR/ORIGINAL_FILENAME_A+ORIGINAL_FILENAME_B`
```
python interpolate.py --name PROJECT_NAME --dataroot DATAROOT --checkpoints_dir CHECK_POINTS_DIR --results_dir RESULTS_DIR
```
See `utils/config.py` and `interpolate.py` to find other options.


## Acknowledgements
This implementation borrows heavily from [the official Tensorflow implementation](https://github.com/hojonathanho/diffusion).

Many utility codes are borrowed from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).