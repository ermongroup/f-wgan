# Experiments on image generation

Here are the code to reproduce the results in the image generation experiments for KL-WGAN.

## Files

- `losses.py`: definition for the new objective
- `train.py`: training script
- `train_fns.py`: training functions containing the GAN training logics
- `BigGAN.py`: model architecture

## How to train

See `train_C10.sh` and `train_C10U.sh` for more details. The important hyperparameters are:
- `--loss`: the loss function (hinge or kl)
- `--conditional` whether use conditional GAN or not

## Acknowledgements

Based on this repository: [https://github.com/ajbrock/BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch)
