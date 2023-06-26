## Main Technologies
PyTorch Lightning - a lightweight PyTorch wrapper for high-performance AI research. Think of it as a framework for organizing your PyTorch code.

Hydra - a framework for elegantly configuring complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

Pytorch Image Models - PyTorch Image Models (timm) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.

## How to run
 - Install dependencies
### clone project
- git clone https://github.com/minakshimathpal/The-School-of-AI.git
- cd The-School-of-AI



### install requirements
pip install -r requirements.txt

## Train model with default configuration
## train on CPU
python emloCarVsDog/train.py trainer=cpu