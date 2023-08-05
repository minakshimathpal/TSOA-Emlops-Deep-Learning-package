# Pytorch Ligtning Model Experimentation Template

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

## Overview
This project is architected to build a robust pipeline for image classification, leveraging the `CIFAR10` dataset and any accessible model from PyTorch Image Models (`TIMM`). It incorporates Docker support, offering a convenient route for reproducibility, and is equipped to conduct both training and evaluation tasks.

A significant attribute of this repository is the integration of `Hydra` for sophisticated configuration composition, command-line overrides, and the instantiation of Python objects. 

Furthermore, the project utilizes `PyTorch Lightning`, simplifying the training and evaluation processes and providing a high-level interface to PyTorch.

## Main Technologies
PyTorch Lightning - a lightweight PyTorch wrapper for high-performance AI research. Think of it as a framework for organizing your PyTorch code.

Hydra - a framework for elegantly configuring complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

Pytorch Image Models - PyTorch Image Models (timm) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.

# Key Features

1. Supports [Pytorch Custom Models](https://pytorch.org/vision/stable/models.html), [Huggingface Models](https://huggingface.co/models) and [Timm Models](https://github.com/huggingface/pytorch-image-models).
2. Model Training and Evaluation using [Pytorch Lightning Framework](https://lightning.ai/).
3. [Docker Container Based Model Training and Evaluation](https://github.com/u6yuvi/dl-package/tree/main#using-docker-containers)
3. Experiments Configuration using [Hydra Template](https://hydra.cc/).
4. Experiment Logging using:
    1. [Tensorboard](https://www.tensorflow.org/tensorboard/get_started).
    2. [Mlflow](https://github.com/mlflow/mlflow/)
    3. [Aim](https://github.com/aimhubio/aim)
5. Data Versioning using Data Version Control

## Dataset

The project utilizes the CIFAR10 dataset for image classification tasks. Refer to the [PyTorch CIFAR10 tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) for a detailed guide on how the dataset is incorporated into the project.

## Model

The project supports any image classification model available in TIMM. Models can be specified in the configuration files and are loaded during runtime.


## 🚀  Quickstart

### How to run on local

### Installation

#### Pip

```bash
# clone project
git clone https://github.com/minakshimathpal/TSOA-Emlops-Deep-Learning-package
cd TSOA-Emlops-Deep-Learning-package

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```
Train model with default/cpu configuration:

To train a model, adjust the settings in the provided configuration files and run the training script. The model's checkpoints will be saved at specified intervals, containing all necessary information to resume training or perform evaluations.

Template contains example with cat vs Dog  classification.
When running ```python classifier/train.py``` or ```classifier_train``` you should see something like this:
<div align="center">

![](https://github.com/minakshimathpal/TSOA-Emlops-Deep-Learning-package/blob/main/artifacts/terminal.png)

</div>

## How to run using Docker

```bash
# Build Docker on local
docker build -t emlov3-pytorchlightning-hydra .

# Since checkpoint will not be persisted between container runs if train and eval are run separately, use below command to run together. 
docker run emlov3-pytorchlightning-hydra sh -c "python3 ninja/train.py && python3 ninja/eval.py"

# Using volume you can mount checkpoint to host directory and run train and eval separately.
docker run --rm -t -v ${pwd}/ckpt:/workspace/ckpt emlov3-pytorchlightning-hydra python classifier/train.py
docker run --rm -t -v ${pwd}/ckpt:/workspace/ckpt emlov3-pytorchlightning-hydra python classifier/eval.py
```

## ⚡  Your Superpowers

<details>
<summary><b>Override any config parameter from command line</b></summary>

```bash
python classifier/train.py trainer.max_epochs=20 model.optimizer.lr=1e-4
```

> **Note**: You can also add new parameters with `+` sign.

```bash
python train.py +model.new_param="owo"
```
</details>

<details>
<summary><b>Train on CPU, GPU</b></summary>

```bash
# train on CPU
python classifier/train.py trainer=cpu
python classifier/eval.py

# You can override any parameter from command line like this
python classifier/train.py trainer.max_epochs=20 data.batch_size=64

# train on 1 GPU
python classifier/train.py trainer=gpu
```
</details>

An eval.py script is provided to load a model from a saved checkpoint and run it on a validation dataset. The script prints test metrics for convenient analysis.
```bash
python classifier/eval.py
```

<details>
<summary><b>Train model with chosen experiment config</b></summary>

To Run Experiments using Hydra
1. Create an experiment hydra file overiding train.yaml file
2. Run training and evaluation with experiment config

```bash
# If "experiment : null added in the train.yaml, the respective experiment.yaml(for eg cat_dog here) will overide the configuration
# If package is install with setup.py in dev mode use following
classifier_train experiment=cat_dog trainer.max_epochs=1 datamodule.batch_size=64

# If packages are installed with requirements file then use
python classifier/train.py experiment=cat_dog trainer.max_epochs=1 datamodule.batch_size=64

# If "experiment:null" not added in train.yaml.Override the train.yaml using
classifier_train +experiment=cat_dog trainer.max_epochs=1 datamodule.batch_size=64
or
python classifier/train.py +experiment=cat_dog trainer.max_epochs=1 datamodule.batch_size=64
```    
3. Run Evaluation using experiment config
```bash
classifier_eval experiment=cat_dog
```
4. Run Prediction/prediction using experiment config
```bash
# If installed in dev mode, run infer with 
# experiment/cat_dog_infer.yaml using
classifier_predict experiment=cat_dog_infer test_path=.data/PetImages_split/test/Cat/15.jpg

# If installed using requirements.txt, use
python classifier/infer.py experiment=cat_dog_infer test_path=./data/PetImages_split/test/Cat/15.jpg
```

> **Note**: Experiment configs are placed in [configs/experiments/](configs/experiments/).
</details>

Running Experiments you would see somthing like this 
- Training using  ```classifier/train.py experiments=cat_dog```  or  ```classifier_train experiments=cat_dog``` :
<div align="center">

![](https://github.com/minakshimathpal/TSOA-Emlops-Deep-Learning-package/blob/main/artifacts/terminal.png)

</div>

- Evaluation using  ```classifier/eval.py experiments=cat_dog``` or ```classifier_eval experiments=cat_dog``` :
<div align="center">

![](https://github.com/minakshimathpal/TSOA-Emlops-Deep-Learning-package/blob/main/artifacts/terminal.png)

</div>

## DVC Configuration
#Set remote storage for storing data and model artifacts
```bash
dvc remote add -d local <path_to_local_directory>

#Push data to remote directory
dvc push data outputs

#Pull data from remote directory
dvc pull

# To switch between versions of code and data run
git checkout master
dvc checkout
```

## Run Multi-Run Experiments using Hydra 
### Without Docker Container
1. Run experiment
```
classifier_train -m hydra/launcher=joblib hydra.launcher.n_jobs=4 experiment=cifar_vit model.net.patch_size=1,2,4,8,16 data.num_workers=4
```
#experiment logs are saved under logs/ folder.

2. Run AIM UI
```
aim up
```
3. Run Tensorboard
```
tensorboard --logdir=logs/tensorboard
```
4. Run MLFlow
```
mlflow ui
```
## Maintainers
  1. Minakshi Mathpal
  
