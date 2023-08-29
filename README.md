# ICDAR 2021 Competition
Official Code for the 4rd Place in Multimodal Emotion Recognition on Comics scenes Challenge
[Link](https://competitions.codalab.org/forums/24580/5534/)

## Table of Contents
- [Installation](#Installation)
- [Training](#Training)
- [Inference](#Inference)
- [Data](#Data)
- [Automatically Logged Data](#loggeddata)

## Installation
**Requirements:**
- numpy==1.19.2
- pandas==1.1.5
- pytorch==1.8.1
- torchtext==0.8.0
- torchvision==0.9.1
- jupyter==1.0.0
- matplotlib==3.3.4
- scipy==1.5.4
- scikit-learn==0.24.1
- ipython==7.16.1
- librosa==0.8.0
- transformers==4.4.2
- timm==0.4.5
- wandb==0.10.33
  
## Training

- Run the Python code for training

```
bash train.sh
```
- After training, the checkpoint is saved in the folder "checkpoint"
  
## Inference

- Run the Python code for testing

```
bash test.sh
```

## Data

The dataset is only downloaded from the challenge [Dataset](https://sites.google.com/view/emotion-recognition-for-comics/dataset)

## Automatically Logged Data
We utilize Wandb to monitor the training and evaluation phases and employ a sweep for parameter optimization. The Wandb link will be displayed in the terminal when executing the code.

