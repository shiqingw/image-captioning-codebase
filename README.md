# Comparison of Two Attention Mechanisms Applied to Image Captioning

This GitHub repository contains the codebase of mini-project 2 of the Deep Learning course at NYU. In this project, we apply Bahdanau and Transformer attention mechanisms to the task of image captioning and compare their performances under different training settings on the Flickr8k dataset.

**Note: Most of the code is adapted from https://www.kaggle.com/code/mdteach/image-captioning-with-attention-pytorch/notebook and https://github.com/senadkurtisi/pytorch-image-captioning.**

Table of Contents:
1. [Prerequisites](#prerequisites)
2. [Flickr8k Dataset](#flickr8k-dataset)
3. [Model with Bahdanau Attention](#model-with-bahdanau-attention)
    * [Model Overview](#model-overview)
    * [Code Usage](#code-usage)
4. [Model with Transformer Layers](#model-with-transformer-layers)
    * [Model Overview](#model-overview-1)
    * [Code Usage](#code-usage-1)
5. [BLEU Scores of the Models](#bleu-scores-of-the-models)
6. [Training Log (loss, BLEU scores) and Trained Weights](#training-log-loss-bleu-scores-and-trained-weights)


## Prerequisites
The code is tested on
- Python 3.8.15
- PyTorch 1.13.0
- Torchvision 0.14.0
- nltk 3.7

## Flickr8k Dataset
In this project, the [Flickr8k](https://forms.illinois.edu/sec/1713398) dataset was used. The dataset was acquired by following instructions from the [machinelearningmastery blog](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/). You can find some dataset files on [this repo](https://github.com/jbrownlee/Datasets). 

Each image is paired with five different captions (annotated by humans), providing precise descriptions of the salient entities and events. The training set consists of 6,000 images, and the test and development (used as validation set in this project) sets each consist of 1,000 images.

You need to put the image data to the following folders:
1. `bahdanau/dataset/flickr8k/Images` for the model using Bahdanau attention;
2. `transformer/dataset/flickr8k/Images` for the model using transformer layers.

The training/validation/test split is already done in this git.

## Model with Bahdanau Attention
### Model Overview
<br>
<p align="center">
  <img src="pics\bahdanau.png"/>
</p>

### Code Usage
1. Step 1: Define your test case in `bahdanau/test_cases.py`. 
    - You can specify the optimizer (`SGD`, `Adam`, or `AdamW`), learning rate (a float number), weight decay (a float number), number of training epoches (a integer), and learning rate scheduler (`'None'` or `'CosineAnnealingLR'`). 
    - Other parameters can also be defined, such as the frequency threshhold for building the vocabulary.
    
2. Step 2: Start training with the following command. The trained weights will be saved to the folder `bahdanau/results/exp_<exp_num>`
```
python bahdanau/main.py --exp_num <exp_num>
```

3. Step 3: Run the following command to generate captions on the test set. This command will use the trained weights in the folder `bahdanau/results/exp_<exp_num>` to generate captions on 50 images in the training set.
```
python bahdanau/draw_samples.py --exp_num <exp_num>
```

## Model with Transformer Layers
### Model Overview
<br>
<p align="center">
  <img src="pics\transformer.png"/>
</p>

### Code Usage
1. Step 1: Define your test case in `transformer/test_cases`. 
    - You can specify the optimizer (`SGD`, `Adam`, or `AdamW`), learning rate (a float number), weight decay (a float number), number of training epoches (a integer), and learning rate scheduler (`'None'` or `'CosineAnnealingLR'`). 
    - - Other parameters can also be defined, such as the image size, model hyperparameters, etc.
    
2. Step 2: Start training with the following command. The trained weights will be saved to the folder `transformer/results/exp_<exp_num>`
```
python transformer/main.py --exp_num <exp_num>
```

3. Step 3: Run the following command to generate captions on the test set. This command will use the trained weights in the folder `transformer/results/exp_<exp_num>` to generate captions on 50 images in the training set.
```
python transformer/draw_samples.py --exp_num <exp_num>
```

## BLEU Scores of the Models
Below is a brief summary of the models with their BLEU-4 score and corresponding folder:

| Model       | Attention Mech.  | BLEU-4 | Folder |
| ----------- | -------------- | ----------- | ----------- |
| 1      | Bahdanau      | 0.208 |`bahdanau/results/exp_005`|
| 2      | Bahdanau      | 0.208 |`bahdanau/results/exp_006`|
| 3      | Bahdanau      | 0.210 |`bahdanau/results/exp_007`|
| 4      | Bahdanau      | 0.208 |`bahdanau/results/exp_008`|
| 5      | Bahdanau      | 0.210 |`bahdanau/results/exp_009`|
| 6      | Bahdanau      | 0.207 |`bahdanau/results/exp_010`|
| 7      | Transformer   | 0.192 |`transformer/results/exp_001`|
| 8      | Transformer   | 0.190 |`transformer/results/exp_002`|
| 9      | Transformer   | 0.192 |`transformer/results/exp_003`|
| 10     | Transformer   | 0.185 |`transformer/results/exp_004`|
| 11     | Transformer   | 0.190 |`transformer/results/exp_009`|
| 12     | Transformer   | 0.194 |`transformer/results/exp_010`|
| 13     | Transformer   | 0.185 |`transformer/results/exp_005`|
| 14     | Transformer   | 0.188 |`transformer/results/exp_006`|
| 15     | Transformer   | 0.189 |`transformer/results/exp_007`|
| 16     | Transformer   | 0.188 |`transformer/results/exp_008`|
| 17     | Transformer   | 0.185 |`transformer/results/exp_013`|
| 17     | Transformer   | 0.188 |`transformer/results/exp_014`|

## Training Log (loss, BLEU scores) and Trained Weights
You can find
1. a training log (`output.out`) that records the training/validation/test losses, BLEU scores, and time during the training;
2. a plot of training/validation/test losses (`loss.png`);
3. plots of test and validation BLEU scores (`test_bleu_scores.png` and `validation_bleu_scores.png`);
4. a dictionary recording all the losses and BLEU scores (`training_info.npy`, you need to use the `pickle` package to load it) ; 

in the corresponding folder of the model. For example, for model 5, these documents can be found at `bahdanau/results/exp_009`; for model 12, these documents can be found at `transformer/results/exp_010`.

The weights file are stored [here](https://drive.google.com/drive/folders/15oScPYEVb3gxHBuh6puOqLF2qF0mGpAq?usp=sharing).