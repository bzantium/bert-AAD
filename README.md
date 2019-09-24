# Adversarial Discriminative Domain Adaptation with BERT
A PyTorch implementation of [Adversarial Discriminative Domain Adaptation](https://arxiv.org/abs/1702.05464) for text dataset with [pretrained BERT](https://github.com/huggingface/pytorch-pretrained-BERT).
<br> This code mainly refers to [pytorch-adda](https://github.com/corenel/pytorch-adda)

## Requirements
- pandas
- pytorch
- pytorch_transformers

## Run the test

```
$ python main.py --pretrain --adapt --src books --tgt dvd
```
