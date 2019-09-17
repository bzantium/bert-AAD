# Adversarial Discriminative Domain Adaptation with BERT
A PyTorch implementation of [Adversarial Discriminative Domain Adaptation](https://arxiv.org/abs/1702.05464) for text dataset with [pretrained BERT](https://github.com/huggingface/pytorch-pretrained-BERT).
<br> This code mainly refers to [pytorch-adda](https://github.com/corenel/pytorch-adda)

## Requirements
- pandas
- pytorch
- pytorch_pretrained_bert

### Install requirements

```
$ pip install -r requirements.txt
```
<br><br>
## Run the test

```
$ python main.py --pretrain --adapt --src books --tgt dvd
```

### Arguments

```
usage: main.py [-h] [--src {books,dvd,electronics,kitchen}]
               [--tgt {books,dvd,electronics,kitchen,blog}] [--pretrain]
               [--adapt] [--random_state RANDOM_STATE] [--seqlen SEQLEN]
               [--batch_size BATCH_SIZE] [--num_epochs_pre NUM_EPOCHS_PRE]
               [--log_step_pre LOG_STEP_PRE] [--eval_step_pre EVAL_STEP_PRE]
               [--save_step_pre SAVE_STEP_PRE] [--num_epochs NUM_EPOCHS]
               [--log_step LOG_STEP] [--eval_step EVAL_STEP]
               [--save_step SAVE_STEP]

Specify Params for Experimental Setting

optional arguments:
  -h, --help            show this help message and exit
  --src {books,dvd,electronics,kitchen}
                        Specify src dataset
  --tgt {books,dvd,electronics,kitchen,blog}
                        Specify tgt dataset
  --pretrain            Force to pretrain source encoder/classifier
  --adapt               Force to adapt target encoder
  --random_state RANDOM_STATE
                        Specify random state
  --seqlen SEQLEN       Specify maximum sequence length
  --batch_size BATCH_SIZE
                        Specify batch size
  --num_epochs_pre NUM_EPOCHS_PRE
                        Specify the number of epochs for pretrain
  --log_step_pre LOG_STEP_PRE
                        Specify log step size for pretrain
  --eval_step_pre EVAL_STEP_PRE
                        Specify eval step size for pretrain
  --save_step_pre SAVE_STEP_PRE
                        Specify save step size for pretrain
  --num_epochs NUM_EPOCHS
                        Specify the number of epochs for adaptation
  --log_step LOG_STEP   Specify log step size for adaptation
  --eval_step EVAL_STEP
                        Specify eval step size for adaptation
  --save_step SAVE_STEP
                        Specify save step size for adaptation
```
