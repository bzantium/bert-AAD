import os
import random
import numpy as np
import pandas as pd
import torch
from lxml import etree
import xml.etree.ElementTree as ET
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import param
import re


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id


def XML2Array(neg_path, pos_path):
    parser = etree.XMLParser(recover=True)
    reviews = []
    negCount = 0
    posCount = 0
    labels = []
    regex = re.compile(r'[\n\r\t+]')

    neg_tree = ET.parse(neg_path, parser=parser)
    neg_root = neg_tree.getroot()

    for rev in neg_root.iter('review_text'):
        text = regex.sub(" ", rev.text)
        reviews.append(text)
        negCount += 1
    labels.extend(np.zeros(negCount, dtype=int))

    pos_tree = ET.parse(pos_path, parser=parser)
    pos_root = pos_tree.getroot()

    for rev in pos_root.iter('review_text'):
        text = regex.sub(" ", rev.text)
        reviews.append(text)
        posCount += 1
    labels.extend(np.ones(posCount, dtype=int))

    reviews = np.array(reviews)
    labels = np.array(labels)

    return reviews, labels


def CSV2Array(path):
    data = pd.read_csv(path, encoding='latin')
    reviews, labels = data.reviews.values.tolist(), data.labels.values.tolist()
    return reviews, labels


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def init_random_seed(manual_seed):
    """Init random seed."""
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_model(args, net, restore=None):
    # restore model weights
    if restore is not None:
        path = os.path.join(param.model_root, args.src, args.model, str(args.seed), restore)
        if os.path.exists(path):
            net.load_state_dict(torch.load(path))
            print("Restore model from: {}".format(os.path.abspath(path)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()
    return net


def save_model(args, net, name):
    """Save trained model."""
    folder = os.path.join(param.model_root, args.src, args.model, str(args.seed))
    path = os.path.join(folder, name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(net.state_dict(), path)
    print("save pretrained model to: {}".format(path))


def convert_examples_to_features(reviews, labels, max_seq_length, tokenizer,
                                 cls_token='[CLS]', sep_token='[SEP]',
                                 pad_token=0):
    features = []
    for ex_index, (review, label) in enumerate(zip(reviews, labels)):
        if (ex_index + 1) % 200 == 0:
            print("writing example %d of %d" % (ex_index + 1, len(reviews)))
        tokens = tokenizer.tokenize(review)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]
        tokens = [cls_token] + tokens + [sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          label_id=label))
    return features


def roberta_convert_examples_to_features(reviews, labels, max_seq_length, tokenizer,
                                         cls_token='<s>', sep_token='</s>',
                                         pad_token=1):
    features = []
    for ex_index, (review, label) in enumerate(zip(reviews, labels)):
        if (ex_index + 1) % 200 == 0:
            print("writing example %d of %d" % (ex_index + 1, len(reviews)))
        tokens = tokenizer.tokenize(review)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]
        tokens = [cls_token] + tokens + [sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          label_id=label))
    return features


def get_data_loader(features, batch_size):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


def MMD(source, target):
    mmd_loss = torch.exp(-1 / (source.mean(dim=0) - target.mean(dim=0)).norm())
    return mmd_loss
