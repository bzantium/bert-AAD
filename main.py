"""Main script for ADDA."""

import param
from train import pretrain, adapt, evaluate
from model import BERTEncoder, BERTClassifier, Discriminator
from utils import XML2Array, CSV2Array, convert_examples_to_features, \
    get_data_loader, init_model
from sklearn.model_selection import train_test_split
from pytorch_transformers import BertTokenizer
import os
import argparse


def parse_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")

    parser.add_argument('--src', type=str, default="books",
                        choices=["books", "dvd", "electronics", "kitchen", "blog", "airline"],
                        help="Specify src dataset")

    parser.add_argument('--tgt', type=str, default="dvd",
                        choices=["books", "dvd", "electronics", "kitchen", "blog", "airline"],
                        help="Specify tgt dataset")

    parser.add_argument('--pretrain', default=False, action='store_true',
                        help='Force to pretrain source encoder/classifier')

    parser.add_argument('--adapt', default=False, action='store_true',
                        help='Force to adapt target encoder')

    parser.add_argument('--random_state', type=int, default=42,
                        help="Specify random state")

    parser.add_argument('--max_seq_length', type=int, default=128,
                        help="Specify maximum sequence length")

    parser.add_argument('--temperature', type=int, default=5,
                        help="Specify temperature")

    parser.add_argument('--batch_size', type=int, default=64,
                        help="Specify batch size")

    parser.add_argument('--pre_epochs', type=int, default=3,
                        help="Specify the number of epochs for pretrain")

    parser.add_argument('--pre_log_step', type=int, default=1,
                        help="Specify log step size for pretrain")

    parser.add_argument('--num_epochs', type=int, default=5,
                        help="Specify the number of epochs for adaptation")

    parser.add_argument('--log_step', type=int, default=1,
                        help="Specify log step size for adaptation")

    return parser.parse_args()


def main():
    args = parse_arguments()
    # argument setting
    print("=== Argument Setting ===")
    print("src: " + args.src)
    print("tgt: " + args.tgt)
    print("random_state: " + str(args.random_state))
    print("max_seq_length: " + str(args.max_seq_length))
    print("batch_size: " + str(args.batch_size))
    print("num_epochs_pre: " + str(args.pre_epochs))
    print("pre_log_step: " + str(args.pre_log_step))
    print("num_epochs: " + str(args.num_epochs))
    print("log_step: " + str(args.log_step))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # preprocess data
    print("=== Processing datasets ===")
    if args.src == 'blog':
        reviews, labels = CSV2Array(os.path.join('data', args.src, 'blog.csv'))

    elif args.src == 'airline':
        reviews, labels = CSV2Array(os.path.join('data', args.src, 'airline.csv'))

    else:
        reviews, labels = XML2Array(os.path.join('data', args.src, 'negative.review'),
                                    os.path.join('data', args.src, 'positive.review'))

    src_train_x, src_test_x, src_train_y, src_test_y = train_test_split(reviews, labels,
                                                                        test_size=0.2,
                                                                        stratify=labels,
                                                                        random_state=args.random_state)
    del reviews, labels

    if args.tgt == 'blog':
        tgt_x, tgt_y = CSV2Array(os.path.join('data', args.tgt, 'blog.csv'))

    elif args.tgt == 'airline':
        tgt_x, tgt_y = CSV2Array(os.path.join('data', args.tgt, 'airline.csv'))
    else:
        tgt_x, tgt_y = XML2Array(os.path.join('data', args.tgt, 'negative.review'),
                                 os.path.join('data', args.tgt, 'positive.review'))

    tgt_x, _, tgt_y, _ = train_test_split(tgt_x, tgt_y,
                                          test_size=0.2,
                                          stratify=tgt_y,
                                          random_state=args.random_state)

    src_train_features = convert_examples_to_features(src_train_x, src_train_y, args.max_seq_length, tokenizer)
    src_test_features = convert_examples_to_features(src_test_x, src_test_y, args.max_seq_length, tokenizer)
    tgt_features = convert_examples_to_features(tgt_x, tgt_y, args.max_seq_length, tokenizer)

    # load dataset

    src_data_loader = get_data_loader(src_train_features, args.batch_size)
    src_data_loader_eval = get_data_loader(src_test_features, args.batch_size)
    tgt_data_loader = get_data_loader(tgt_features, args.batch_size)

    # load models
    src_encoder = BERTEncoder()
    src_classifier = BERTClassifier()
    tgt_encoder = BERTEncoder()
    critic = Discriminator()

    src_encoder = init_model(src_encoder,
                             restore=param.src_encoder_restore)
    src_classifier = init_model(src_classifier,
                                restore=param.src_classifier_restore)
    tgt_encoder = init_model(tgt_encoder,
                             restore=param.tgt_encoder_restore)
    critic = init_model(critic,
                        restore=param.d_model_restore)

    # train source model
    print("=== Training classifier for source domain ===")
    if args.pretrain:
        src_encoder, src_classifier = pretrain(
            args, src_encoder, src_classifier, src_data_loader)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    evaluate(src_encoder, src_classifier, src_data_loader)
    evaluate(src_encoder, src_classifier, src_data_loader_eval)
    evaluate(src_encoder, src_classifier, tgt_data_loader)

    for params in src_encoder.parameters():
        params.requires_grad = False

    for params in src_classifier.parameters():
        params.requires_grad = False

    if args.adapt:
        tgt_encoder.load_state_dict(src_encoder.state_dict())

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    if args.adapt:
        tgt_encoder = adapt(args, src_encoder, tgt_encoder, critic,
                            src_classifier, src_data_loader, tgt_data_loader)

    # eval target encoder on lambda0.1 set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    evaluate(src_encoder, src_classifier, tgt_data_loader)
    print(">>> domain adaption <<<")
    evaluate(tgt_encoder, src_classifier, tgt_data_loader)


if __name__ == '__main__':
    main()
