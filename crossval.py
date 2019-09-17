"""Cross Validation script for ADDA."""

from core import eval_tgt, train_src, train_tgt
from models import BERTEncoder, BERTClassifier, Discriminator
from utils import XML2Array, TSV2Array, review2seq, \
    get_data_loader
from sklearn.model_selection import StratifiedKFold
import numpy as np
import xlsxwriter
import os
import argparse
import torch

if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")

    parser.add_argument('--src', type=str, default="books", choices=["books", "dvd", "electronics", "kitchen"],
                        help="Specify src dataset")

    parser.add_argument('--tgt', type=str, default="dvd", choices=["books", "dvd", "electronics", "kitchen", "blog"],
                        help="Specify tgt dataset")

    parser.add_argument('--seqlen', type=int, default=100,
                        help="Specify maximum sequence length")

    parser.add_argument('--n_splits', type=int, default=5,
                        help="Specify the number of splits")

    parser.add_argument('--batch_size', type=int, default=64,
                        help="Specify batch size")

    parser.add_argument('--num_epochs_pre', type=int, default=5,
                        help="Specify the number of epochs for pretrain")

    parser.add_argument('--log_step_pre', type=int, default=1,
                        help="Specify log step size for pretrain")

    parser.add_argument('--eval_step_pre', type=int, default=1,
                        help="Specify eval step size for pretrain")

    parser.add_argument('--num_epochs', type=int, default=5,
                        help="Specify the number of epochs for adaptation")

    parser.add_argument('--log_step', type=int, default=1,
                        help="Specify log step size for adaptation")

    parser.add_argument('--eval_step', type=int, default=1,
                        help="Specify eval step size for adaptation")

    args = parser.parse_args()

    # argument setting
    print("=== Argument Setting ===")
    print("src: " + args.src)
    print("tgt: " + args.tgt)
    print("seqlen: " + str(args.seqlen))
    print("n_splits: " + str(args.n_splits))
    print("batch_size: " + str(args.batch_size))
    print("num_epochs_pre: " + str(args.num_epochs_pre))
    print("log_step_pre: " + str(args.log_step_pre))
    print("eval_step_pre: " + str(args.eval_step_pre))
    print("num_epochs: " + str(args.num_epochs))
    print("log_step: " + str(args.log_step))
    print("eval_step: " + str(args.eval_step))

    # preprocess data
    print("=== Processing datasets ===")
    src_reviews, src_labels = XML2Array(os.path.join('data', args.src, 'negative.review'),
                                        os.path.join('data', args.src, 'positive.review'))

    if args.tgt == 'blog':
        tgt_X, tgt_Y = TSV2Array(os.path.join('data', args.tgt, 'blog.review'))

    else:
        tgt_X, tgt_Y = XML2Array(os.path.join('data', args.tgt, 'negative.review'),
                                 os.path.join('data', args.tgt, 'positive.review'))

    tgt_X = review2seq(tgt_X)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True)
    if not os.path.exists('experiments'):
        os.mkdir('experiments')
    workbook = xlsxwriter.Workbook(os.path.join('experiments',
                                                args.src + '-' +
                                                args.tgt + '-' +
                                                'results.xlsx'))
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, 'fold')
    worksheet.write(0, 1, 'src-src')
    worksheet.write(0, 2, 'src-tgt')
    worksheet.write(0, 3, 'tgt-tgt')
    src_src_stack = []
    src_tgt_stack = []
    tgt_tgt_stack = []

    for fold_index, (train_index, test_index) in enumerate(skf.split(src_reviews, src_labels)):
        print('[Cross Validation Fold #%.1d]' % (fold_index + 1))
        src_X_train = review2seq(src_reviews[train_index])
        src_X_test = review2seq(src_reviews[test_index])
        src_Y_train = src_labels[train_index]
        src_Y_test = src_labels[test_index]

        # load dataset
        src_data_loader = get_data_loader(src_X_train, src_Y_train, args.batch_size, args.seqlen)
        src_data_loader_eval = get_data_loader(src_X_test, src_Y_test, args.batch_size, args.seqlen)
        tgt_data_loader = get_data_loader(tgt_X, tgt_Y, args.batch_size, args.seqlen)

        # load models
        src_encoder = BERTEncoder()
        src_classifier = BERTClassifier()
        tgt_encoder = BERTEncoder()
        critic = Discriminator()

        #if torch.cuda.device_count() > 1:
        #    src_encoder = torch.nn.DataParallel(src_encoder)
        #    src_classifier = torch.nn.DataParallel(src_classifier)
        #    tgt_encoder = torch.nn.DataParallel(tgt_encoder)
        #    critic = torch.nn.DataParallel(critic)

        if torch.cuda.is_available():
            src_encoder.cuda()
            src_classifier.cuda()
            tgt_encoder.cuda()
            critic.cuda()

        # freeze source encoder params
        # if torch.cuda.device_count() > 1:
        #     for params in src_encoder.module.encoder.embeddings.parameters():
        #         params.requires_grad = False
        # else:
        #     for params in src_encoder.encoder.embeddings.parameters():
        #         params.requires_grad = False

        # train source model
        print("=== Training classifier for source domain ===")
        src_encoder, src_classifier = train_src(
            args, src_encoder, src_classifier, src_data_loader)

        # eval source model
        print("=== Evaluating classifier for source domain ===")
        eval_tgt(src_encoder, src_classifier, src_data_loader)
        src_src = eval_tgt(src_encoder, src_classifier, src_data_loader_eval)

        for params in src_encoder.parameters():
            params.requires_grad = False

        for params in src_classifier.parameters():
            params.requires_grad = False

        tgt_encoder.load_state_dict(src_encoder.state_dict())

        # freeze target encoder params
        #for params in tgt_encoder.parameters():
        #    params.requires_grad = False
        #for params in tgt_encoder.encoder.embeddings.parameters():
        #    params.requires_grad = True
        #for params in tgt_encoder.encoder.pooler.parameters():
        #    params.requires_grad = True
        #for params in tgt_encoder.encoder.encoder.layer[11].parameters():
        #    params.requires_grad = True

        # train target encoder by GAN
        print("=== Training encoder for target domain ===")
        tgt_encoder = train_tgt(args, src_encoder, tgt_encoder, critic,
                                src_classifier, src_data_loader, tgt_data_loader)

        # eval target encoder on lambda0.1 set of target dataset
        print("=== Evaluating classifier for encoded target domain ===")
        print(">>> source only <<<")
        src_tgt = eval_tgt(src_encoder, src_classifier, tgt_data_loader)
        print(">>> domain adaption <<<")
        tgt_tgt = eval_tgt(tgt_encoder, src_classifier, tgt_data_loader)

        worksheet.write(fold_index + 1, 0, fold_index + 1)
        worksheet.write(fold_index + 1, 1, src_src)
        worksheet.write(fold_index + 1, 2, src_tgt)
        worksheet.write(fold_index + 1, 3, tgt_tgt)
        src_src_stack.append(src_src)
        src_tgt_stack.append(src_tgt)
        tgt_tgt_stack.append(tgt_tgt)
        print()

    worksheet.write(args.n_splits + 1, 0, 'mean')
    worksheet.write(args.n_splits + 1, 1, np.mean(src_src_stack))
    worksheet.write(args.n_splits + 1, 2, np.mean(src_tgt_stack))
    worksheet.write(args.n_splits + 1, 3, np.mean(tgt_tgt_stack))
    worksheet.write(args.n_splits + 2, 0, 'std')
    worksheet.write(args.n_splits + 2, 1, np.std(src_src_stack))
    worksheet.write(args.n_splits + 2, 2, np.std(src_tgt_stack))
    worksheet.write(args.n_splits + 2, 3, np.std(tgt_tgt_stack))
    workbook.close()
