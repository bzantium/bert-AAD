"""Adversarial adaptation to train target encoder."""

import torch
from utils import make_cuda
import torch.nn.functional as F
import torch.nn as nn
import param
import torch.optim as optim
from utils import save_model


def pretrain(args, encoder, classifier, data_loader):
    """Train classifier for source domain."""

    # setup criterion and optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                           lr=param.c_learning_rate)
    CELoss = nn.CrossEntropyLoss()

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    for epoch in range(args.pre_epochs):
        for step, (reviews, mask, labels) in enumerate(data_loader):
            reviews = make_cuda(reviews)
            mask = make_cuda(mask)
            labels = make_cuda(labels)

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for discriminator
            feat = encoder(reviews, mask)
            preds = classifier(feat)
            cls_loss = CELoss(preds, labels)

            # optimize source classifier
            cls_loss.backward()
            optimizer.step()

            # print step info
            if (step + 1) % args.pre_log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: cls_loss=%.4f"
                      % (epoch + 1,
                         args.pre_epochs,
                         step + 1,
                         len(data_loader),
                         cls_loss.item()))

    # save final model
    save_model(args, encoder, param.src_encoder_path)
    save_model(args, classifier, param.src_classifier_path)

    return encoder, classifier


def adapt(args, src_encoder, tgt_encoder, discriminator,
          src_classifier, src_data_loader, tgt_data_train_loader, tgt_data_all_loader):
    """Train encoder for target domain."""

    # set train state for Dropout and BN layers
    src_encoder.eval()
    src_classifier.eval()
    tgt_encoder.train()
    discriminator.train()

    # setup criterion and optimizer
    BCELoss = nn.BCELoss()
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    optimizer_G = optim.Adam(tgt_encoder.parameters(), lr=param.d_learning_rate)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=param.d_learning_rate)
    len_data_loader = min(len(src_data_loader), len(tgt_data_train_loader))

    for epoch in range(args.num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_train_loader))
        for step, ((reviews_src, src_mask, _), (reviews_tgt, tgt_mask, _)) in data_zip:
            reviews_src = make_cuda(reviews_src)
            src_mask = make_cuda(src_mask)

            reviews_tgt = make_cuda(reviews_tgt)
            tgt_mask = make_cuda(tgt_mask)

            # zero gradients for optimizer
            optimizer_D.zero_grad()

            # extract and concat features
            with torch.no_grad():
                feat_src = src_encoder(reviews_src, src_mask)
            feat_src_tgt = tgt_encoder(reviews_src, src_mask)
            feat_tgt = tgt_encoder(reviews_tgt, tgt_mask)
            feat_concat = torch.cat((feat_src_tgt, feat_tgt), 0)

            # predict on discriminator
            pred_concat = discriminator(feat_concat.detach())

            # prepare real and fake label
            label_src = make_cuda(torch.ones(feat_src_tgt.size(0))).unsqueeze(1)
            label_tgt = make_cuda(torch.zeros(feat_tgt.size(0))).unsqueeze(1)
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for discriminator
            dis_loss = BCELoss(pred_concat, label_concat)
            dis_loss.backward()

            for p in discriminator.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)
            # optimize discriminator
            optimizer_D.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            # zero gradients for optimizer
            optimizer_G.zero_grad()
            T = args.temperature

            # predict on discriminator
            pred_tgt = discriminator(feat_tgt)

            # logits for KL-divergence
            with torch.no_grad():
                src_prob = F.softmax(src_classifier(feat_src) / T, dim=-1)
            tgt_prob = F.log_softmax(src_classifier(feat_src_tgt) / T, dim=-1)
            kd_loss = KLDivLoss(tgt_prob, src_prob.detach()) * T * T

            # compute loss for target encoder
            gen_loss = BCELoss(pred_tgt, label_src)
            loss_tgt = args.alpha * gen_loss + args.beta * kd_loss
            loss_tgt.backward()
            torch.nn.utils.clip_grad_norm_(tgt_encoder.parameters(), args.max_grad_norm)
            # optimize target encoder
            optimizer_G.step()

            if (step + 1) % args.log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: "
                      "acc=%.4f g_loss=%.4f d_loss=%.4f kd_loss=%.4f"
                      % (epoch + 1,
                         args.num_epochs,
                         step + 1,
                         len_data_loader,
                         acc.item(),
                         gen_loss.item(),
                         dis_loss.item(),
                         kd_loss.item()))

        evaluate(tgt_encoder, src_classifier, tgt_data_all_loader)

    return tgt_encoder


def evaluate(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (reviews, mask, labels) in data_loader:
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        labels = make_cuda(labels)

        with torch.no_grad():
            feat = encoder(reviews, mask)
            preds = classifier(feat)
        loss += criterion(preds, labels).item()
        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum().item()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = %.4f, Avg Accuracy = %.4f" % (loss, acc))

    return acc
