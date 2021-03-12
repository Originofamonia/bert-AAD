"""Adversarial adaptation to train target encoder."""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import param
import torch.optim as optim
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from utils import make_cuda, save_model


def pretrain(args, encoder, classifier, data_loader):
    """Train classifier for source domain."""

    # setup criterion and optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                           lr=param.c_lr)
    CELoss = nn.CrossEntropyLoss()

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    for epoch in range(args.pre_epochs):
        pbar = tqdm(data_loader)
        for step, (reviews, mask, labels) in enumerate(pbar):
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
            if step % args.pre_log_step == 0:
                desc = f"Epoch [{epoch}/{args.pre_epochs}] Step [{step}/{len(data_loader)}]: " \
                       f"c_loss={cls_loss.item():.4f} "
                pbar.set_description(desc=desc)

    # save final model
    # save_model(args, encoder, param.src_encoder_path)
    # save_model(args, classifier, param.src_classifier_path)

    return encoder, classifier


def adapt(args, src_encoder, tgt_encoder, discriminator,
          src_classifier, src_data_loader, tgt_data_train_loader, tgt_data_all_loader):
    """Train tgt_encoder using bert-AAD"""

    # set train state for Dropout and BN layers
    src_encoder.eval()
    src_classifier.eval()
    tgt_encoder.train()
    discriminator.train()

    # setup criterion and optimizer
    bce_loss = nn.BCELoss()
    kl_div_loss = nn.KLDivLoss(reduction='batchmean')
    optimizer_g = optim.Adam(tgt_encoder.parameters(), lr=param.d_lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=param.d_lr)
    len_data_loader = min(len(src_data_loader), len(tgt_data_train_loader))

    for epoch in range(args.num_epochs):
        # zip source and target data pair
        pbar = tqdm(zip(src_data_loader, tgt_data_train_loader))
        for step, ((reviews_src, src_mask, _), (reviews_tgt, tgt_mask, _)) in enumerate(pbar):
            reviews_src = make_cuda(reviews_src)
            src_mask = make_cuda(src_mask)

            reviews_tgt = make_cuda(reviews_tgt)
            tgt_mask = make_cuda(tgt_mask)

            # zero gradients for optimizer
            optimizer_d.zero_grad()

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

            # domain discriminator loss of discriminator
            dis_loss = bce_loss(pred_concat, label_concat)
            dis_loss.backward()
            # increase the clip_value from 0.01 to 0.1 is bad
            for p in discriminator.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)
            # optimize discriminator
            optimizer_d.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            # zero gradients for optimizer
            optimizer_g.zero_grad()
            t = args.temperature

            # predict on discriminator
            pred_tgt = discriminator(feat_tgt)

            # logits for KL-divergence
            with torch.no_grad():
                src_prob = F.softmax(src_classifier(feat_src) / t, dim=-1)
            tgt_prob = F.log_softmax(src_classifier(feat_src_tgt) / t, dim=-1)
            kd_loss = kl_div_loss(tgt_prob, src_prob.detach()) * t * t

            # compute loss for target encoder
            gen_loss = bce_loss(pred_tgt, label_src)  # domain loss of tgt encoder
            loss_tgt = args.alpha * gen_loss + args.beta * kd_loss
            loss_tgt.backward()
            torch.nn.utils.clip_grad_norm_(tgt_encoder.parameters(), args.max_grad_norm)
            # optimize target encoder
            optimizer_g.step()

            if step % args.log_step == 0:
                desc = f"Epoch [{epoch}/{args.num_epochs}] Step [{step}/{len_data_loader}]: acc={acc.item():.4f} " \
                       f"g_loss={gen_loss.item():.4f} d_loss={dis_loss.item():.4f} kd_loss={kd_loss.item():.4f}"
                pbar.set_description(desc=desc)

        evaluate(tgt_encoder, src_classifier, tgt_data_all_loader)

    return tgt_encoder


def adda_adapt(args, src_encoder, tgt_encoder, critic,
               src_data_loader, tgt_data_loader):
    """
    Adapt tgt encoder by ADDA
    """
    # set train state for Dropout and BN layers
    src_encoder.eval()
    tgt_encoder.train()
    critic.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(), lr=param.d_lr)
                               # betas=(args.beta1, args.beta2))
    optimizer_critic = optim.Adam(critic.parameters(), lr=param.d_lr)
                                  # betas=(args.beta1, args.beta2))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    for epoch in range(args.num_epochs):
        # zip source and target data pair
        pbar = tqdm(zip(src_data_loader, tgt_data_loader))
        for step, ((reviews_src, src_mask, _), (reviews_tgt, tgt_mask, _)) in enumerate(pbar):

            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(reviews_src, src_mask)
            feat_tgt = tgt_encoder(reviews_tgt, tgt_mask)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = torch.squeeze(critic(feat_concat.detach()))

            # prepare real and fake label
            label_src = make_cuda(torch.ones(feat_src.size(0)).long())
            label_tgt = make_cuda(torch.zeros(feat_tgt.size(0)).long())
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for critic
            print('pred_concat:', pred_concat.size())
            print('label_concat:', label_concat.size())
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # optimize critic
            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            # zero gradients for optimizer
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_encoder(reviews_tgt, tgt_mask)

            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            label_tgt = make_cuda(torch.ones(feat_tgt.size(0)).long())

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()

            if (step + 1) % args.log_step == 0:
                desc = "Epoch [{}/{}] Step [{}/{}]: t_loss={:.4f} c_loss={:.4f} " \
                       "acc={:.4f}".format(epoch,
                                           args.num_epochs,
                                           step,
                                           len_data_loader,
                                           loss_tgt.item(),
                                           loss_critic.item(),
                                           acc.item())
                pbar.set_description(desc=desc)

    # torch.save(critic.state_dict(), os.path.join(
    #     args.model_root, "ADDA-critic.pt"))
    # torch.save(tgt_encoder.state_dict(), os.path.join(
    #     args.model_root, "ADDA-target-encoder.pt"))
    return tgt_encoder


def src_tgt_free_adapt(args, src_encoder, tgt_encoder, discriminator, src_classifier, s_feature_dict, t_feature_dict,
                       src_data_loader, tgt_data_train_loader, tgt_data_all_loader):
    """Train encoder for target domain w src tgt data free"""

    # set train state for Dropout and BN layers
    src_encoder.eval()
    src_classifier.eval()
    tgt_encoder.train()
    discriminator.train()

    # setup criterion and optimizer
    bce_loss = nn.BCELoss()
    kl_div_loss = nn.KLDivLoss(reduction='batchmean')
    optimizer_g = optim.Adam(tgt_encoder.parameters(), lr=param.d_lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=param.d_lr)
    len_data_loader = min(len(src_data_loader), len(tgt_data_train_loader))

    for epoch in range(args.num_epochs):
        # zip source and target data pair
        pbar = tqdm(range(len_data_loader))
        for step in enumerate(pbar):
            feat_src = np.zeros([args.batch_size, param.hidden_size])
            source_logits = np.random.randint(0, args.num_class, args.batch_size)

            feat_tgt = np.zeros([args.batch_size, param.hidden_size])
            for b in range(args.batch_size):
                feat_src[b, :] = s_feature_dict[source_logits[b], np.random.randint(0, param.num_resample), :]
                feat_tgt[b, :] = t_feature_dict[np.random.randint(0, param.num_resample), :]
            feat_src = torch.tensor(feat_src, dtype=torch.float32).cuda()
            feat_tgt = torch.tensor(feat_tgt, dtype=torch.float32).cuda()

            # zero gradients for optimizer
            optimizer_d.zero_grad()

            # extract and concat features
            # with torch.no_grad():
            #     feat_src = src_encoder(reviews_src, src_mask)
            # feat_src_tgt = tgt_encoder(reviews_src, src_mask)
            # not sure what src_mask is, so haven't implement feat_src_tgt
            # try ADDA adapt first, BERT-ADDA
            feat_concat = torch.cat((feat_src_tgt, feat_tgt), 0)

            # predict on discriminator
            pred_concat = discriminator(feat_concat.detach())

            # prepare real and fake label
            label_src = make_cuda(torch.ones(feat_src_tgt.size(0))).unsqueeze(1)
            label_tgt = make_cuda(torch.zeros(feat_tgt.size(0))).unsqueeze(1)
            label_concat = torch.cat((label_src, label_tgt), 0)

            # domain discriminator loss of discriminator
            dis_loss = bce_loss(pred_concat, label_concat)
            dis_loss.backward()
            # increase the clip_value from 0.01 to 0.1 is bad
            for p in discriminator.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)
            # optimize discriminator
            optimizer_d.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            # zero gradients for optimizer
            optimizer_g.zero_grad()
            t = args.temperature

            # predict on discriminator
            pred_tgt = discriminator(feat_tgt)

            # logits for KL-divergence
            with torch.no_grad():
                src_prob = F.softmax(src_classifier(feat_src) / t, dim=-1)
            tgt_prob = F.log_softmax(src_classifier(feat_src_tgt) / t, dim=-1)
            kd_loss = kl_div_loss(tgt_prob, src_prob.detach()) * t * t

            # compute loss for target encoder
            gen_loss = bce_loss(pred_tgt, label_src)  # domain loss of tgt encoder
            loss_tgt = args.alpha * gen_loss + args.beta * kd_loss
            loss_tgt.backward()
            torch.nn.utils.clip_grad_norm_(tgt_encoder.parameters(), args.max_grad_norm)
            # optimize target encoder
            optimizer_g.step()

            if step % args.log_step == 0:
                desc = f"Epoch [{epoch}/{args.num_epochs}] Step [{step}/{len_data_loader}]: acc={acc.item():.4f} " \
                       f"g_loss={gen_loss.item():.4f} d_loss={dis_loss.item():.4f} kd_loss={kd_loss.item():.4f}"
                pbar.set_description(desc=desc)

        evaluate(tgt_encoder, src_classifier, tgt_data_all_loader)

    return tgt_encoder


def src_gmm(args, src_encoder, src_classifier, src_data_loader):
    """
    calculate source feature centroids; maybe used for get s_feature_dict
    based on source_target_free.py
    """
    src_encoder.eval()
    src_classifier.eval()
    cov = np.zeros([2, 256])  # num_classes, feature_dim
    mean = np.zeros([2, 256])  # change later
    # s_feature_dict used when adapting
    s_feature_dict = np.zeros([2, param.num_resample, param.hidden_size])  # num_classes, num_samples, feature_dim
    for i in range(2):  # num_classes
        x = []
        pbar = tqdm(src_data_loader)
        for j, (reviews, masks, labels) in enumerate(pbar):
            reviews = make_cuda(reviews)
            masks = make_cuda(masks)
            for b in range(reviews.size(0)):
                if labels[b] == i:
                    with torch.no_grad():
                        review = reviews[b:b + 1]
                        mask = masks[b:b + 1]
                        s_feature = src_encoder(review, mask)
                        s_feature = s_feature.cpu().numpy()
                        if not x:
                            x = s_feature.T
                        else:
                            x = np.concatenate([x, s_feature.T], 1)

        cov[i, :] = np.var(x, 1)
        mean[i, :] = np.mean(x, 1)
        # probably not needed here
        s_feature_dict[i, :, :] = np.random.multivariate_normal(mean[i, :], np.diag(cov[i, :]), param.num_resample)

    np.savez(os.path.join(param.model_root, 'src_mean_cov'), mean, cov)
    return s_feature_dict


def tgt_gmm(args, tgt_encoder, tgt_data_all_loader, num_cluster):
    """
    build target GMM and resample from it, used in adapt
    based on source_target_free.py
    """
    tgt_encoder.eval()
    feature_s = torch.Tensor().float().cuda()
    k = 0
    pbar = tqdm(tgt_data_all_loader)
    for j, (review, mask, _) in enumerate(pbar):
        review = make_cuda(review)
        mask = make_cuda(mask)
        batch_feature = tgt_encoder(review, mask)
        print(batch_feature.size())
        feature_s = torch.cat((feature_s, batch_feature), 0)
        k += 1

        feature_s = feature_s.cpu().numpy()
        try:
            # num_class * num_cluster, not sure why num_cluster is not 1
            gmm = GaussianMixture(n_components=2 * num_cluster,
                                  ).fit(feature_s)
        except Exception as e:
            print(e)

        tgt_mean = gmm.mean_
        tgt_var = gmm.covariance_
        print(gmm.converged_)
        t_feature_dict = np.zeros([param.num_resample, param.hidden_size])  # param.num_resample samples, feature_dim
        p = gmm.weights_
        p[len(p) - 1] += 1 - np.sum(p)
        decrete = np.random.multinomial(param.num_resample, p, 1)
        k = 0
        for i in range(2 * num_cluster):
            t_feature_dict[k: k + decrete[0, i], :] = np.random.multivariate_normal(tgt_mean[i, :], tgt_var[i, :, :],
                                                                                    decrete[0, i])
            k += decrete[0, i]

        return t_feature_dict


def dann_adapt(args, src_encoder, tgt_encoder, discriminator, src_classifier,
               src_data_loader, tgt_data_train_loader, tgt_data_all_loader):
    """
    haven't implemented dann
    """

    # set train state for Dropout and BN layers
    src_encoder.eval()
    src_classifier.eval()
    tgt_encoder.train()
    discriminator.train()

    # setup criterion and optimizer
    bce_loss = nn.BCELoss()
    kl_div_loss = nn.KLDivLoss(reduction='batchmean')
    optimizer_g = optim.Adam(tgt_encoder.parameters(), lr=param.d_lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=param.d_lr)
    len_data_loader = min(len(src_data_loader), len(tgt_data_train_loader))

    for epoch in range(args.num_epochs):
        # zip source and target data pair
        pbar = tqdm(zip(src_data_loader, tgt_data_train_loader))
        for step, ((reviews_src, src_mask, _), (reviews_tgt, tgt_mask, _)) in enumerate(pbar):
            reviews_src = make_cuda(reviews_src)
            src_mask = make_cuda(src_mask)

            reviews_tgt = make_cuda(reviews_tgt)
            tgt_mask = make_cuda(tgt_mask)

            # zero gradients for optimizer
            optimizer_d.zero_grad()

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

            # domain discriminator loss of discriminator
            dis_loss = bce_loss(pred_concat, label_concat)
            dis_loss.backward()
            # increase the clip_value from 0.01 to 0.1 is bad
            for p in discriminator.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)
            # optimize discriminator
            optimizer_d.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            # zero gradients for optimizer
            optimizer_g.zero_grad()
            t = args.temperature

            # predict on discriminator
            pred_tgt = discriminator(feat_tgt)

            # logits for KL-divergence
            with torch.no_grad():
                src_prob = F.softmax(src_classifier(feat_src) / t, dim=-1)
            tgt_prob = F.log_softmax(src_classifier(feat_src_tgt) / t, dim=-1)
            kd_loss = kl_div_loss(tgt_prob, src_prob.detach()) * t * t

            # compute loss for target encoder
            gen_loss = bce_loss(pred_tgt, label_src)  # domain loss of tgt encoder
            loss_tgt = args.alpha * gen_loss + args.beta * kd_loss
            loss_tgt.backward()
            torch.nn.utils.clip_grad_norm_(tgt_encoder.parameters(), args.max_grad_norm)
            # optimize target encoder
            optimizer_g.step()

            if step % args.log_step == 0:
                desc = f"Epoch [{epoch}/{args.num_epochs}] Step [{step}/{len_data_loader}]: acc={acc.item():.4f} " \
                       f"g_loss={gen_loss.item():.4f} d_loss={dis_loss.item():.4f} kd_loss={kd_loss.item():.4f}"
                pbar.set_description(desc=desc)

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
