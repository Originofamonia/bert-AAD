"""Main script for bert-AAD source target free"""

import param
from train import pretrain, evaluate, src_gmm, tgt_gmm, src_tgt_free_adapt
from model import (BertEncoder, DistilBertEncoder, DistilRobertaEncoder,
                   BertClassifier, Discriminator, RobertaEncoder, RobertaClassifier)
from utils import XML2Array, CSV2Array, convert_examples_to_features, \
    roberta_convert_examples_to_features, get_data_loader, init_model
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, RobertaTokenizer
import torch
import torch.nn as nn
import os
import random
import argparse


def parse_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")
    parser.add_argument('--src', type=str, default="books",
                        choices=["books", "dvd", "electronics", "kitchen", "blog", "airline", "imdb"],
                        help="Specify src dataset")
    parser.add_argument('--tgt', type=str, default="dvd",
                        choices=["books", "dvd", "electronics", "kitchen", "blog", "airline", "imdb"],
                        help="Specify tgt dataset")
    parser.add_argument('--pretrain', default=True, action='store_true',
                        help='Force to pretrain source encoder/classifier')
    parser.add_argument('--adapt', default=True, action='store_true',
                        help='Force to adapt target encoder')
    parser.add_argument('--seed', type=int, default=42,
                        help="Specify random state")
    parser.add_argument('--train_seed', type=int, default=42,
                        help="Specify random state")
    parser.add_argument('--load', default=False, action='store_true',
                        help="Load saved model")
    parser.add_argument('--model', type=str, default="bert",
                        choices=["bert", "distilbert", "roberta", "distilroberta"],
                        help="Specify model type")
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help="Specify maximum sequence length")
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Specify adversarial weight")
    parser.add_argument('--beta', type=float, default=1.0,
                        help="Specify KD loss weight")
    parser.add_argument('--temperature', type=int, default=20,
                        help="Specify temperature")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--clip_value", type=float, default=1e-2,
                        help="lower and upper clip value for disc. weights")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Specify batch size")
    parser.add_argument('--pre_epochs', type=int, default=3,
                        help="Specify the number of epochs for pretrain")
    parser.add_argument('--pre_log_step', type=int, default=1,
                        help="Specify log step size for pretrain")
    parser.add_argument('--num_epochs', type=int, default=3,
                        help="Specify the number of epochs for adaptation")
    parser.add_argument('--num_class', type=int, default=2,
                        help="Specify the number of classes in dataset")
    parser.add_argument('--log_step', type=int, default=5,
                        help="Specify log step size for adaptation")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def get_dataloader(args, tokenizer):
    print("=== Processing datasets ===")
    if args.src in ['blog', 'airline', 'imdb']:
        src_x, src_y = CSV2Array(os.path.join('data', args.src, args.src + '.csv'))
    else:
        src_x, src_y = XML2Array(os.path.join('data', args.src, 'negative.review'),
                                 os.path.join('data', args.src, 'positive.review'))
    src_x, src_test_x, src_y, src_test_y = train_test_split(src_x, src_y,
                                                            test_size=0.2,
                                                            stratify=src_y,
                                                            random_state=args.seed)
    if args.tgt in ['blog', 'airline', 'imdb']:
        tgt_x, tgt_y = CSV2Array(os.path.join('data', args.tgt, args.tgt + '.csv'))
    else:
        tgt_x, tgt_y = XML2Array(os.path.join('data', args.tgt, 'negative.review'),
                                 os.path.join('data', args.tgt, 'positive.review'))
    tgt_train_x, tgt_test_y, tgt_train_y, tgt_test_y = train_test_split(tgt_x, tgt_y,
                                                                        test_size=0.2,
                                                                        stratify=tgt_y,
                                                                        random_state=args.seed)
    if args.model in ['roberta', 'distilroberta']:
        src_features = roberta_convert_examples_to_features(src_x, src_y, args.max_seq_length, tokenizer)
        src_test_features = roberta_convert_examples_to_features(src_test_x, src_test_y, args.max_seq_length, tokenizer)
        tgt_features = roberta_convert_examples_to_features(tgt_x, tgt_y, args.max_seq_length, tokenizer)
        tgt_train_features = roberta_convert_examples_to_features(tgt_train_x, tgt_train_y, args.max_seq_length,
                                                                  tokenizer)
    else:
        src_features = convert_examples_to_features(src_x, src_y, args.max_seq_length, tokenizer)
        src_test_features = convert_examples_to_features(src_test_x, src_test_y, args.max_seq_length, tokenizer)
        tgt_features = convert_examples_to_features(tgt_x, tgt_y, args.max_seq_length, tokenizer)
        tgt_train_features = convert_examples_to_features(tgt_train_x, tgt_train_y, args.max_seq_length, tokenizer)
    # load dataset
    src_loader = get_data_loader(src_features, args.batch_size)
    src_eval_loader = get_data_loader(src_test_features, args.batch_size)
    tgt_train_loader = get_data_loader(tgt_train_features, args.batch_size)
    tgt_all_loader = get_data_loader(tgt_features, args.batch_size)
    return src_eval_loader, src_loader, tgt_all_loader, tgt_train_loader


def main():
    args = parse_arguments()
    set_seed(args.train_seed)

    if args.model in ['roberta', 'distilroberta']:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # preprocess data
    src_eval_loader, src_loader, tgt_all_loader, tgt_train_loader = get_dataloader(args, tokenizer)

    # load models
    if args.model == 'bert':
        src_encoder = BertEncoder()
        tgt_encoder = BertEncoder()
        src_classifier = BertClassifier()
    elif args.model == 'distilbert':
        src_encoder = DistilBertEncoder()
        tgt_encoder = DistilBertEncoder()
        src_classifier = BertClassifier()
    elif args.model == 'roberta':
        src_encoder = RobertaEncoder()
        tgt_encoder = RobertaEncoder()
        src_classifier = RobertaClassifier()
    else:
        src_encoder = DistilRobertaEncoder()
        tgt_encoder = DistilRobertaEncoder()
        src_classifier = RobertaClassifier()
    discriminator = Discriminator()

    if args.load:
        src_encoder = init_model(args, src_encoder, restore=param.src_encoder_path)
        src_classifier = init_model(args, src_classifier, restore=param.src_classifier_path)
        tgt_encoder = init_model(args, tgt_encoder, restore=param.tgt_encoder_path)
        discriminator = init_model(args, discriminator, restore=param.d_model_path)
    else:
        src_encoder = init_model(args, src_encoder)
        src_classifier = init_model(args, src_classifier)
        tgt_encoder = init_model(args, tgt_encoder)
        discriminator = init_model(args, discriminator)

    # parallel models
    if torch.cuda.device_count() > 1:
        print('Let\'s use {} GPUs!'.format(torch.cuda.device_count()))
        src_encoder = nn.DataParallel(src_encoder)
        src_classifier = nn.DataParallel(src_classifier)
        tgt_encoder = nn.DataParallel(tgt_encoder)
        discriminator = nn.DataParallel(discriminator)

    # train source model
    print("=== Training classifier for source domain ===")
    if args.pretrain:
        src_encoder, src_classifier = pretrain(
            args, src_encoder, src_classifier, src_loader)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    evaluate(src_encoder, src_classifier, src_loader)
    evaluate(src_encoder, src_classifier, src_eval_loader)
    evaluate(src_encoder, src_classifier, tgt_all_loader)

    for params in src_encoder.parameters():
        params.requires_grad = False

    for params in src_classifier.parameters():
        params.requires_grad = False

    s_feature_dict = src_gmm(args, src_encoder, src_classifier, src_loader)
    t_feature_dict = tgt_gmm(args, tgt_encoder, tgt_all_loader, 1)

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    if args.adapt:
        tgt_encoder.load_state_dict(src_encoder.state_dict())
        tgt_encoder = src_tgt_free_adapt(args, src_encoder, tgt_encoder, discriminator, src_classifier,
                             s_feature_dict, t_feature_dict, src_loader, tgt_train_loader, tgt_all_loader)

    # argument setting
    print(f"model_type: {args.model}; max_seq_len: {args.max_seq_length}; batch_size: {args.batch_size}; "
          f"pre_epochs: {args.pre_epochs}; num_epochs: {args.num_epochs}; adv weight: {args.alpha}; "
          f"KD weight: {args.beta}; temperature: {args.temperature}; src: {args.src}; tgt: {args.tgt}")

    # eval target encoder on lambda0.1 set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    evaluate(src_encoder, src_classifier, tgt_all_loader)
    print(">>> domain adaption <<<")
    evaluate(tgt_encoder, src_classifier, tgt_all_loader)


if __name__ == '__main__':
    main()
