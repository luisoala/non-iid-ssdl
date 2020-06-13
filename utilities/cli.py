# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.`

import re
import argparse
import logging



DEFAULT_PATH = "/media/Data/saul/Datasets/Inbreast_folder_per_class"

NUMBER_LABELED_OBSERVATIONS = 150
BATCH_SIZE = 8

LAMBDA_DEFAULT = 25
# Modified from
K_DEFAULT = 2
T_DEFAULT = 0.25
ALPHA_DEFAULT = 0.75
LR_DEFAULT = 2e-6
WEIGHT_DECAY_DEFAULT = 1e-4
DEFAULT_RESULTS_FILE = "Stats.csv"
LOG = logging.getLogger('main')

__all__ = ['parse_cmd_args', 'parse_dict_args']


def create_parser():
    parser = argparse.ArgumentParser(description='PyTorch Mix Match Training')
    
    parser.add_argument('--exp_creator', type=str, default="No",
                        help='Whether to use script to create experiment')

    parser.add_argument('--dataset', type=str, default="No data set specified",
                        help='Name of the dataset used for the experiments')

    parser.add_argument('--path_labeled', type=str, default=DEFAULT_PATH,
                        help='The directory with the labeled data')
    parser.add_argument('--path_unlabeled', type=str, default="",
                        help='The directory with the unlabeled data')

    parser.add_argument('--results_file_name', type=str, default=DEFAULT_RESULTS_FILE,
                        help='Name of results file')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-b', '--batch_size', default=BATCH_SIZE, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=LR_DEFAULT, type=float,
                        metavar='LR', help='learning rate')

    parser.add_argument('--weight_decay', default=WEIGHT_DECAY_DEFAULT, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--K_transforms', default=K_DEFAULT, type=int, metavar='K', help = 'Number of simple transformations')
    parser.add_argument('--T_sharpening', default=T_DEFAULT, type=float, metavar='T',
                        help='Sharpening coefficient')

    parser.add_argument('--alpha_mix', default=ALPHA_DEFAULT, type=float, metavar='A',
                        help='Mix alpha coefficient')

    parser.add_argument('--mode', default="fully_supervised", type=str,
                        help='Modes: fully_supervised, partial_supervised, ssdl')
    #int -1 no bal, 5 bal
    parser.add_argument('--balanced', default=-1, type=int,
                        help='Balance the cross entropy loss')

    parser.add_argument('--lambda_unsupervised', default=LAMBDA_DEFAULT, type=float,
                        help='Unsupervised learning coefficient')

    parser.add_argument('--number_labeled', default=NUMBER_LABELED_OBSERVATIONS, type=int, metavar='A',
                        help='Number of labeled observations')

    parser.add_argument('--model', default="densenet", type=str, metavar='A',
                        help='Model to use')

    parser.add_argument('--num_classes', default=5, type=int,
                        help='Number of classes')

    parser.add_argument('--size_image', default=32, type=int,
                        help='Image input size')

    parser.add_argument('--log_folder', type=str, default="logs",
                        help='logging folder')



    parser.add_argument('--norm_stats', type=str, default="MNIST",
                        help='mean std values for dataset: MNIST and COVID provided')

    parser.add_argument('--save_weights', default=False, type=bool,
                        help='Save the weights of the last model found in training')

    parser.add_argument('--weights_path_name', type=str, default="",
                        help='path to store weights')

    parser.add_argument('--rampup_coefficient', default=3000, type=int,
                        help='Rampup coefficient for the unsupervised term')

    return parser


def parse_commandline_args():
    return create_parser().parse_args()


def parse_dict_args(**kwargs):
    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))

    logging.info("Using these command line args: %s", " ".join(cmdline_args))

    return create_parser().parse_args(cmdline_args)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2epochs(v):
    try:
        if len(v) == 0:
            epochs = []
        else:
            epochs = [int(string) for string in v.split(",")]
    except:
        raise argparse.ArgumentTypeError(
            'Expected comma-separated list of integers, got "{}"'.format(v))
    if not all(0 < epoch1 < epoch2 for epoch1, epoch2 in zip(epochs[:-1], epochs[1:])):
        raise argparse.ArgumentTypeError(
            'Expected the epochs to be listed in increasing order')
    return epochs
