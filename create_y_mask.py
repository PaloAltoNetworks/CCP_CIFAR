"""
create_y_mask.py

Use this script to create masks over the labels of a dataset. This mask determines
how each sample with be treated during training. A unique y_mask is responsible
for implementing each data scenario/severity level.

The Y mask consists of three possible values for each sample. '1' corresponds
to a known/given label. '0' corresponds to an unknown label. '-1' corresponds to
a sample which will be removed when the dataset is loaded.

@author: Brody Kutt (bkutt@paloaltonetworks.com)
"""

import os
import sys
import utils
import argparse
import numpy as np


def print_dataset_stats(y, y_mask, int2label):
    """
    """
    n_classes = len(int2label)
    all_fps_by_int = {}
    for i in range(-1, n_classes - 1):
        all_fps_by_int[i] = []
    for i in range(len(y)):
        if (y_mask[i] == 1):
            all_fps_by_int[y[i]].append(y[i])
        elif (y_mask[i] == 0):
            all_fps_by_int[-1].append(y[i])
    utils.print_fps_stats(all_fps_by_int, int2label)


def create_y_mask(y, perc_on, int2label, labeled_cls_idxs, unlabeled_cls_idxs,
                  labeled_imbalance_factor, unlabeled_imbalance_factor):
    """
    """
    print('Original stats...')
    print_dataset_stats(y, np.ones_like(y, dtype=np.int8), int2label)
    n_classes = len(int2label)
    y_mask = np.zeros_like(y, dtype=np.int8)
    if (perc_on == 1.0):
        y_mask = np.ones_like(y, dtype=np.int8)
    elif (perc_on > 0.0):
        all_idxs = np.array([i for i in range(len(y))])
        idx_by_y = [[] for i in range(n_classes)]
        for cls_idx in range(n_classes):
            idx_by_y[cls_idx] = all_idxs[y == cls_idx]
        n_per_class = np.array(
            [len(idx_by_y[i]) for i in range(len(idx_by_y))], dtype='int')
        n_keep_per_class = np.ceil(n_per_class * float(perc_on)).astype('int')
        n_keep_per_class = np.maximum(n_keep_per_class,
                                      np.minimum(n_per_class, 1))
        keep_idxs = []
        for group_of_idxs, n_keep in zip(idx_by_y, n_keep_per_class):
            rnd_sample = np.random.choice(group_of_idxs,
                                          size=n_keep,
                                          replace=False)
            keep_idxs += rnd_sample.flatten().tolist()
        y_mask[keep_idxs] += 1
    # Enforce class filters
    all_cls_idxs = set(data['int2label'].keys())
    all_cls_idxs.remove(-1)
    labeled_filter = list(all_cls_idxs - labeled_cls_idxs)
    unlabeled_filter = list(all_cls_idxs - unlabeled_cls_idxs)
    for l_cls_idx in labeled_filter:
        y_mask[np.logical_and(y == l_cls_idx, y_mask == 1)] = -1
    for u_cls_idx in unlabeled_filter:
        y_mask[np.logical_and(y == u_cls_idx, y_mask == 0)] = -1
    # Create imbalance in labeled data
    if (labeled_imbalance_factor < 1.0):
        imbalance_idx_cutoff = len(labeled_cls_idxs) / 2.0
        for l_cls_idx in labeled_cls_idxs:
            if (l_cls_idx >= imbalance_idx_cutoff):
                l_cls_samp_idxs = np.nonzero(
                    np.logical_and(y == l_cls_idx, y_mask == 1))[0]
                np.random.shuffle(l_cls_samp_idxs)
                y_mask[l_cls_samp_idxs[:int(
                    len(l_cls_samp_idxs) *
                    (1.0 - labeled_imbalance_factor))]] = -1
    # Create imbalance in unlabeled data
    if (unlabeled_imbalance_factor < 1.0):
        imbalance_idx_cutoff = len(unlabeled_cls_idxs) / 2.0
        for u_cls_idx in unlabeled_cls_idxs:
            if (u_cls_idx >= imbalance_idx_cutoff):
                u_cls_samp_idxs = np.nonzero(
                    np.logical_and(y == u_cls_idx, y_mask == 0))[0]
                np.random.shuffle(u_cls_samp_idxs)
                y_mask[u_cls_samp_idxs[:int(
                    len(u_cls_samp_idxs) *
                    (1.0 - unlabeled_imbalance_factor))]] = -1
    # Ensure all truly unlabeled samples remain off
    y_mask[y == -1] = 0
    print('\nNew stats...')
    print_dataset_stats(y, y_mask, int2label)
    return y_mask


if __name__ == '__main__':
    # Parse the arguments and options
    parser = argparse.ArgumentParser(
        description=('Create a mask over labels in a dataset.'))
    parser.add_argument(
        '--npz_fp',
        required=True,
        help=('The NPZ containing the data for which you wish to create a '
              'mask.'))
    parser.add_argument(
        '--save_fp',
        required=False,
        default=None,
        help=('The filepath to the NPZ you want to save the mask in '
              '(default: None).'))
    parser.add_argument(
        '--perc_on',
        required=True,
        type=float,
        help=("Percent of each class you'd like to keep labels for (computed "
              "number of samples is rounded). E.g., input 0.6 to mark 40 "
              "percent of the labels to be turned off. There will always be "
              "at least one labeled and one unlabeled sample per class in the "
              "mask unless passed 0.0 or 1.0."))
    parser.add_argument(
        '--labeled_classes',
        required=False,
        default=None,
        help=("Comma-separated integers corresponding to which classes you "
              "want to be included in labeled data. Labeled data from classes "
              "NOT included here will be removed (default: all classes). Use "
              "-1 to remove all labeled data."))
    parser.add_argument(
        '--unlabeled_classes',
        required=False,
        default=None,
        help=
        ("Comma-separated integers corresponding to which classes you "
         "want to be included in unlabeled data. Unlabeled data from classes "
         "NOT included here will be removed (default: all classes). Use "
         "-1 to remove all unlabeled data."))
    parser.add_argument(
        '--labeled_imbalance_factor',
        required=False,
        type=float,
        default=1.0,
        help=('What percent of data to keep in the last half of labeled '
              'classes to create imbalance (default: 1.0).'))
    parser.add_argument(
        '--unlabeled_imbalance_factor',
        required=False,
        type=float,
        default=1.0,
        help=('What percent of data to keep in the last half of unlabeled '
              'classes to create imbalance (default: 1.0).'))
    parser.add_argument('--random_seed',
                        required=False,
                        type=int,
                        default=utils.random_seed(),
                        help=('Random seed for repeatability.'))
    args = parser.parse_args()
    if (args.save_fp is not None
            and not os.path.isdir(os.path.dirname(args.save_fp))):
        os.makedirs(os.path.dirname(args.save_fp))
    np.random.seed(args.random_seed)
    print('-' * 80)
    print('Loading NPZ data...')
    data = utils.load_data_npz(args.npz_fp, load_X=False)
    if (data is None):
        print('ERR: create_y_mask.py: Failed to load: %s' % args.npz_fp)
        sys.exit(-1)  # Exit failure

    if (args.labeled_classes is not None):
        labeled_cls_idxs = set(
            [int(x) for x in args.labeled_classes.split(',')])
        if (-1 in labeled_cls_idxs and len(labeled_cls_idxs) == 1):
            labeled_cls_idxs = set([])
    else:
        labeled_cls_idxs = set(data['int2label'].keys())
        labeled_cls_idxs.remove(-1)
    if (args.unlabeled_classes is not None):
        unlabeled_cls_idxs = set(
            [int(x) for x in args.unlabeled_classes.split(',')])
        if (-1 in unlabeled_cls_idxs and len(unlabeled_cls_idxs) == 1):
            unlabeled_cls_idxs = set([])
    else:
        unlabeled_cls_idxs = set(data['int2label'].keys())
        unlabeled_cls_idxs.remove(-1)

    print('--> Loaded %d samples!' % len(data['y']))
    print('Computing y mask...')
    y_mask = create_y_mask(data['y'], args.perc_on, data['int2label'],
                           labeled_cls_idxs, unlabeled_cls_idxs,
                           args.labeled_imbalance_factor,
                           args.unlabeled_imbalance_factor)
    print('--> Computed y_mask of shape %s with %s labels left on' %
          (str(y_mask.shape), str(np.sum(y_mask > 0))))
    if (args.save_fp is not None):
        print('Saving...')
        np.savez_compressed(args.save_fp, y_mask=y_mask)
        print('--> Saved to: %s' % str(args.save_fp))
    print('\nExiting...')
    print('-' * 80)
