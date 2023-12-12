"""
get_processed_dataset.py

Use this script to create an NPZ archive of a perturbed experimental
copy of a CIFAR dataset that implements a specific data scenario. You can
then migrate this NPZ archive for use with other algorithms.

The saved NPZ archive consists of the following:

    X_l:            ndarray, shape=<n, height, width, channels>, dtype=uint8, the labeled training image data with range [0, 255].
    y_l:            ndarray, shape=<n>, dtype=int8, the int class of each training image that corresponds to a labeled sample.
    basenames_l:    ndarray, shape=<n>, dtype='<U40', the unique filename of each labeled training image.
    X_u:            ndarray, shape=<n, height, width, channels>, dtype=uint8, the unlabeled training image data with range [0, 255].
    y_u:            ndarray, shape=<n>, dtype=int8, the int class of each training image that corresponds to an unlabeled sample.
    basenames_u:    ndarray, shape=<n>, dtype='<U40', the unique filename of each unlabeled training image.
    X_test:         ndarray, shape=<n, height, width, channels>, dtype=uint8, the test image pixel data with range [0, 255]
    y_test:         ndarray, shape=<n>, dtype=int8, the int class of each test image.
    basenames_test: ndarray, shape=<n>, dtype='<U40', the unique filename of each test image.
    int2label:      dict, a int --> str dictionary that maps integer to class name.

@author: Brody Kutt (bkutt@paloaltonetworks.com)
"""

import utils
import argparse
import numpy as np

if __name__ == '__main__':
    # Parse the arguments and options
    parser = argparse.ArgumentParser(description=(
        'Extracts and processes an experimental copy of CIFAR for use with '
        'other algorithms.'))
    parser.add_argument("--train_fp",
                        required=True,
                        type=str,
                        help=("Path to training NPZ."))
    parser.add_argument("--test_fp",
                        required=True,
                        type=str,
                        help=("Path to testing NPZ."))
    parser.add_argument("--y_mask_fp",
                        required=True,
                        type=str,
                        help=("Path to the NPZ containing the Y mask."))
    parser.add_argument("--save_fp",
                        required=True,
                        type=str,
                        help=("Where to save the new dataset file."))
    parser.add_argument(
        "--noise_ratio",
        required=False,
        default=0.0,
        type=float,
        help=('Percent of noise for noisy-label experiments (default: 0.0).'))
    parser.add_argument("--random_seed",
                        required=False,
                        default=utils.random_seed(),
                        type=int,
                        help=('Random seed for repeatability (default: %d).') %
                        utils.random_seed())
    args = parser.parse_args()
    np.random.seed(args.random_seed)
    print('-' * 80)
    # Load full dataset
    data = utils.load_dataset(args.train_fp,
                              args.test_fp,
                              args.y_mask_fp,
                              noise_ratio=args.noise_ratio)
    print('\nSaving to disk...')
    np.savez_compressed(
        args.save_fp,
        X_l=data['X'][data['y_mask'] == 1, :, :].astype('uint8'),
        y_l=data['y'][data['y_mask'] == 1].astype('int8'),
        basenames_l=data['basenames'][data['y_mask'] == 1],
        X_u=data['X'][data['y_mask'] == 0, :, :].astype('uint8'),
        y_u=data['y'][data['y_mask'] == 0].astype('int8'),
        basenames_u=data['basenames'][data['y_mask'] == 0],
        X_test=data['X_val'].astype('uint8'),
        y_test=data['y_val'].astype('int8'),
        basenames_test=data['basenames_val'],
        int2label=data['int2label'])
    print('--> Saved to: %s' % args.save_fp)
    print('\nExiting...')
    print('-' * 80)
