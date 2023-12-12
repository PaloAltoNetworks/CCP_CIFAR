"""
dataset_construct.py

Process the pickled files of CIFAR and save them in a custom NPZ archive.

The custom NPZ archive consists of the following:

    X:          ndarray, shape=<n, height, width, channels>, dtype=uint8, the image pixel data with range [0, 255]
    y:          ndarray, shape=<n>, dtype=int8, the int class of each image.
    basenames:  ndarray, shape=<n>, dtype='<U40', the unique filename of each image.
    int2label:  dict, a int --> str dictionary that maps integer to class name.

@author: Brody Kutt (bkutt@paloaltonetworks.com)
"""

import os
import sys
import utils
import pickle
import argparse
import numpy as np


def unpickle(file):
    """
    """
    with open(file, 'rb') as fpointer:
        dict = pickle.load(fpointer, encoding='latin1')
    return dict


def get_int2label(cifar_version):
    """
    """
    if (cifar_version == 10):
        return {
            -1: 'unlabeled',
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
        }
    elif (cifar_version == 100):
        labels = [
            'beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium fish',
            'flatfish', 'ray', 'shark', 'trout', 'orchids', 'poppies', 'roses',
            'sunflowers', 'tulips', 'bottles', 'bowls', 'cans', 'cups',
            'plates', 'apples', 'mushrooms', 'oranges', 'pears',
            'sweet peppers', 'clock', 'computer keyboard', 'lamp', 'telephone',
            'television', 'bed', 'chair', 'couch', 'table', 'wardrobe', 'bee',
            'beetle', 'butterfly', 'caterpillar', 'cockroach', 'bear',
            'leopard', 'lion', 'tiger', 'wolf', 'bridge', 'castle', 'house',
            'road', 'skyscraper', 'cloud', 'forest', 'mountain', 'plain',
            'sea', 'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
            'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'crab',
            'lobster', 'snail', 'spider', 'worm', 'baby', 'boy', 'girl', 'man',
            'woman', 'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
            'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', 'maple', 'oak',
            'palm', 'pine', 'willow', 'bicycle', 'bus', 'motorcycle',
            'pickup truck', 'train', 'lawn-mower', 'rocket', 'streetcar',
            'tank', 'tractor'
        ]
        int2label = {-1: 'unlabeled'}
        for i in range(100):
            int2label[i] = labels[i]
        return int2label
    else:
        raise Exception


def construct_npz(raw_data_dir, cifar_version, save_fp=None):
    """
    """
    print('Discovering all data files...')
    fps = utils.discover_fps(raw_data_dir)
    print('--> Found %d files!' % len(fps))
    print('Unpickling...')
    all_X, all_y, all_fns = [], [], []
    for fp in fps:
        data_batch = unpickle(fp)
        X = data_batch['data']
        # Reshape image data
        X = X.reshape(len(X), 3, 32, 32)
        # Transpose image data
        X = X.transpose(0, 2, 3, 1)
        for img in X:
            all_X.append(img)
        key = 'labels' if cifar_version == 10 else 'fine_labels'
        for y in data_batch[key]:
            all_y.append(y)
        for fn in data_batch['filenames']:
            all_fns.append(fn)
    all_X = np.stack(all_X, axis=0)
    all_y = np.array(all_y, dtype=np.int8)
    all_fns = np.array(all_fns)

    # # Optionally visualize the data...
    # import matplotlib
    # matplotlib.use('Agg')  # Force no use of Xwindows backend
    # import matplotlib.pyplot as plt
    # plt.imshow(all_X[0])
    # print(all_y[0])
    # plt.savefig('test.png')

    int2label = get_int2label(cifar_version)
    n_classes = len(int2label) - 1
    all_fns_by_int = {}
    for cls_idx in range(n_classes):
        all_fns_by_int[cls_idx] = []
    for i in range(len(all_y)):
        all_fns_by_int[all_y[i]].append(all_fns[i])
    utils.print_fps_stats(all_fns_by_int, int2label)
    print('--> Done constructing matrices!')
    final_result = {
        'X': all_X,
        'y': all_y,
        'basenames': all_fns,
        'int2label': int2label
    }
    # Store all the requested data, if desired
    if (save_fp is not None):
        print('--> Writing matrices to disk...')
        np.savez_compressed(save_fp,
                            X=all_X,
                            y=all_y,
                            basenames=all_fns,
                            int2label=int2label)
        print('--> Wrote results to: %s' % save_fp)
    print('Final X shape: %s' % str(final_result['X'].shape))
    print('Final y shape: %s' % str(final_result['y'].shape))
    return final_result


if __name__ == '__main__':
    # Parse the arguments and options
    parser = argparse.ArgumentParser(
        description=('Create a data matrix from raw images.'))
    parser.add_argument('--raw_data_dir',
                        dest='raw_data_dir',
                        required=True,
                        metavar='path/to/raw_data/',
                        help=('Path to pickled raw data files.'))
    parser.add_argument('--cifar_version',
                        required=True,
                        type=int,
                        help=('Specify either 10 or 100.'))
    parser.add_argument(
        '--save_fp',
        default=None,
        required=False,
        metavar='path/to/save.npz',
        help=('Where to save the .npz file containing the data. If None, no '
              '.npz file will be saved to disk (default: None).'))
    args = parser.parse_args()
    if (args.cifar_version != 10 and args.cifar_version != 100):
        print('ERR: dataset_construct.py: CIFAR version not recognized.\n')
        parser.print_help()
        sys.exit(-1)  # Exit failure
    # Check for valid raw data path
    if (not os.path.isdir(args.raw_data_dir)):
        print('ERR: dataset_construct.py: Path to raw data was not '
              'found!\n')
        parser.print_help()
        sys.exit(-1)  # Exit failure
    if (args.save_fp is not None
            and not os.path.isdir(os.path.dirname(args.save_fp))):
        os.makedirs(os.path.dirname(args.save_fp))
    # Pass arguments to main function
    print('-' * 80)
    print('Loading CIFAR-%d...' % args.cifar_version)
    print('Computing X and y with the following settings:')
    print(('Path to raw data: %s' % args.raw_data_dir))
    print(('Path to save location: %s' % args.save_fp))
    construct_npz(args.raw_data_dir, args.cifar_version, save_fp=args.save_fp)
    print('\nExiting...')
    print('-' * 80)
