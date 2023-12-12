"""
utils.py

@author: Brody Kutt (bkutt@paloaltonetworks.com)
"""

import os
import sys
import numpy as np
from datetime import datetime


def random_seed():
    """
    For repeatability.
    """
    return 1994


def datetime_format():
    """
    """
    return '%Y-%m-%d %H:%M:%S'


def datetime_str():
    """
    Get a string that displays the current date and time in standard format
    """
    return datetime.now().strftime(datetime_format())


def load_data_npz(fp, load_X=True, load_y=True):
    """
    Load the X and y matrices from an npz compressed archive. Used for both SOI
    and EV data.
    """
    result = {}
    data = None
    try:
        data = np.load(fp, mmap_mode=None, allow_pickle=True)
    except IOError:
        pass
    if (data is None):
        try:
            fp += '.npz'  # Try adding the file extension
            data = np.load(fp, mmap_mode=None, allow_pickle=True)
        except IOError:
            return None
    result['basenames'] = data['basenames']
    if (load_X):
        result['X'] = data['X']
    if (load_y):
        result['y'] = data['y'].astype('int')
        if ('int2label' in data):
            result['int2label'] = data['int2label'].item()
    return result


def load_y_mask(path):
    """
    """
    data = None
    try:
        data = np.load(path, mmap_mode=None, allow_pickle=True)
    except IOError:
        pass
    if (data is None):
        try:
            path += '.npz'  # Try adding the file extension
            data = np.load(path, mmap_mode=None, allow_pickle=True)
        except IOError:
            return None
    return data['y_mask'].astype('int')


def load_dataset(train_fp, val_fp=None, y_mask_fp=None, noise_ratio=0.0):
    """
    """
    print('Loading the training data...')
    data = {}
    result = load_data_npz(train_fp)
    if (result is None):
        print('ERR: Failed to read: %s' % train_fp)
        sys.exit(-1)  # Exit failure
    data['basenames'] = result['basenames']
    data['X'] = result['X']
    data['y'] = result['y']
    data['int2label'] = result['int2label']
    n_classes = len(data['int2label']) - 1
    print('--> Loaded %d samples!' % len(data['y']))
    for i in range(-1, n_classes):
        print(('Class %d: %d' % (i, np.sum((data['y'] == i).astype('int')))))

    if (val_fp is not None):
        print('\nLoading the validation data...')
        result = load_data_npz(val_fp)
        if (result is None):
            print('ERR: Failed to read: %s' % val_fp)
            sys.exit(-1)  # Exit failure
        data['basenames_val'] = result['basenames']
        data['X_val'] = result['X']
        data['y_val'] = result['y']
        print(('--> Loaded %d samples!' % len(data['y_val'])))
        for i in range(-1, n_classes):
            print('Class %d: %d' %
                  (i, np.sum((data['y_val'] == i).astype('int'))))

    if (y_mask_fp is not None):
        print('\nLoading the y mask...')
        y_mask = load_y_mask(y_mask_fp)
        if (y_mask is None):
            print('ERR: Failed to load: %s' % y_mask_fp)
            sys.exit(-1)  # Exit failure
        data['y_mask'] = y_mask
        print('--> Loaded y_mask of size: %d' % len(data['y_mask']))
    else:
        print('Using y_mask of ones...')
        data['y_mask'] = np.ones(len(data['y']), dtype='int')

    if (np.sum(data['y_mask'] == -1) > 0):
        print('--> Using y_mask to remove entries marked for removal...')
        data['basenames'] = data['basenames'][data['y_mask'] != -1]
        data['X'] = data['X'][data['y_mask'] != -1]
        data['y'] = data['y'][data['y_mask'] != -1]
        data['y_mask'] = data['y_mask'][data['y_mask'] != -1]
        # Rename the class indices
        l_cls_idxs = list(set(data['y'][data['y_mask'] == 1]))
        assert (-1 not in l_cls_idxs)
        l_cls_idxs.sort()
        new_l_cls_idxs = [i for i in range(len(l_cls_idxs))]
        new_int2label = {-1: 'Unlabeled'}
        old2new = {}
        for old_idx, new_idx in zip(l_cls_idxs, new_l_cls_idxs):
            new_int2label[new_idx] = data['int2label'][old_idx]
            old2new[old_idx] = new_idx
        for j in range(len(data['y'])):
            if (data['y'][j] not in l_cls_idxs):
                data['y'][j] = -1
                assert (data['y_mask'][j] == 0)
            else:
                data['y'][j] = old2new[data['y'][j]]
        if (val_fp is not None):
            for j in range(len(data['y_val'])):
                if (data['y_val'][j] not in l_cls_idxs):
                    data['y_val'][j] = -1
                else:
                    data['y_val'][j] = old2new[data['y_val'][j]]
            data['basenames_val'] = data['basenames_val'][data['y_val'] != -1]
            data['X_val'] = data['X_val'][data['y_val'] != -1]
            data['y_val'] = data['y_val'][data['y_val'] != -1]
            data['int2label'] = new_int2label
    # Make a copy of Y before any noise is added
    data['y_clean'] = np.copy(data['y'])
    if (noise_ratio > 0.0):
        print('--> Using noise ratio of %.2f to corrupt the training '
              'labels...' % noise_ratio)
        n = len(data['y_mask'])
        n_cls = len(set(data['y']))
        if (-1 in set(data['y'])):
            n_cls -= 1
        # Break up all indices into class buckets
        u_idxs = []
        l_idxs = [[] for i in range(n_cls)]
        for i in range(n):
            if (data['y'][i] == -1 or data['y_mask'][i] == 0):
                u_idxs.append(i)
            else:
                l_idxs[int(data['y'][i])].append(i)
        # Convert to NP arrays and shuffle
        u_idxs = np.array(u_idxs)
        for c in range(n_cls):
            l_idxs[c] = np.array(l_idxs[c])
            np.random.shuffle(l_idxs[c])
        # Get all indices of samples that will have class changed
        idxs_to_perturb = np.array([], dtype='int')
        for c in range(n_cls):
            cls_idxs = l_idxs[c]
            idxs_to_perturb = np.concatenate(
                (idxs_to_perturb, cls_idxs[:int(noise_ratio * len(cls_idxs))]))
        rnd_cls_perturbations = np.random.randint(low=1,
                                                  high=n_cls,
                                                  size=len(idxs_to_perturb),
                                                  dtype=data['y'].dtype)
        # Do label perturbation
        data['y'][idxs_to_perturb] = np.mod(
            data['y'][idxs_to_perturb] + rnd_cls_perturbations, n_cls)

    print('\nTraining set summary:')
    for i in sorted(set(data['y'])):
        print('Class %d (%s):' % (i, data['int2label'][i]))
        print('--> Total samples: %d' % np.sum(data['y'] == i))
        print('--> Labeled documents: %d' %
              np.sum(np.logical_and(data['y'] == i, data['y_mask'] == 1)))
        print('--> Unlabeled documents: %d' %
              np.sum(np.logical_and(data['y'] == i, data['y_mask'] == 0)))

    if (val_fp is not None):
        print('\nValidation set summary:')
        for i in sorted(set(data['y_val'])):
            print('Class %d (%s):' % (i, data['int2label'][i]))
            print('--> Total samples: %d' % np.sum(data['y_val'] == i))

    return data


def discover_fps(path):
    """
    Discovers and returns a list of filepaths for every file located below a
    top directory.
    """
    fp_list = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            fp_list.append(os.path.join(root, filename))
    return fp_list


def print_fps_stats(all_fps_by_int, int2label):
    """
    Print some information about the class list and frequency distribution.
    """
    n = float(np.sum([len(all_fps_by_int[i]) for i in all_fps_by_int]))
    print(('-' * 71))
    print(('|%-45s|%4s|%8s|%9s|' % ('CLASS_LABEL', 'INT', 'COUNT', 'PERCENT')))
    print(('-' * 71))
    for i in all_fps_by_int:
        fps = all_fps_by_int[i]
        print(('|%-45s|%4d|%8d|%8s%%|' %
               (int2label[i], i, len(fps),
                str(round(((len(fps) / np.maximum(n, 1)) * 100.0), 1)))))
    print(('-' * 71))


def get_class_labels(int2label):
    """
    Skips over the unlabeled class
    """
    return [int2label[i] for i in range(len(int2label) - 1)]


def print_cm(cm, labels, normalize=True):
    """
    Print a confusion matrix.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    columnwidth = max([len(x) for x in labels] + [8])  # 8 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label, end=" ")
        for j in range(len(labels)):
            if normalize:
                if not np.isnan(cm[i, j]):
                    cell = "%{0}.3f".format(columnwidth) % cm[i, j]
                else:
                    cell = (" " * (columnwidth - 1)) + "-"
            else:
                cell = "%{0}d".format(columnwidth) % cm[i, j]
            print(cell, end=" ")
        print()


def filt_with_y_mask(arr, y_mask):
    """
    """
    y_mask_to_idxs = np.nonzero(y_mask)[0]
    return arr[y_mask_to_idxs]


def shannon_entropy(pk, qk, base=2.0):
    """
    Compute the shannon entropy between probability distributions P and Q with
    a specified base.
    """
    return np.sum(pk * np.log(pk / np.maximum(qk, 1e-6))) / np.log(np.maximum(base, 1e-6))


def eliminate_duplicates(dict1, dict2):
    """
    Remove duplicates between two dictionaries. The larger of the dictionaries
    will have the duplicate values removed from it. Returns the number of
    duplicates found.
    """
    num_duplicates = 0
    if (len(dict1) <= len(dict2)):
        smaller_dict = dict1
        larger_dict = dict2
    else:
        smaller_dict = dict2
        larger_dict = dict1

    for key in list(smaller_dict.keys()):
        if (key in larger_dict):
            num_duplicates += 1
            larger_dict.pop(key)
    return num_duplicates


def merge_dicts(all_dicts, rm_duplicates=True, v=False):
    """
    Merges a list of dictionaries into 1 dictionary containing all entries.
    This function will optionally eliminate duplicated keys.
    """
    merged = all_dicts[0].copy()  # Start with one dict's keys and values
    for i in range(1, len(all_dicts)):
        if (rm_duplicates):
            num_duplicates = eliminate_duplicates(merged, all_dicts[i])
            if (v):
                print(('%d duplicates found and removed.' % num_duplicates))
        merged.update(all_dicts[i])  # Add a new dictionary's keys and values
    return merged
