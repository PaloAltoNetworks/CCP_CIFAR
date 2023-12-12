"""
CNN_Input.py

Data feeder class. Implements feeding balanced batches during CCP training.
Also maintains and updates the Y mask and state of Q vectors. Includes
functions for measuring pseudo-label accuracy.

@author: Brody Kutt (bkutt@paloaltonetworks.com)
"""

import random
import numpy as np


class CNN_Input:

    def __init__(self,
                 X,
                 y=None,
                 y_clean=None,
                 y_mask=None,
                 X_val=[],
                 y_val=[],
                 random_seed=1994,
                 n_classes=-1,
                 use_unlab=False,
                 train_batch_size=64,
                 balance_train_batches=True):

        if (y is not None):
            assert (len(X) == len(y))
            if (y_mask is not None):
                assert (len(y) == len(y_mask))
            else:
                y_mask = np.ones(len(y))
        self.using_val = len(X_val) > 0
        if (self.using_val):
            assert (len(X_val) == len(y_val))

        self.X = X  # All the training images
        self.y = y  # All the class indices
        self.y_clean = y_clean  # All labels before noise perturbation (if any)
        self.X_val = X_val  # All the validation images
        self.y_val = y_val  # All the validation labels
        self.y_mask = y_mask  # For masking labels
        self.orig_y_mask = np.copy(y_mask)
        self.orig_unlab_idxs = self.orig_y_mask == 0
        if (y_mask is not None):
            self.n_unlab = int(np.sum(1.0 - self.y_mask))
            self.orig_total_unlab = np.sum(1.0 - y_mask)
        else:
            self.n_unlab = -1
            self.orig_total_unlab = 0.0
        self.i = 0  # Index for batch sampling
        self.num_epochs = 0  # How many epochs processed so far
        self.random_seed = random_seed  # Random seed for repeatability
        self.n_classes = n_classes
        if (n_classes == -1 and y is not None):
            self.n_classes = len(set(y))
        self.n_samples = len(X)
        self.use_unlab = use_unlab
        self.balance_train_batches = balance_train_batches
        self.train_batch_size = train_batch_size
        random.seed(random_seed)
        np.random.seed(random_seed)
        if (self.balance_train_batches
                and self.train_batch_size < self.n_classes + 1):
            print('*' * 80)
            print('WARNING: The training batch size is too small. The '
                  'batches will not be balanced.')
            print('*' * 80)
        self.X_train = None  # Training data
        self.q_train = None  # Training true classes; one hot encoded
        self.n_train = None  # Calculate this once
        self.filt_n_train = None  # Number of labeled training samples.
        self.n_val = None  # Number of validation samples
        self.epoch_idxs = None  # Indices into all training data
        self.epoch_size = None  # How many batches per epoch when processing all data
        self.filt_epoch_idxs = None  # Indices into the labeled training data
        self.filt_epoch_size = None  # How many batches per epoch when processing labeled data
        self.cls2idxs = None  # Indices of training data organized by class
        # Normalize the images between 0.0 and 1.0
        self.X = self.X.astype('float32')
        self.X_val = self.X_val.astype('float32')
        self.X = self.X / 255.0
        self.X_val = self.X_val / 255.0
        if (self.using_val):
            # Use external validation set (val data already been assigned)
            self.X_train = self.X
            self.q_train = self.one_hot(self.y)
            # Update lengths
            self.n_train, self.n_val = len(self.X_train), len(self.y_val)
            self.reset()

    def one_hot(self, y, minval=0.0, maxval=1.0):
        """
        Convert an array of classes into a one-hot encoded version. Unlabeled
        samples (-1) will be assigned the zero vector.
        """
        assert (self.n_classes > 0)
        y_one_hot = np.ones((len(y), self.n_classes)) * minval
        y_one_hot[np.arange(len(y)), y] = maxval
        y_one_hot = y_one_hot * np.reshape(
            (y != -1).astype('float'), [len(y), 1])
        return y_one_hot

    def reset(self):
        """
        Recalculate/reset all indices/counts/epoch iterator/etc
        """
        assert (self.using_val)
        self.filt_n_train = np.sum(self.y_mask)
        self.calc_cls2idxs()
        self.randomize_epoch_idxs()
        self.randomize_filt_epoch_idxs()
        self.i = 0
        self.num_epochs = 0
        self.n_unlab = int(np.sum(1.0 - self.y_mask))

    def calc_cls2idxs(self):
        """
        Calculate a map of which indices into the training data correspond to
        which classes. This changes as Q vectors are progressively updated.
        """
        self.cls2idxs = {-1: []}
        self.filt_cls2idxs = {-1: []}
        self.all_filt_idxs = []
        for i in range(self.n_classes):
            self.cls2idxs[i] = []
            self.filt_cls2idxs[i] = []
        for i, q_row in enumerate(self.q_train):
            if (np.sum(q_row) == 0.0):
                self.cls2idxs[-1].append(i)
            else:
                self.cls2idxs[np.argmax(q_row)].append(i)
        # Repeat for filtered data
        for i, q_row in enumerate(self.q_train):
            if (self.y_mask[i] == 1):
                self.all_filt_idxs.append(i)
                self.filt_cls2idxs[np.argmax(q_row)].append(i)
        self.all_filt_idxs = np.array(self.all_filt_idxs)
        self.filt_n_train = len(self.all_filt_idxs)

    def randomize_epoch_idxs(self):
        """
        Randomizes the training data indices of labeled/unlabeled data for a
        new epoch. Optionally balances the classes.
        """
        assert (self.using_val)
        if (not self.balance_train_batches):
            self.epoch_idxs = random.sample(list(range(self.n_train)),
                                            self.n_train)
            self.epoch_idxs = np.array(self.epoch_idxs)
        else:
            # Use upsampling to increase under-represented classes
            # Randomize lists inside class2idxs
            for i in range(-1, self.n_classes):
                random.shuffle(self.cls2idxs[i])
            # Construct series of indices that produces balanced batches
            cls_pointers, cls_lapped, n_samps_per_cls = {}, {}, {}
            for i in range(-1, self.n_classes):
                cls_pointers[i] = 0
                n_samps_per_cls[i] = len(self.cls2idxs[i])
                if (n_samps_per_cls[i] > 0):
                    cls_lapped[i] = False
                else:
                    cls_lapped[i] = True
            self.epoch_idxs = []
            cur_cls = -1

            def process_idx():
                if (n_samps_per_cls[cur_cls] > 0):
                    new_idx = self.cls2idxs[cur_cls][cls_pointers[cur_cls]]
                    self.epoch_idxs.append(new_idx)
                    cls_pointers[cur_cls] += 1
                    if (cls_pointers[cur_cls] >= n_samps_per_cls[cur_cls]):
                        cls_lapped[cur_cls] = True
                        cls_pointers[cur_cls] = 0
                        random.shuffle(self.cls2idxs[cur_cls])

            while (not all(cls_lapped.values())):
                process_idx()
                if (cur_cls == -1):
                    # Sample unlabeled data at 5x rate
                    for _ in range(4):
                        process_idx()
                cur_cls += 1
                if (cur_cls == self.n_classes):
                    cur_cls = -1
            self.epoch_idxs = np.array(self.epoch_idxs)
        self.epoch_size = len(self.epoch_idxs)
        self.steps_per_epoch = np.maximum(
            self.epoch_size // float(self.train_batch_size), 1)
        self.new_epoch = True

    def randomize_filt_epoch_idxs(self):
        """
        Randomizes the training data indices marked as labeled for a new epoch.
        Optionally balances the classes.
        """
        assert (self.using_val)
        if (not self.balance_train_batches):
            self.filt_epoch_idxs = np.copy(self.all_filt_idxs)
            np.random.shuffle(self.filt_epoch_idxs)
        else:
            # Use upsampling to increase under-represented classes
            # Randomize lists inside class2idxs
            for i in range(self.n_classes):
                random.shuffle(self.filt_cls2idxs[i])
            # Construct series of indices that produces balanced batches
            cls_pointers, cls_lapped, n_samps_per_cls = {}, {}, {}
            for i in range(self.n_classes):
                cls_pointers[i] = 0
                n_samps_per_cls[i] = len(self.filt_cls2idxs[i])
                if (n_samps_per_cls[i] > 0):
                    cls_lapped[i] = False
                else:
                    cls_lapped[i] = True
            self.filt_epoch_idxs = []
            cur_cls = 0

            def process_idx():
                if (n_samps_per_cls[cur_cls] > 0):
                    new_idx = self.filt_cls2idxs[cur_cls][
                        cls_pointers[cur_cls]]
                    self.filt_epoch_idxs.append(new_idx)
                    cls_pointers[cur_cls] += 1
                    if (cls_pointers[cur_cls] >= n_samps_per_cls[cur_cls]):
                        cls_lapped[cur_cls] = True
                        cls_pointers[cur_cls] = 0
                        random.shuffle(self.filt_cls2idxs[cur_cls])

            while (not all(cls_lapped.values())):
                process_idx()
                cur_cls += 1
                if (cur_cls == self.n_classes):
                    cur_cls = 0
            self.filt_epoch_idxs = np.array(self.filt_epoch_idxs)
        self.filt_epoch_size = len(self.filt_epoch_idxs)
        self.filt_steps_per_epoch = np.maximum(
            self.filt_epoch_size // float(self.train_batch_size), 1)
        self.new_epoch = True

    def get_val_batch_iter(self, batch_size=256):
        """
        Generates a batch iterator for a validation dataset.
        """
        assert (self.using_val)
        k = 0  # Keep track of where we are in the loop
        while (k < self.n_val):
            batch = {'y_mask': []}
            if (k + batch_size >= self.n_val):
                batch['X'] = self.X_val[k:self.n_val]
                batch['y'] = self.y_val[k:self.n_val]
                k += batch_size
            else:
                batch['X'] = self.X_val[k:k + batch_size]
                batch['y'] = self.y_val[k:k + batch_size]
                k += batch_size
            batch['selector'] = np.ones([len(batch['X'])], dtype=bool)
            yield batch

    def get_train_batch_iter(self, batch_size=256):
        """
        Generates a batch iterator for the training dataset.
        """
        assert (self.using_val)
        k = 0  # Keep track of where we are in the loop
        while (k < self.filt_n_train):
            batch = {}
            if (k + batch_size >= self.filt_n_train):
                batch['X'] = self.X_train[
                    self.all_filt_idxs[k:self.filt_n_train]]
                batch['y'] = self.y_clean[
                    self.all_filt_idxs[k:self.filt_n_train]]
                batch['q'] = self.q_train[
                    self.all_filt_idxs[k:self.filt_n_train], :]
                batch['y_mask'] = self.y_mask[
                    self.all_filt_idxs[k:self.filt_n_train]]
                k += batch_size
            else:
                batch['X'] = self.X_train[self.all_filt_idxs[k:k + batch_size]]
                batch['y'] = self.y_clean[self.all_filt_idxs[k:k + batch_size]]
                batch['q'] = self.q_train[self.all_filt_idxs[k:k +
                                                             batch_size], :]
                batch['y_mask'] = self.y_mask[self.all_filt_idxs[k:k +
                                                                 batch_size]]
                k += batch_size
            batch['selector'] = np.ones([len(batch['X'])], dtype=bool)
            yield batch

    def next_batch(self):
        """
        Return a new batch of labeled and unlabeled data. Updates indices.
        """
        assert (self.using_val)
        assert (self.train_batch_size > 0)
        if (not self.use_unlab):
            return self._filt_next_batch()

        self.new_epoch = False
        batch = {}
        if (self.i + self.train_batch_size >= self.epoch_size):
            # Grab the rest of the data that hasn't been seen yet
            old_epoch_size = self.epoch_size
            batch['epoch_idxs'] = self.epoch_idxs[self.i:old_epoch_size]
            self.num_epochs += 1
            self.randomize_epoch_idxs()
            # Grab data from new epoch
            diff = (self.i + self.train_batch_size) - old_epoch_size
            batch['epoch_idxs'] = np.append(batch['epoch_idxs'],
                                            self.epoch_idxs[0:diff])
            self.i = diff
        else:
            batch['epoch_idxs'] = self.epoch_idxs[self.i:self.i +
                                                  self.train_batch_size]
            self.i += self.train_batch_size
        # Ensure no duplicates
        batch['epoch_idxs'] = list(set(batch['epoch_idxs']))
        # Grab data
        batch['epoch_idxs'] = np.array(batch['epoch_idxs'])
        X = self.X_train[batch['epoch_idxs']]
        y_clean = self.y_clean[batch['epoch_idxs']]
        q = self.q_train[batch['epoch_idxs'], :]
        y_mask = self.y_mask[batch['epoch_idxs']]
        # Use y_mask to break X into labeled and unlabeled chunks
        lab_X = X[y_mask == 1]
        unlab_X = X[y_mask == 0]
        lab_q = q[y_mask == 1]
        unlab_q = q[y_mask == 0]
        lab_epoch_idxs = batch['epoch_idxs'][y_mask == 1]
        unlab_epoch_idxs = batch['epoch_idxs'][y_mask == 0]
        # Recombine with unlabeled data at the bottom of the batch
        batch['X'] = np.concatenate((lab_X, unlab_X), axis=0)
        batch['q'] = np.concatenate((lab_q, unlab_q), axis=0)
        batch['epoch_idxs'] = np.concatenate(
            (lab_epoch_idxs, unlab_epoch_idxs), axis=0)
        batch['y_clean'] = np.concatenate(
            (y_clean[y_mask == 1], y_clean[y_mask == 0]), axis=0)
        # Selector indicates which samples are marked as labeled/unlabeled
        batch['selector'] = np.concatenate(
            (np.ones_like(lab_epoch_idxs, dtype=bool),
             np.zeros_like(unlab_epoch_idxs, dtype=bool)),
            axis=0)
        return batch

    def _filt_next_batch(self):
        """
        Return a new batch of data marked as labeled. Updates indices.
        """
        self.new_epoch = False
        batch = {}
        if (self.i + self.train_batch_size >= self.filt_epoch_size):
            # Grab the rest of the data that hasn't been seen yet
            old_epoch_size = self.filt_epoch_size
            batch['epoch_idxs'] = self.filt_epoch_idxs[self.i:old_epoch_size]
            self.num_epochs += 1
            self.randomize_filt_epoch_idxs()
            # Grab data from new epoch
            diff = (self.i + self.train_batch_size) - old_epoch_size
            batch['epoch_idxs'] = np.append(batch['epoch_idxs'],
                                            self.filt_epoch_idxs[0:diff])
            self.i = diff
        else:
            batch['epoch_idxs'] = self.filt_epoch_idxs[self.i:self.i +
                                                       self.train_batch_size]
            self.i += self.train_batch_size
        # Ensure no duplicates
        batch['epoch_idxs'] = list(set(batch['epoch_idxs']))
        # Grab data
        batch['epoch_idxs'] = np.array(batch['epoch_idxs'])
        batch['X'] = self.X_train[batch['epoch_idxs']]
        batch['y_clean'] = self.y_clean[batch['epoch_idxs']]
        batch['q'] = self.q_train[batch['epoch_idxs'], :]
        batch['selector'] = np.ones_like(batch['epoch_idxs'], dtype=bool)
        return batch

    def update_q_train(self, new_q_train):
        """
        """
        self.q_train = new_q_train

    def update_y_mask(self, new_y_mask):
        """
        """
        self.y_mask = new_y_mask
        self.reset()

    def calc_acc(self, sel_idxs):
        """
        Return propagation accuracy of selected idxs. Uses the current state of
        q_train and y_clean.
        """
        sel_idxs = np.logical_and(sel_idxs, self.y_clean != -1)
        sel_y_clean = self.y_clean[sel_idxs]
        sel_q_train = np.argmax(self.q_train, axis=1)[sel_idxs]
        if (len(sel_q_train) > 0):
            return np.sum(sel_y_clean == sel_q_train) / float(len(sel_q_train))
        else:
            return -1.0

    def calc_prop_acc(self):
        """
        Return propagation accuracy of samples marked as unlabeled.
        """
        return self.calc_acc(self.orig_y_mask == 0)

    def calc_l_prop_acc(self):
        """
        Return propagation accuracy of samples marked as labeled.
        """
        return self.calc_acc(self.orig_y_mask == 1)

    def calc_avg_err(self, sel_idxs):
        """
        Return propagation error of selected idxs. Uses the current state of
        q_train and y_clean.
        """
        sel_idxs = np.logical_and(sel_idxs, self.y_clean != -1)
        sel_y_clean_one_hot = self.one_hot(self.y_clean[sel_idxs])
        sel_q_train = np.clip(self.q_train[sel_idxs, :], 0.0, 1.0)
        if (len(sel_q_train) > 0):
            return np.mean(np.sum(np.abs(sel_y_clean_one_hot - sel_q_train),
                                  axis=-1),
                           axis=0)
        else:
            return -1.0

    def calc_prop_err(self):
        """
        Return propagation error of samples marked as unlabeled.
        """
        return self.calc_avg_err(self.orig_y_mask == 0)

    def calc_l_prop_err(self):
        """
        Return propagation error of samples marked as labeled.
        """
        return self.calc_avg_err(self.orig_y_mask == 1)

    def calc_avg_unknown_stren(self):
        """
        Return propagation error of samples where the true class is unknown.
        """
        unknown_q = self.q_train[self.y == -1]
        if (len(unknown_q) > 0):
            q_stren = np.max(unknown_q, axis=1)
            return np.mean(q_stren)
        else:
            return -1.0
