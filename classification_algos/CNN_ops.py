"""
CNN_ops.py

Helper functions mostly for CCP_Builder.py

@author: Brody Kutt (bkutt@paloaltonetworks.com)
"""

import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(currentdir))
import utils
import numpy as np
import tensorflow as tf
from datetime import datetime
import tensorflow_addons as tfa
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import device_lib
from tensorflow.python.training import saver as saver_lib

CUR_DEVICE_IDX = 0


def log_device_placement():
    """
    Set this to true to debug GPU usage
    """
    return False


def sim_metrics():
    """
    Available similarity metrics
    """
    return ['adj_cos', 'sqrt_cos', 'angular']


def batch_summary_format():
    """
    The format of the CSV summary file to be written during batch training
    """
    return [
        'TIME', 'STEP', 'EPOCH', 'CE_LOSS', 'SC_LOSS', 'AVG_CORR_PROP_STREN',
        'AVG_INCORR_PROP_STREN', 'PROP_ACC', 'BATCH_ACC'
    ]


def train_summary_format():
    """
    The format of the CSV summary file to be written during batch training
    """
    return [
        'TIME', 'STEP', 'EPOCH', 'TRAIN_LOSS', 'TRAIN_ACC', 'TRAIN_EMA_LOSS',
        'TRAIN_EMA_ACC'
    ]


def val_summary_format():
    """
    The format of the CSV summary file to be written during batch training
    """
    return [
        'TIME', 'STEP', 'EPOCH', 'VAL_LOSS', 'VAL_ACC', 'VAL_EMA_LOSS',
        'VAL_EMA_ACC'
    ]


def record_result(fp, results):
    """
    Write results into the val and train summary files.
    """
    time_str = datetime.now().strftime(utils.datetime_format())
    results = [time_str] + results
    str_results = []
    for r in results:
        if (isinstance(r, list)):
            if (len(r) == 0):
                str_results.append('-')
            else:
                str_results.append('|'.join([str(e) for e in r]))
        else:
            str_results.append(str(r))
    fp.write(','.join(str_results) + '\n')
    fp.flush()


def get_avail_devices():
    """
    Return a list of all available devices, preferring GPUs if available.
    Note that this is referring to number of CPUs not number of possible
    cores/threads.
    """
    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    cpus = [x.name for x in local_device_protos if x.device_type == 'CPU']
    device_ids = []
    if (len(gpus) > 0):
        for device_str in gpus:
            device_ids.append(device_str.split(':')[-1])
        device_type = 'gpu'
    else:
        for device_str in cpus:
            device_ids.append(device_str.split(':')[-1])
        device_type = 'cpu'
    return device_type, device_ids


def next_device(device_type, device_ids, cur_device_idx):
    """
    Get the name of the next device to assign something to.
    """
    device_name = '/%s:%d' % (device_type, cur_device_idx)
    cur_device_idx = (cur_device_idx + 1) % max(len(device_ids), 1)
    return device_name, cur_device_idx


def pairwise_cos_sim(A, B, weight_vec=None, pre_normalized=False):
    """
    Computes the pairwise cosine similarity vector. The range is between
    -1 and 1, inclusive. 1 corresponds to higher similarity.

    Args:
        A,            [m, d] matrix
        B,            [n, d] matrix
        weight_vec,   unused
    Returns:
        D,            [m, n] matrix of pairwise cosine similarities
    """
    if (not pre_normalized):
        A = tf.math.l2_normalize(A, -1)
        B = tf.math.l2_normalize(B, -1)
    return tf.matmul(A, B, transpose_b=True)


def pairwise_adj_cos_sim(A, B, weight_vec=None, pre_normalized=False):
    """
    Computes the pairwise cosine similarity vector. The range is between
    0 and 1, inclusive. 1 corresponds to higher similarity.

    Args:
        A,            [m, d] matrix
        B,            [n, d] matrix
        weight_vec,   unused
        epsilon,      float
    Returns:
        D,            [m, n] matrix of pairwise adjusted cosine similarities
    """
    with tf.name_scope('pair_adj_cos_sim'):
        cos_sim = pairwise_cos_sim(A, B, pre_normalized=pre_normalized)
        return (cos_sim + 1.0) / 2.0


def pairwise_sqrt_cos_sim(A, B, weight_vec=None, pre_normalized=False):
    """
    Computes the pairwise square root cosine similarity vector. The range is
    between 0 and 1, inclusive. 1 corresponds to higher similarity.

    Args:
        A,            [m, d] matrix
        B,            [n, d] matrix
        weight_vec,   unused
        epsilon,      float
    Returns:
        D,            [m, n] matrix of pairwise sqrt cosine similarities
    """
    with tf.name_scope('pair_adj_cos_sim'):
        cos_sim = pairwise_cos_sim(A, B, pre_normalized=pre_normalized)
        # Clip to safe range
        cos_sim = tf.clip_by_value(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
        return 1.0 - tf.math.sqrt((1.0 - cos_sim) / 2.0)


def pairwise_angular_sim(A, B, weight_vec=None, pre_normalized=False):
    """
    Computes the pairwise angular similarity vector. The range is between
    0 and 1, inclusive. 1 corresponds to higher similarity.

    Args:
        A,            [m, d] matrix
        B,            [n, d] matrix
        weight_vec,   unused
        epsilon,      float
    Returns:
        D,            [m, n] matrix of pairwise angular similarities
    """
    with tf.name_scope('pair_ang_sim'):
        cos_sim = pairwise_cos_sim(A, B, pre_normalized=pre_normalized)
        # Clip to safe range
        cos_sim = tf.clip_by_value(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
        return 1.0 - (tf.math.acos(cos_sim) / np.pi)


def pairwise_cos_sim_np(A, B, weight_vec=None, pre_normalized=False):
    """
    NumPy implementation

    Computes the pairwise cosine similarity vector. The range is between
    -1 and 1, inclusive. 1 corresponds to higher similarity.

    Args:
        A,            [m, d] matrix
        B,            [n, d] matrix
        weight_vec,   unused
    Returns:
        D,            [m, n] matrix of pairwise cosine similarities
    """
    if (not pre_normalized):
        A = normalize(A, norm='l2', axis=-1)
        B = normalize(B, norm='l2', axis=-1)
    return np.matmul(A, np.transpose(B))


def pairwise_adj_cos_sim_np(A, B, weight_vec=None, pre_normalized=False):
    """
    NumPy implementation

    Computes the pairwise cosine similarity vector. The range is between
    0 and 1, inclusive. 1 corresponds to higher similarity.

    Args:
        A,            [m, d] matrix
        B,            [n, d] matrix
        weight_vec,   unused
        epsilon,      float
    Returns:
        D,            [m, n] matrix of pairwise adjusted cosine similarities
    """
    cos_sim = pairwise_cos_sim_np(A, B, pre_normalized=pre_normalized)
    return (cos_sim + 1.0) / 2.0


def pairwise_sqrt_cos_sim_np(A, B, weight_vec=None, pre_normalized=False):
    """
    NumPy implementation

    Computes the pairwise square root cosine similarity vector. The range is
    between 0 and 1, inclusive. 1 corresponds to higher similarity.

    Args:
        A,            [m, d] matrix
        B,            [n, d] matrix
        weight_vec,   unused
        epsilon,      float
    Returns:
        D,            [m, n] matrix of pairwise sqrt cosine similarities
    """
    cos_sim = pairwise_cos_sim_np(A, B, pre_normalized=pre_normalized)
    # Clip to safe range
    cos_sim = np.clip(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
    return 1.0 - np.sqrt((1.0 - cos_sim) / 2.0)


def pairwise_angular_sim_np(A, B, weight_vec=None, pre_normalized=False):
    """
    NumPy implementation

    Computes the pairwise angular similarity vector. The range is between
    0 and 1, inclusive. 1 corresponds to higher similarity.

    Args:
        A,            [m, d] matrix
        B,            [n, d] matrix
        weight_vec,   unused
        epsilon,      float
    Returns:
        D,            [m, n] matrix of pairwise angular similarities
    """
    cos_sim = pairwise_cos_sim_np(A, B, pre_normalized=pre_normalized)
    # Clip to safe range
    cos_sim = np.clip(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
    return 1.0 - (np.arccos(cos_sim) / np.pi)


def WRN(inp, is_training, activation_func, drop_keep_prob, n_classes,
        random_seed, cifar_version):
    """
    This implementation of WRN is based on https://github.com/akshaymehra24/WideResnet/blob/master/WRN_converted_to_tf.py
    """

    def wide_basic(inputs, in_planes, out_planes, stride, name, index):
        if stride != 1 or in_planes != out_planes:
            skip_c = tf.compat.v1.layers.Conv2D(
                out_planes,
                1,
                strides=stride,
                use_bias=True,
                padding='same',
                kernel_initializer=tf.keras.initializers.HeNormal(
                    seed=random_seed))(inputs)
        else:
            skip_c = inputs
        x = tf.compat.v1.layers.BatchNormalization(momentum=0.9,
                                                   scale=True,
                                                   center=True)(
                                                       inputs,
                                                       training=is_training)
        x = activation_func(inputs)
        x = tf.compat.v1.layers.Conv2D(
            out_planes,
            3,
            strides=1,
            use_bias=True,
            padding='same',
            kernel_initializer=tf.keras.initializers.HeNormal(
                seed=random_seed))(x)
        # x = tf.compat.v1.layers.Dropout(rate=1.0 - drop_keep_prob, seed=random_seed)(x, training=is_training)
        x = tf.nn.dropout(x, rate=1 - drop_keep_prob)  # Much faster this way
        x = tf.compat.v1.layers.BatchNormalization(momentum=0.9,
                                                   scale=True,
                                                   center=True)(
                                                       x, training=is_training)
        x = activation_func(x)
        x = tf.compat.v1.layers.Conv2D(
            out_planes,
            3,
            strides=stride,
            use_bias=True,
            padding='same',
            kernel_initializer=tf.keras.initializers.HeNormal(
                seed=random_seed))(x)
        x = tf.add(skip_c, x)
        return x

    def wide_layer(out, in_planes, out_planes, num_blocks, stride, name):
        strides = [stride] + [1] * int(num_blocks - 1)
        i = 0
        for strid in strides:
            out = wide_basic(out,
                             in_planes,
                             out_planes,
                             strid,
                             name='layer1_' + str(i) + '_',
                             index=i)
            in_planes = out_planes
            i += 1
        return out

    def make_resnet_filter(inp, depth, widen_factor):
        n = (depth - 4) / 6
        nStages = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        x = tf.compat.v1.layers.Conv2D(
            nStages[0],
            3,
            strides=1,
            use_bias=False,
            padding='same',
            kernel_initializer=tf.keras.initializers.HeNormal(
                seed=random_seed))(inp)
        with tf.device(next_device()):
            x = wide_layer(x,
                           nStages[0],
                           nStages[1],
                           n,
                           stride=1,
                           name='layer1_')
        with tf.device(next_device()):
            x = wide_layer(x,
                           nStages[1],
                           nStages[2],
                           n,
                           stride=2,
                           name='layer2_')
        with tf.device(next_device()):
            x = wide_layer(x,
                           nStages[2],
                           nStages[3],
                           n,
                           stride=2,
                           name='layer3_')
        x = tf.compat.v1.layers.BatchNormalization(momentum=0.9,
                                                   scale=True,
                                                   center=True)(
                                                       x, training=is_training)
        x = activation_func(x)
        x = tfa.layers.AdaptiveAveragePooling2D(output_size=[1, 1])(x)
        x = tf.reshape(x, (-1, nStages[-1]))
        return x

    device_type, device_ids = get_avail_devices()
    n_devices = len(device_ids)

    def next_device():
        """
        Get the name of the next device to assign something to.
        """
        global CUR_DEVICE_IDX
        device_name = '/%s:%d' % (device_type, CUR_DEVICE_IDX)
        CUR_DEVICE_IDX = (CUR_DEVICE_IDX + 1) % max(n_devices, 1)
        return device_name

    if (cifar_version == 10):
        widen_factor = 2
    else:
        widen_factor = 8
    return make_resnet_filter(inp, depth=28, widen_factor=widen_factor)


def dense_layer(inp,
                in_dim,
                out_dim,
                activation_func=None,
                use_bias=True,
                prefix='dense'):
    """
    Fully connected layer used for the projection heads.
    """
    dense_z = tf.reshape(inp, [-1, in_dim], name='%s-dense_z' % prefix)
    # Define weights and biases for final projection
    dense_W = tf.compat.v1.get_variable(
        '%s-dense_W' % prefix,
        initializer=lambda: tf.random.truncated_normal([in_dim, out_dim],
                                                       stddev=0.1),
        trainable=True,
        dtype=tf.float32)
    if (use_bias):
        dense_b = tf.compat.v1.get_variable(
            '%s-dense_biases' % prefix,
            initializer=lambda: tf.constant(0.1, shape=[out_dim]),
            trainable=True,
            dtype=tf.float32)
        out = tf.compat.v1.nn.xw_plus_b(dense_z, dense_W, dense_b)
    else:
        out = tf.linalg.matmul(dense_z, dense_W)
    if (activation_func is not None):
        out = activation_func(out)
    return out


def _define_output_dirs(base_save_path,
                        name,
                        prefix='',
                        batch_summary_format=batch_summary_format(),
                        train_summary_format=train_summary_format(),
                        val_summary_format=val_summary_format()):
    """
    Set up the output directory.
    """
    # Create folder for all the results and models of this instance, if needed
    class_dir = os.path.join(base_save_path, name)
    if (len(prefix) > 0):
        class_dir = os.path.join(class_dir, prefix)
    if (not os.path.exists(class_dir)):
        os.makedirs(class_dir)
    # To store all paths and files
    resources = {}
    # Create the output directory for batch, training and validation
    # summaries
    resources['batch_summary_path'] = os.path.join(
        class_dir, (name + '_batch_summary.csv'))
    resources['train_summary_path'] = os.path.join(
        class_dir, (name + '_train_summary.csv'))
    resources['val_summary_path'] = os.path.join(class_dir,
                                                 (name + '_val_summary.csv'))
    print(('Writing batch summary to: %s' % resources['batch_summary_path']))
    print(
        ('Writing training summary to: %s' % resources['train_summary_path']))
    print(
        ('Writing validation summary to: %s' % resources['val_summary_path']))
    resources['batch_summary_file'] = open(resources['batch_summary_path'],
                                           'w')
    resources['train_summary_file'] = open(resources['train_summary_path'],
                                           'w')
    resources['val_summary_file'] = open(resources['val_summary_path'], 'w')
    resources['batch_summary_file'].write(','.join(batch_summary_format) +
                                          '\n')
    resources['train_summary_file'].write(','.join(train_summary_format) +
                                          '\n')
    resources['val_summary_file'].write(','.join(val_summary_format) + '\n')

    # Create the output directory for models
    model_dir = os.path.join(base_save_path, name, 'models')
    if (not os.path.exists(model_dir)):
        os.makedirs(model_dir)
    # Only save warmup modles for this demo
    resources['model_dir_warmup'] = os.path.join(model_dir, 'warmup')

    if (not os.path.exists(resources['model_dir_warmup'])):
        os.makedirs(resources['model_dir_warmup'])
    resources['model_fp_warmup'] = os.path.join(resources['model_dir_warmup'],
                                                name)
    resources['warmup_chk_path'] = ''

    resources['saver'] = saver_lib.Saver(max_to_keep=1,
                                         write_version=saver_pb2.SaverDef.V2)
    print('Writing warmup models to: %s' % resources['model_fp_warmup'])
    return resources


def training_loop(train_step,
                  test_step,
                  to_training,
                  to_ema,
                  cnn_input,
                  eval_on_train_data,
                  cls_min_epochs,
                  cls_max_evals_since_overwrite,
                  cls_epochs_per_eval,
                  debug_mode,
                  sess,
                  resources,
                  class_labels,
                  in_cls=True):
    """
    One common loop to use during warmup, CCP iterations, and classifier
    training. Evaluations on test data are only completed during classifier
    training i.e. in_cls=True.
    """
    # Keep track of the min val set loss (and associated stats)
    # we've seen so far
    min_val_loss = np.inf
    best_val_acc = -np.inf
    best_val_cm = None
    ema_min_val_loss = np.inf
    ema_best_val_acc = -np.inf
    ema_best_val_cm = None
    do_eval = in_cls
    # The number of val set evaluations since we've beaten our
    # best val set loss. If this gets large, we can be confident
    # that we wont beat it in this training session.
    num_evals_since_overwrite = 0
    # Enter the training loop
    cur_step_num = 0
    if (debug_mode):
        cls_epochs_per_eval = 1
        cls_min_epochs = 3
    while (True):
        # Check if eval needs to be done
        if (do_eval):
            overwrite = False
            print('\n------- Doing eval on training weights... -------')
            if eval_on_train_data:
                print('\nDoing train set evaluation...')
                result = test_step(cur_step_num,
                                   cnn_input.get_train_batch_iter())
            print('\nDoing val set evaluation...')
            result = test_step(cur_step_num, cnn_input.get_val_batch_iter())
            # Check to see if a new best models have been found
            if (min_val_loss > result['avg_loss']):
                print('New min loss found! Loss improved from: %f --> %f' %
                      (min_val_loss, result['avg_loss']))
                # Update best performance vars
                min_val_loss = result['avg_loss']
                num_evals_since_overwrite = 0  # Reset this to 0
                overwrite = True
            if (best_val_acc < result['acc']):
                print('New best acc found! Acc improved from: %f --> %f' %
                      (best_val_acc, result['acc']))
                best_val_acc = result['acc']
                best_val_cm = result['opt_acc_cm']
                num_evals_since_overwrite = 0  # Reset this to 0
                overwrite = True

            print('\n------- Doing eval on EMA weights... -------')
            sess.run(to_ema())
            if eval_on_train_data:
                print('\nDoing train set evaluation...')
                ema_result = test_step(cur_step_num,
                                       cnn_input.get_train_batch_iter())
                record_result(resources['train_summary_file'], [
                    cur_step_num, cnn_input.num_epochs, result['avg_loss'],
                    result['acc'], ema_result['avg_loss'], ema_result['acc']
                ])
            print('\nDoing val set evaluation...')
            ema_result = test_step(cur_step_num,
                                   cnn_input.get_val_batch_iter())
            record_result(resources['val_summary_file'], [
                cur_step_num, cnn_input.num_epochs, result['avg_loss'],
                result['acc'], ema_result['avg_loss'], ema_result['acc']
            ])
            # Check to see if a new best models have been found
            if (ema_min_val_loss > ema_result['avg_loss']):
                print('New min loss found! Loss improved from: %f --> %f' %
                      (ema_min_val_loss, ema_result['avg_loss']))
                # Update best performance vars
                ema_min_val_loss = ema_result['avg_loss']
                num_evals_since_overwrite = 0  # Reset this to 0
                overwrite = True
            if (ema_best_val_acc < ema_result['acc']):
                print('New best acc found! Acc improved from: %f --> %f' %
                      (ema_best_val_acc, ema_result['acc']))
                ema_best_val_acc = ema_result['acc']
                ema_best_val_cm = ema_result['opt_acc_cm']
                num_evals_since_overwrite = 0  # Reset this to 0
                overwrite = True

            if (not overwrite):
                num_evals_since_overwrite += 1

            print('\nBest results so far...')
            print('VAL_LOSS: %s, VAL_ACC: %s' % (min_val_loss, best_val_acc))
            print('EMA_VAL_LOSS: %s, EMA_VAL_ACC: %s' %
                  (ema_min_val_loss, ema_best_val_acc))
            print('Number of evals since an overwrite: %d\n' %
                  num_evals_since_overwrite)
            do_eval = False  # Reset flag
            sess.run(to_training())  # Switch back to training weights
        # Check exit conditions
        if (in_cls and cnn_input.num_epochs >= cls_min_epochs
                and num_evals_since_overwrite > cls_max_evals_since_overwrite):
            break
        # Process a training batch
        cur_step_num, in_cls, early_exit = train_step(
            cnn_input.next_batch(), resources['batch_summary_file'])
        if (not in_cls):
            num_evals_since_overwrite = 0
            do_eval = False
        else:
            if (cnn_input.new_epoch
                    and cnn_input.num_epochs % cls_epochs_per_eval == 0):
                do_eval = True
        if ((debug_mode and cnn_input.num_epochs > 3 and in_cls)
                or early_exit):
            break

    # Done with training
    print('-' * 40)
    if (min_val_loss < np.inf):
        print('The maximum number of evals per overwrite has been '
              'exceeded. The final results are as follows:\n')
        print('--> Best Val Loss: %s' % str(min_val_loss))
        print('--> Best Val Acc: %s' % str(best_val_acc))
        print('--> Best Val Confusion Matrix:')
        utils.print_cm(best_val_cm, class_labels, normalize=False)
        print('--> Best Val Confusion Matrix Normalized:')
        utils.print_cm(best_val_cm, class_labels)
        print('\n---- EMA Results... ----\n')
        print('--> Best Val Loss: %s' % str(ema_min_val_loss))
        print('--> Best Val Acc: %s' % str(ema_best_val_acc))
        print('--> Best Val Confusion Matrix:')
        utils.print_cm(ema_best_val_cm, class_labels, normalize=False)
        print('--> Best Val Confusion Matrix Normalized:')
        utils.print_cm(ema_best_val_cm, class_labels)
    return {
        'min_val_loss': min_val_loss,
        'best_val_acc': best_val_acc,
        'best_val_cm': best_val_cm,
        'ema_min_val_loss': ema_min_val_loss,
        'ema_best_val_acc': ema_best_val_acc,
        'ema_best_val_cm': ema_best_val_cm
    }


def eye_in_idxs(shape,
                row_start,
                row_end,
                col_start,
                col_end,
                dtype=tf.float32):
    """
    Create a 2D identity matrix anywhere inside a larger matrix of zeros.
    """
    m, n = shape[0], shape[1]
    eye_dim = row_end - row_start
    eye = tf.eye(eye_dim, dtype=dtype)
    eye = tf.concat([
        tf.zeros([row_start, eye_dim]), eye,
        tf.zeros([m - row_end, eye_dim])
    ], 0)
    eye = tf.concat(
        [tf.zeros([m, col_start]), eye,
         tf.zeros([m, n - col_end])], 1)
    return eye
