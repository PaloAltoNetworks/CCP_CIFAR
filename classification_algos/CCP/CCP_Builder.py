"""
CCP_Builder.py

Class to manage all aspects of the training and evaluation of CCP. Called by
train_CCP.py

@author: Brody Kutt (bkutt@paloaltonetworks.com)
"""

import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(currentdir))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
import utils
import CNN_ops
import CCP_losses
import numpy as np
import tensorflow as tf
from CNN_Input import CNN_Input
from sklearn.metrics import confusion_matrix
import instance_transformations as transformations

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Disable INFO logs in TF

# Set this to true to cut training short (only used for debugging)
DEBUG_MODE = False
# Print stats every X batches
STEPS_PER_PRINT = 10
# Learning rate during the first iteration of CCP if it has been altered
ALTER_FIRST_ITER_LR = 0.0006
# The target number of epochs for the cosine learning rate scheduler during classifier training
CLS_LR_TAR_EPOCHS = 512
# The target number of epochs for the cosine learning rate scheduler during CCP
SC_LR_TAR_EPOCHS = 400
# The target number of epochs for the cosine learning rate scheduler during warmup
WU_LR_TAR_EPOCHS = 1028
# What to divide the maximum KL Divergence parameter by after each CCP iteration
MAX_KL_DIV_DIVIDER = 2.0


class CCP_Builder(object):
    """
    """

    def __init__(
        self,
        # The identifiable name of the model that will be saved once this
        # instance has undergone a training process. The "winner" model
        # with the lowest loss will be the one who is saved with this name
        name,
        # Base path to save all models and training results.
        base_save_path,
        # Layers of the contrastive loss projection head
        z_dims=[],
        # Layers of the classification projection head
        g_dims=[],
        # The type of similarity function to use.
        sim_metric='adj_cos',
        # What kind of activation function to use throughout
        activation='relu',
        # The coefficient for the L2 loss. Use l2_lambda=0.0 for no
        # regularization.
        l2_lambda=0.0,
        # Seed all random generation for repeatability
        random_seed=1994,
        # A dictionary to map prototype numbers to interpretable string
        # labels.
        int2label=None,
        # Temperature
        tau=0.1,
        # Use transformations during CCP iterations
        trans_in_ccp=False,
        # Use transformations during classifier training
        trans_in_cls=False,
        # How many epochs per CCP iteration. To run until loss convergence,
        # use -1.
        ccp_iter_schedule=[-1, -1, -1, -1],
        # Whether to scale Q vectors
        scale_q=False,
        # Maximum KL-Divergence used during subsampling
        max_kl_div=-1.0,
        # Whether to keep known labels the same throughout training
        clamp_labeled=True,
        # Which version of the CIFAR dataset (10 or 100)
        cifar_version=10,
        # Set to false to replace credibility with softmax
        use_cred=True,
        # Use a combination of CCE and contrastive loss during classifier
        # training
        use_comb_loss=True):

        self.name = name
        self.base_save_path = base_save_path
        self.base_save_path = base_save_path
        self.l2_lambda = l2_lambda
        self.sim_metric = sim_metric
        if (sim_metric == 'angular'):
            self.sim_func = CNN_ops.pairwise_angular_sim
        elif (sim_metric == 'adj_cos'):
            self.sim_func = CNN_ops.pairwise_adj_cos_sim
        elif (sim_metric == 'sqrt_cos'):
            self.sim_func = CNN_ops.pairwise_sqrt_cos_sim
        else:
            assert (sim_metric in CNN_ops.sim_metrics())
        self.activation = activation
        if (activation == 'relu'):
            self.activation_func = tf.nn.relu
        elif (activation == 'gelu'):
            self.activation_func = tf.nn.gelu
        elif (activation == 'elu'):
            self.activation_func = tf.nn.elu
        else:
            assert (activation in ['gelu', 'relu', 'elu'])
        self.random_seed = random_seed
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        self.int2label = int2label
        self.n_classes = len(int2label) - 1
        self.z_dims = z_dims
        self.g_dims = g_dims
        # Add final layer to f_g() calculate class scores using CE
        self.g_dims.append(self.n_classes)
        self.class_labels = utils.get_class_labels(int2label=self.int2label)
        self.sec_until_10000 = -1
        # Small epsilon value to avoid zero division
        self.epsilon = 1e-6
        # Size of output vectors from ResNet
        self.cifar_version = cifar_version
        assert (self.cifar_version == 10 or self.cifar_version == 100)
        if (self.cifar_version == 10):
            self.wrn_out_dim = 64 * 2
        else:
            self.wrn_out_dim = 64 * 8
        self.b_dim = self.wrn_out_dim
        if (len(self.z_dims) > 0):
            self.z_dim = self.z_dims[-1]
        else:
            self.z_dim = self.b_dim
        self.use_cred = use_cred
        self.use_comb_loss = use_comb_loss
        self.ccp_iter_schedule = ccp_iter_schedule
        self.n_iters = len(ccp_iter_schedule)
        self.scale_q = scale_q
        self.tau = tau
        self.trans_in_ccp = trans_in_ccp
        self.trans_in_cls = trans_in_cls
        self.max_kl_div = max_kl_div
        self.abs_min_keep_perc = 0.01
        self.last_keep_perc = 0.09
        self.clamp_labeled = clamp_labeled
        # Parameters for normalization
        if (self.cifar_version == 10):
            self.means = (0.4914, 0.4822, 0.4465)
            self.stds = (0.2471, 0.2435, 0.2616)
        else:
            self.means = (0.5071, 0.4867, 0.4408)
            self.stds = (0.2675, 0.2565, 0.2761)

    def _build_CCP(self):
        """
        Define the TF graph
        """
        # Placeholder for data
        self.X_batch = tf.compat.v1.placeholder(
            tf.float32,
            # [batch_size, img_height, img_width, n_channels]
            [None, 32, 32, 3],
            name='X_batch')
        self.batch_size_ref = tf.shape(self.X_batch)[0]
        # Placeholder to indicate which samples should be treated as unlabeled
        self.selector = tf.compat.v1.placeholder_with_default(
            tf.ones([0], dtype=tf.bool),  # Dummy value
            shape=[None],
            name='selector')
        # Placeholder for credibility vectors in a batch
        self.q_batch = tf.compat.v1.placeholder_with_default(
            tf.zeros([0, self.n_classes], dtype=tf.float32),  # Dummy value
            shape=[None, self.n_classes],
            name='q_batch')
        self.n_l = tf.reduce_sum(tf.cast(self.selector, tf.int32))
        self.n_u = self.batch_size_ref - self.n_l
        # Placeholder for probability of keeping a neuron in the dropout layer
        self.drop_keep_prob = tf.compat.v1.placeholder_with_default(
            1.0, shape=[], name='drop_keep_prob')
        # Placeholder for training flag
        self.is_training = tf.compat.v1.placeholder_with_default(
            False, shape=[], name='is_training')
        # Placeholder for pre-training warmup flag
        self.is_pretraining = tf.compat.v1.placeholder_with_default(
            False, shape=[], name='is_pretraining')
        # Whether to expand the batch with transformations
        self.use_trans = tf.compat.v1.placeholder_with_default(
            False, shape=[], name='use_trans')
        # Placeholder for learning rate
        self.lr = tf.compat.v1.placeholder_with_default(0.001,
                                                        shape=[],
                                                        name='lr')

        # Do transformations on samples
        with tf.name_scope('X_trans'):
            X_ij = tf.cond(self.use_trans,
                           lambda: self.get_trans(self.X_batch),
                           lambda: self.X_batch)
            self.q_ij = tf.cond(
                self.use_trans,
                lambda: tf.concat([self.q_batch, self.q_batch], axis=0),
                lambda: self.q_batch)
            self.selector_ij = tf.cond(
                self.use_trans,
                lambda: tf.concat([self.selector, self.selector], axis=0),
                lambda: self.selector)
            self.ij_batch_size_ref = tf.shape(X_ij)[0]

        with tf.name_scope('normalize'):
            # Normalize the images
            std_X_ij = []
            for channel in range(3):
                std_X_ij.append(
                    (X_ij[:, :, :, channel] - self.means[channel]) /
                    self.stds[channel])
            X_ij = tf.stack(std_X_ij, axis=-1)

        with tf.name_scope('f_b'):
            # Send samples through the encoder network
            bx_ij = self.f_b(X_ij, self.is_training)
            bx_ij = tf.reshape(bx_ij, [self.ij_batch_size_ref, self.b_dim],
                               name='bx_ij')

        with tf.name_scope('f_z'):
            # Send samples through the contrastive projection head
            zx_ij = self.projection_head(bx_ij, self.z_dims, proj_type='Z')
            # Normalize encoding to unit hypersphere
            zx_ij = tf.math.l2_normalize(zx_ij, -1)
            self.zx_ij = tf.reshape(zx_ij,
                                    [self.ij_batch_size_ref, self.z_dim],
                                    name='zx_ij')

        with tf.name_scope('cred_prop'):
            # Do credibility propagation (Algorithm 2) to unlabeled data
            new_q_u = tf.cond(
                tf.math.greater(self.n_u, 0), lambda: CCP_losses.compute_cp(
                    self.zx_ij, self.q_ij, self.selector_ij, self.sim_func,
                    self.n_classes, self.epsilon, self.use_trans, self.use_cred
                ),
                lambda: tf.zeros([self.n_u, self.n_classes], dtype=tf.float32))
            self.new_q_u = tf.reshape(new_q_u, [self.n_u, self.n_classes],
                                      name='new_q_u')
            # To get propagated labels for labeled data, if requested, simply
            # switch the labeled and unlabeled data (invert the selector)
            new_q_l = tf.cond(
                tf.math.greater(self.n_l, 0), lambda: CCP_losses.compute_cp(
                    self.zx_ij, self.q_ij, tf.math.logical_not(
                        self.selector_ij), self.sim_func, self.n_classes, self.
                    epsilon, self.use_trans, self.use_cred),
                lambda: tf.zeros([self.n_l, self.n_classes], dtype=tf.float32))
            self.new_q_l = tf.reshape(new_q_l, [self.n_l, self.n_classes],
                                      name='new_q_l')

        with tf.name_scope('f_g'):
            # Send samples through the classification projection head
            gx_ij = self.projection_head(bx_ij, self.g_dims, proj_type='G')
            gx_ij = tf.reshape(gx_ij, [self.ij_batch_size_ref, self.n_classes],
                               name='gx_ij')

        with tf.name_scope('output'):
            # Use softmax to get "probabilities"
            self.y_batch_score = tf.nn.softmax(gx_ij,
                                               axis=1,
                                               name='y_batch_score')
            # Take maximum to get class verdicts
            self.y_batch_pred = tf.argmax(gx_ij, axis=1, name='y_batch_pred')

        with tf.name_scope('losses'):
            # Soft contrastive loss on all data
            self.sc_losses = CCP_losses.get_sc_losses(self.zx_ij, self.q_ij,
                                                      self.sim_func, self.tau,
                                                      self.n_classes,
                                                      self.epsilon,
                                                      self.is_pretraining)
            # Select labeled (or pseudo-labeled) samples
            sel_q_ij = self.q_ij[self.selector_ij]
            sel_y_batch_score = self.y_batch_score[self.selector_ij]
            # Cross entropy loss on all data that shouldnt be treated as unlabeled
            self.ce_losses = CCP_losses.get_ce_losses(sel_y_batch_score,
                                                      sel_q_ij, self.epsilon)
            # Take the average of losses in the batch
            self.avg_ce_loss = tf.reduce_mean(self.ce_losses)
            self.avg_sc_loss = tf.reduce_mean(self.sc_losses)
            # Combine the losses into one
            self.comb_loss = self.avg_ce_loss + self.avg_sc_loss
            # Apply regularization
            self.l2_vars = [
                v for v in tf.compat.v1.get_collection(
                    tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES) if
                ('bias' not in v.name and 'batch_normalization' not in v.name)
            ]
            l2_loss = tf.reduce_sum([tf.nn.l2_loss(W) for W in self.l2_vars])
            self.ce_loss_reg = self.avg_ce_loss + (self.l2_lambda * l2_loss)
            self.sc_loss_reg = self.avg_sc_loss + (self.l2_lambda * l2_loss)
            self.comb_loss_reg = self.comb_loss + (self.l2_lambda * l2_loss)

    def get_trans(self, X):
        """
        Compute two transformed batches and concat them together.
        """
        # Compute two random transformations
        Xi = transformations.get_rnd_trans(X)
        Xj = transformations.get_rnd_trans(X)
        return tf.concat([Xi, Xj], axis=0)

    def f_b(self, inp, is_training):
        """
        Encoder network
        """
        # Send through wide resnet
        return CNN_ops.WRN(inp, is_training, self.activation_func,
                           self.drop_keep_prob, self.n_classes,
                           self.random_seed, self.cifar_version)

    def projection_head(self, inp, dims, proj_type):
        """
        Implements the contrastive and classification projection head
        """
        out = inp
        cur_dim = self.b_dim
        for i, new_dim in enumerate(dims):
            out = CNN_ops.dense_layer(
                out,
                cur_dim,
                new_dim,
                activation_func=self.activation_func
                if new_dim != dims[-1] else None,
                use_bias=True if
                (new_dim != dims[-1] or proj_type == 'G') else False,
                prefix='%s_%d_%d' % (proj_type, i, new_dim))
            cur_dim = new_dim
        return out

    def train(
        self,
        # All of the input images
        X,
        # The class labels
        y,
        # Class labels without noise perturbations. If the noise_ratio is
        # set to 0.0, this will be equal to y.
        y_clean,
        # Indicates which labels to keep, which to hide.
        y_mask,
        # The size of batches to be used in contrastive training
        contrastive_batch_size,
        # The size of batches to be used in classifier training
        cls_batch_size,
        # The minimum number of epochs that must be processed before
        # termination during classifier training.
        cls_min_epochs,
        # The number of epochs processed until an evaluation during
        # classifier training.
        cls_epochs_per_eval,
        # The maximum number of evaluations without improvement until
        # training is terminated during classifier training.
        cls_max_evals_since_overwrite,
        # Data used for evaluation.
        X_val=[],
        y_val=[],
        # The probability of a neuron not being dropped during training.
        drop_keep_prob=1.0,
        # Initial learning rate for the optimizer
        init_learning_rate=0.06,
        # Set this to False to skip running evals on the training data
        eval_on_train_data=True,
        # Set this to False to skip saving any batch results
        record_batch_evals=True,
        # Optional path to an NPZ file containing the desired initialization of Q vectors.
        init_q_fp=None,
        # Whether to use warmup
        do_warmup=True,
        # Alternatively, specify a path to a warmed up network checkpoint
        warmup_chk_path='',
        # Whether to alter the first CCP iteration learning rate
        alter_first_iter=True,
        # The EMA weight to accumulate the batch SSC loss
        sc_ema_weight=0.01,
        # Defines the termination condition for a CCP iteration
        sc_max_epochs_since_overwrite=10,
        # Defines the termination condition for wamrup
        wu_max_epochs_since_overwrite=40):

        self.eval_on_train_data = eval_on_train_data
        self.init_q_fp = init_q_fp
        self.base_learning_rate = init_learning_rate
        self.sc_ema_weight = sc_ema_weight
        self.sc_max_epochs_since_overwrite = sc_max_epochs_since_overwrite
        self.wu_max_epochs_since_overwrite = wu_max_epochs_since_overwrite
        assert (not (do_warmup and (warmup_chk_path != '')))
        if (warmup_chk_path != ''):
            assert (os.path.exists(os.path.dirname(warmup_chk_path)))
        with tf.Graph().as_default():
            session_conf = tf.compat.v1.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=CNN_ops.log_device_placement())
            session_conf.gpu_options.allow_growth = True
            sess = tf.compat.v1.Session(config=session_conf)
            with sess.as_default():
                self._build_CCP()  # Build the graph

                def reset_network():
                    """
                    """
                    print('--> Resetting network variables...')
                    sess.run(tf.compat.v1.global_variables_initializer())

                def reset_q_unlab():
                    """
                    Reset the Q vectors of all unlabeled data to \vec{0}
                    """
                    print('--> Resetting unlabeled entries in Q...')
                    new_q = np.copy(ccp_input.q_train)
                    new_q[ccp_input.orig_unlab_idxs, :] = 0.0
                    ccp_input.update_q_train(new_q)

                def calc_lr(cur_step_num):
                    """
                    Calculate a new learning rate based on the cosine learning
                    schedule.
                    """
                    if (ccp_input.use_unlab):
                        steps_per_epoch = ccp_input.steps_per_epoch
                    else:
                        steps_per_epoch = ccp_input.filt_steps_per_epoch
                    if (self.in_warmup):
                        tar_steps = (WU_LR_TAR_EPOCHS * steps_per_epoch)
                    elif (self.in_ccp):
                        tar_steps = (SC_LR_TAR_EPOCHS * steps_per_epoch)
                    elif (self.in_cls):
                        tar_steps = (CLS_LR_TAR_EPOCHS * steps_per_epoch)
                    else:
                        assert (False)
                    if (cur_step_num <= tar_steps):
                        return np.cos(
                            (np.pi * 7.0 * cur_step_num) /
                            (16.0 * tar_steps)) * self.base_learning_rate
                    return 0.000001

                def save_q_vecs(iter_n, q_vecs=None):
                    """
                    Save Q vectors to disk.
                    """
                    print('Saving Q vectors...')
                    q_fp = os.path.join(self.base_save_path, self.name,
                                        'q_vecs', 'iter_%d_q.npz' % iter_n)
                    if (not os.path.isdir(os.path.dirname(q_fp))):
                        os.makedirs(os.path.dirname(q_fp))
                    if (q_vecs is None):
                        np.savez_compressed(q_fp, q_vecs=ccp_input.q_train)
                    else:
                        np.savez_compressed(q_fp, q_vecs=q_vecs)
                    print('--> Saved Q vectors to %s' % q_fp)

                def scale_q_stren(q_vecs):
                    """
                    Scale Q vectors such that the maximum strength is 1.0.
                    """
                    print('Scaling Q vectors...')
                    if (self.clamp_labeled):
                        tar_q_stren = np.max(q_vecs[ccp_input.orig_unlab_idxs],
                                             axis=1)
                    else:
                        tar_q_stren = np.max(q_vecs, axis=1)
                    max_q = np.max(tar_q_stren)
                    if (max_q > 0.0):
                        print('--> New 1.0 point: %f' % max_q)
                        scaled_q = q_vecs / float(max_q)
                    else:
                        print('--> Q strength max is %f, No scaling' % max_q)
                        scaled_q = q_vecs
                    if (self.clamp_labeled):
                        # Return labeled data to original values
                        scaled_q[ccp_input.orig_y_mask == 1] = q_vecs[
                            ccp_input.orig_y_mask == 1]
                    return scaled_q

                def cred_adjust(q_vecs):
                    """
                    Implements the credibility adjustment in NumPy.
                    """
                    if (not self.use_cred):
                        print(
                            '\n*** WARNING: cred_adjust was called but use_cred=False!!***\n'
                        )
                    print('Doing credibility adjustment...')
                    n = q_vecs.shape[0]
                    n_classes = q_vecs.shape[1]
                    q_reshape = np.repeat(q_vecs[:, np.newaxis, :],
                                          n_classes,
                                          axis=1)  # <n, n_c, n_c>
                    idxs = np.logical_not(np.eye(n_classes,
                                                 dtype=bool))  # <n_c, n_c>
                    idxs_reshape = np.repeat(idxs[np.newaxis, :, :], n, axis=0)
                    q_reshape = np.reshape(q_reshape[idxs_reshape],
                                           (n, n_classes, n_classes - 1))
                    next_q = np.max(q_reshape, axis=2)  # <n, n_c>
                    cred_q = q_vecs - next_q
                    if (self.clamp_labeled):
                        # Return labeled data to original values
                        cred_q[ccp_input.orig_y_mask == 1] = q_vecs[
                            ccp_input.orig_y_mask == 1]
                    return cred_q

                def adjust_q_vecs(q_vecs, clip=True, scale=True, cred=True):
                    """
                    Clip, scale, and credibility adjust Q vectors.
                    """
                    if (self.use_cred):
                        if (scale):
                            q_vecs = scale_q_stren(q_vecs)
                        if (cred):
                            q_vecs = cred_adjust(q_vecs)
                        if (clip):
                            q_vecs = np.clip(q_vecs, 0.0, 1.0)
                    return q_vecs

                def get_keep_perc_w_kl_div(unclip_q_vecs, scale_q,
                                           last_keep_perc):
                    """
                    Calculate what percent of weakest pseudo-labels to reset.
                    Implements Algorithm 3 in the paper.
                    """
                    assert (self.max_kl_div >= 0.0)
                    unclip_q_stren = np.max(unclip_q_vecs, axis=1)
                    clip_q_vecs = adjust_q_vecs(unclip_q_vecs,
                                                clip=True,
                                                scale=scale_q,
                                                cred=scale_q)
                    if (self.clamp_labeled):
                        prop_unclip_q_stren = unclip_q_stren[
                            ccp_input.orig_unlab_idxs]
                        prop_clip_q_vecs = clip_q_vecs[
                            ccp_input.orig_unlab_idxs]
                    else:
                        prop_unclip_q_stren = unclip_q_stren
                        prop_clip_q_vecs = clip_q_vecs
                    print('--> Computing KL Divergences...')
                    # Get the full mass distribution first
                    anchor_dist = np.sum(prop_clip_q_vecs,
                                         axis=0) / np.sum(prop_clip_q_vecs)
                    # Compute mass distributions at candidate keep percentages (and KL divergence)
                    min_keep_perc = np.maximum(self.abs_min_keep_perc,
                                               last_keep_perc + 0.01)
                    keep_perc_candidates = np.array([
                        j / 100.0
                        for j in range(int(min_keep_perc * 100.0), 101)
                    ])
                    print('----> Using candidates between %0.2f and %0.2f' %
                          (np.min(keep_perc_candidates),
                           np.max(keep_perc_candidates)))
                    all_kl_div = np.zeros([len(keep_perc_candidates)],
                                          dtype='float32')
                    for i, perc in enumerate(keep_perc_candidates):
                        k = int(
                            np.round(len(prop_unclip_q_stren) * (1.0 - perc)))
                        new_idxs = np.argpartition(prop_unclip_q_stren, kth=k)
                        prop_y_mask = np.ones_like(prop_unclip_q_stren)
                        prop_y_mask[new_idxs[:k]] = 0
                        if (self.clamp_labeled):
                            new_y_mask = np.copy(ccp_input.orig_y_mask)
                            new_y_mask[ccp_input.orig_unlab_idxs] = prop_y_mask
                            sel_idxs = np.logical_and(
                                ccp_input.orig_unlab_idxs, new_y_mask == 1)
                        else:
                            new_y_mask = np.copy(prop_y_mask)
                            sel_idxs = new_y_mask == 1
                        sel_clip_q_vecs = clip_q_vecs[sel_idxs]
                        new_dist = np.sum(sel_clip_q_vecs,
                                          axis=0) / np.sum(sel_clip_q_vecs)
                        all_kl_div[i] = utils.shannon_entropy(
                            new_dist, anchor_dist, 2.0)
                    # Choose the minimum keep percentage that obeys the maximum KL divergence
                    chosen_idx = -np.max((all_kl_div <= self.max_kl_div) *
                                         np.arange(len(all_kl_div), 0, -1))
                    chosen_keep_perc_candidate = keep_perc_candidates[
                        chosen_idx]
                    print(
                        '----> Chosen percentage of strongest samples to keep: %0.5f'
                        % chosen_keep_perc_candidate)
                    print('------> Associated KL Divergence: %0.5f' %
                          all_kl_div[chosen_idx])
                    return prop_unclip_q_stren, chosen_keep_perc_candidate

                def update_sc_ema(new_sc_loss):
                    """
                    Update our EMA of soft contrastive loss with a new value
                    and check for an overwrite if a new epoch is being
                    processed.
                    """
                    if (self.sc_ema is None):
                        self.sc_ema = new_sc_loss
                    else:
                        self.sc_ema = (new_sc_loss * self.sc_ema_weight) + (
                            self.sc_ema * (1.0 - self.sc_ema_weight))
                    # Check best EMA overwrite
                    if (ccp_input.new_epoch):
                        if ((self.sc_ema - self.best_sc_ema) < -0.001):
                            print('Best SC loss EMA overwritten! %f --> %f' %
                                  (self.best_sc_ema, self.sc_ema))
                            self.best_sc_ema = self.sc_ema
                            self.n_epochs_since_overwrite = 0
                        else:
                            self.n_epochs_since_overwrite += 1
                            print('Number of epochs without improvement: %d' %
                                  self.n_epochs_since_overwrite)

                def train_step(batch, batch_summary_file):
                    """
                    Implements the operations needed to proceed with warmup,
                    CCP, or classifier training depending what stage the
                    training session is in.
                    """
                    new_step_num = tf.compat.v1.train.global_step(
                        sess, global_step)
                    feed_dict = {
                        self.X_batch: batch['X'],
                        self.q_batch: batch['q'],
                        self.selector: batch['selector'],
                        self.is_training: True,
                        self.drop_keep_prob: drop_keep_prob,
                        self.lr: calc_lr(new_step_num)
                    }

                    ce_loss = -1.0  # Cross-entropy loss
                    sc_loss = -1.0  # Soft contrastive loss
                    batch_acc = -1.0  # Accuracy of the batch predictions
                    prop_acc = -1.0  # Accuracy of pseudo-labels assigned to unlabeled data
                    avg_corr_prop_stren = -1.0  # Average strength of correct pseudo-labels assigned to unlabeled data
                    avg_incorr_prop_stren = -1.0  # Average strength of incorrect pseudo-labels assigned to unlabeled data
                    avg_true_unlab_prop_stren = -1.0  # Average strength of pseudo-labels among samples where the true class is not known
                    avg_l_corr_prop_stren = -1.0  # Average strength of correct pseudo-labels assigned to labeled data
                    avg_l_incorr_prop_stren = -1.0  # Average strength of incorrect pseudo-labels assigned to labeled data
                    l_prop_acc = -1.0  # Accuracy of pseudo-labels assigned to labeled data
                    early_exit = False  # Flag to exit the training session

                    ###########################################################
                    # PROCEED WITH WARMUP
                    ###########################################################

                    if (self.in_warmup):

                        def save_warmup():
                            """
                            """
                            resources['warmup_chk_path'] = resources[
                                'saver'].save(sess,
                                              resources['model_fp_warmup'])
                            print('--> Saved new warmup model to %s' %
                                  resources['warmup_chk_path'])

                        feed_dict[self.use_trans] = True
                        feed_dict[self.is_pretraining] = True
                        _, sc_loss = sess.run(
                            [minimize_op_sc, self.avg_sc_loss], feed_dict)
                        update_sc_ema(sc_loss)

                        if (new_step_num % STEPS_PER_PRINT == 0):
                            print('%s: step %d, epoch %d, avg_sc_loss %.5f' %
                                  (utils.datetime_str(), new_step_num,
                                   ccp_input.num_epochs, sc_loss))
                        if (ccp_input.new_epoch):
                            # Save state of model weights
                            save_warmup()

                        # Check for exit of warmup
                        if (self.n_epochs_since_overwrite
                                > self.wu_max_epochs_since_overwrite
                                or (DEBUG_MODE and ccp_input.num_epochs > 2)):
                            print('\nExiting warmup phase...')
                            save_warmup()
                            print('--> Saved new warmup model to %s' %
                                  resources['warmup_chk_path'])
                            if (self.n_iters > 0):
                                assert (int(
                                    np.sum(1.0 - ccp_input.orig_y_mask)) > 0
                                        or not self.clamp_labeled)
                                print(
                                    '\n---- Starting CCP iterations... ----\n')
                                self.in_warmup = False
                                self.in_ccp = True
                                self.cur_iter = 0
                                self.in_cls = False
                                self.sc_ema = None
                                self.best_sc_ema = np.inf
                                self.n_epochs_since_overwrite = 0
                                if (alter_first_iter):
                                    print(
                                        '--> Altering the first iteration...')
                                    self.base_learning_rate = ALTER_FIRST_ITER_LR
                                    ccp_input.balance_train_batches = True
                                else:
                                    self.base_learning_rate = init_learning_rate
                                    ccp_input.balance_train_batches = True
                                ccp_input.update_y_mask(
                                    np.copy(ccp_input.orig_y_mask))
                            elif (cls_epochs_per_eval > 0):
                                print(
                                    '\n---- Beginning classifier training... ----\n'
                                )
                                self.in_warmup = False
                                self.in_ccp = False
                                self.in_cls = True
                                ccp_input.balance_train_batches = False
                                ccp_input.use_unlab = False
                                ccp_input.train_batch_size = cls_batch_size
                            else:
                                self.in_warmup = False
                                self.in_ccp = False
                                self.in_cls = False
                                early_exit = True
                            sess.run(global_step.assign(0))
                            ccp_input.reset()
                            new_step_num = 0

                    ###########################################################
                    # PROCEED WITH CCP ITERATIONS
                    ###########################################################

                    elif (self.in_ccp):
                        feed_dict[self.use_trans] = self.trans_in_ccp
                        feed_dict[self.is_pretraining] = False

                        # Directly minimize loss...no need to track EMA of model weights
                        if (not self.clamp_labeled):
                            _, sc_loss, q_u, q_l = sess.run([
                                minimize_op_sc, self.avg_sc_loss, self.new_q_u,
                                self.new_q_l
                            ], feed_dict)
                        else:
                            _, sc_loss, q_u = sess.run([
                                minimize_op_sc, self.avg_sc_loss, self.new_q_u
                            ], feed_dict)
                            q_l = batch['q'][batch['selector']]
                        update_sc_ema(sc_loss)

                        if (np.sum(np.logical_not(batch['selector'])) > 0):
                            avg_corr_prop_stren, avg_incorr_prop_stren = -1.0, -1.0
                            # Calculate propagation performance to unlabeled data (that are known)
                            u_y_clean = batch['y_clean'][np.logical_not(
                                batch['selector'])]
                            known_idxs = u_y_clean != -1
                            known_y_prop = np.argmax(q_u[known_idxs], axis=1)
                            known_prop_stren = np.max(q_u[known_idxs], axis=1)
                            true_unlab_prop_stren = np.max(
                                q_u[np.logical_not(known_idxs)], axis=1)
                            known_u_y_clean = u_y_clean[known_idxs]
                            # Get average confidence of correct and incorrect propagations
                            corr_prop_stren = known_prop_stren[known_u_y_clean
                                                               == known_y_prop]
                            if (len(corr_prop_stren) > 0):
                                avg_corr_prop_stren = np.mean(corr_prop_stren)
                            incorr_prop_stren = known_prop_stren[
                                known_u_y_clean != known_y_prop]
                            if (len(incorr_prop_stren) > 0):
                                avg_incorr_prop_stren = np.mean(
                                    incorr_prop_stren)
                            if (len(true_unlab_prop_stren) > 0):
                                avg_true_unlab_prop_stren = np.mean(
                                    true_unlab_prop_stren)
                            # Get overall propagation accuracy
                            if (len(known_y_prop) > 0):
                                prop_acc = np.sum(
                                    known_u_y_clean == known_y_prop) / float(
                                        len(known_y_prop))
                            else:
                                prop_acc = -1.0
                        if (not self.clamp_labeled):
                            # Calculate performance of propagation to labeled data
                            l_y_clean = batch['y_clean'][batch['selector']]
                            assert (np.sum(l_y_clean == -1) == 0)
                            y_prop = np.argmax(q_l, axis=1)
                            prop_stren = np.max(q_l, axis=1)
                            # Get average confidence of correct and incorrect propagations
                            corr_prop_stren = prop_stren[l_y_clean == y_prop]
                            if (len(corr_prop_stren) > 0):
                                avg_l_corr_prop_stren = np.mean(
                                    corr_prop_stren)
                            incorr_prop_stren = prop_stren[l_y_clean != y_prop]
                            if (len(incorr_prop_stren) > 0):
                                avg_l_incorr_prop_stren = np.mean(
                                    incorr_prop_stren)
                            # Get overall propagation accuracy
                            if (len(y_prop) > 0):
                                l_prop_acc = np.sum(
                                    l_y_clean == y_prop) / float(len(y_prop))
                            else:
                                l_prop_acc = -1.0

                        # Update running averages
                        l_idx, u_idx = 0, 0
                        for idx, select in zip(batch['epoch_idxs'],
                                               batch['selector']):
                            if (select):
                                self.q_running_avgs['counts'][idx] += 1
                                self.q_running_avgs['q_vec_sums'][idx] += q_l[
                                    l_idx]
                                l_idx += 1
                            else:
                                self.q_running_avgs['counts'][idx] += 1
                                self.q_running_avgs['q_vec_sums'][idx] += q_u[
                                    u_idx]
                                u_idx += 1

                        if (new_step_num % STEPS_PER_PRINT == 0):
                            msg = '%s: step %d, epoch %d, sc_loss %.5f, ce_loss %.5f, corr_prop_stren %.5f, incorr_prop_stren %.5f, prop_acc %.5f' % (
                                utils.datetime_str(), new_step_num,
                                ccp_input.num_epochs, sc_loss, ce_loss,
                                avg_corr_prop_stren, avg_incorr_prop_stren,
                                prop_acc)
                            if (avg_true_unlab_prop_stren > -1.0):
                                msg += ', true_unlab_prop_stren %.5f' % avg_true_unlab_prop_stren
                            if (not self.clamp_labeled):
                                msg += ', l_corr_prop_stren %.5f, l_incorr_prop_stren %.5f, l_prop_acc %.5f' % (
                                    avg_l_corr_prop_stren,
                                    avg_l_incorr_prop_stren, l_prop_acc)
                            print(msg)

                        # Check for exit of this iteration
                        exit_iter = False
                        if (self.ccp_iter_schedule[self.cur_iter] > 0
                                and ccp_input.num_epochs
                                > self.ccp_iter_schedule[self.cur_iter]):
                            exit_iter = True  # If we have a set number of epochs that we've exceeded
                        elif (self.ccp_iter_schedule[self.cur_iter] < 0
                              and self.n_epochs_since_overwrite
                              > self.sc_max_epochs_since_overwrite):
                            # If we are going by loss convergence and the loss has converged
                            exit_iter = True
                        if (exit_iter
                                or (DEBUG_MODE and ccp_input.num_epochs > 2)):
                            self.cur_iter += 1
                            print('\nFinished CCP iteration %d/%d!' %
                                  (self.cur_iter, self.n_iters))
                            lens = np.array(self.q_running_avgs['counts'])
                            for j in range(ccp_input.n_train):
                                if (lens[j] == 0):
                                    if (not DEBUG_MODE):
                                        print(
                                            'WARNING: Index %d has no pseudo-labels! Using zero vector.'
                                            % j)
                                    lens[j] += 1
                            if (self.clamp_labeled):
                                max_len = np.max(
                                    lens[ccp_input.orig_unlab_idxs])
                                min_len = np.min(
                                    lens[ccp_input.orig_unlab_idxs])
                            else:
                                max_len = np.max(lens)
                                min_len = np.min(lens)
                            print(
                                '--> Minimum amount of pseudo-labels to average over: %d'
                                % min_len)
                            print(
                                '--> Maximum amount of pseudo-labels to average over: %d'
                                % max_len)
                            print('--> Computing average pseudo-labels...')
                            avg_q = np.array(
                                [(self.q_running_avgs['q_vec_sums'][i] /
                                  float(self.q_running_avgs['counts'][i]))
                                 for i in range(ccp_input.n_train)],
                                dtype=np.float32)
                            assert (avg_q.shape[0] == ccp_input.n_train)
                            assert (avg_q.shape[1] == self.n_classes)
                            ccp_input.update_q_train(avg_q)
                            save_q_vecs(self.cur_iter)
                            assert (
                                avg_q.shape[0] == ccp_input.q_train.shape[0])
                            ccp_input.update_q_train(
                                adjust_q_vecs(ccp_input.q_train,
                                              clip=True,
                                              scale=self.scale_q,
                                              cred=self.scale_q))

                            print('----> New overall prop accuracy: %f' %
                                  ccp_input.calc_prop_acc())
                            print('----> New overall prop error: %f' %
                                  ccp_input.calc_prop_err())
                            if (not self.clamp_labeled):
                                print(
                                    '------> New overall labeled data prop accuracy: %f'
                                    % ccp_input.calc_l_prop_acc())
                                print(
                                    '------> New overall labeled data prop error: %f'
                                    % ccp_input.calc_l_prop_err())
                            # Get strength of correct/incorrect propagation on unlabeled data
                            known_idxs = ccp_input.y_clean != -1
                            y_1d = np.argmax(ccp_input.q_train, axis=1)
                            corr_idxs = ccp_input.y_clean == y_1d
                            incorr_idxs = ccp_input.y_clean != y_1d
                            corr_orig_unlab_idxs = np.logical_and(
                                ccp_input.orig_unlab_idxs, corr_idxs)
                            incorr_orig_unlab_idxs = np.logical_and(
                                ccp_input.orig_unlab_idxs, incorr_idxs)
                            corr_prop_q = ccp_input.q_train[np.logical_and(
                                corr_orig_unlab_idxs, known_idxs)]
                            incorr_prop_q = ccp_input.q_train[np.logical_and(
                                incorr_orig_unlab_idxs, known_idxs)]
                            corr_prop_q_stren = np.max(corr_prop_q, axis=1)
                            incorr_prop_q_stren = np.max(incorr_prop_q, axis=1)
                            if (len(corr_prop_q_stren) == 0):
                                corr_prop_q_stren = [-1.0]
                            if (len(incorr_prop_q_stren) == 0):
                                incorr_prop_q_stren = [-1.0]
                            unknown_stren = ccp_input.calc_avg_unknown_stren()
                            print(
                                '----> Average strength of correct propagation: %f'
                                % np.mean(corr_prop_q_stren))
                            print(
                                '----> Average strength of incorrect propagation: %f'
                                % np.mean(incorr_prop_q_stren))
                            print(
                                '----> Average strength of unknown samples: %f'
                                % unknown_stren)
                            if (not self.clamp_labeled):
                                # Get strength of correct/incorrect propagation on labeled data
                                l_corr_lab_idxs = np.logical_and(
                                    np.logical_not(ccp_input.orig_unlab_idxs),
                                    corr_idxs)
                                l_incorr_lab_idxs = np.logical_and(
                                    np.logical_not(ccp_input.orig_unlab_idxs),
                                    incorr_idxs)
                                l_corr_prop_q = ccp_input.q_train[
                                    l_corr_lab_idxs]
                                l_incorr_prop_q = ccp_input.q_train[
                                    l_incorr_lab_idxs]
                                l_corr_prop_q_stren = np.max(l_corr_prop_q,
                                                             axis=1)
                                l_incorr_prop_q_stren = np.max(l_incorr_prop_q,
                                                               axis=1)
                                if (len(l_corr_prop_q_stren) == 0):
                                    l_corr_prop_q_stren = [-1.0]
                                if (len(l_incorr_prop_q_stren) == 0):
                                    l_incorr_prop_q_stren = [-1.0]
                                print(
                                    '------> Average strength of labeled correct propagation: %f'
                                    % np.mean(l_corr_prop_q_stren))
                                print(
                                    '------> Average strength of labeled incorrect propagation: %f'
                                    % np.mean(l_incorr_prop_q_stren))
                                self.ccp_perf_history.append([
                                    ccp_input.calc_l_prop_acc(),
                                    ccp_input.calc_l_prop_err(),
                                    np.mean(l_corr_prop_q_stren),
                                    np.mean(l_incorr_prop_q_stren),
                                    ccp_input.calc_prop_acc(),
                                    ccp_input.calc_prop_err(),
                                    np.mean(corr_prop_q_stren),
                                    np.mean(incorr_prop_q_stren), new_step_num
                                ])
                            else:
                                self.ccp_perf_history.append([
                                    ccp_input.calc_prop_acc(),
                                    ccp_input.calc_prop_err(),
                                    np.mean(corr_prop_q_stren),
                                    np.mean(incorr_prop_q_stren), new_step_num
                                ])
                            if (unknown_stren >= 0.0):
                                self.ccp_perf_history[-1].append(unknown_stren)
                            print(
                                '------> Copy/Paste friendly performance history:\n'
                            )
                            for row in self.ccp_perf_history:
                                print(','.join([str(e) for e in row]))
                            print(
                                '\nDeciding which unlabeled samples to reset via subsampling...'
                            )
                            if (self.max_kl_div > 0.0
                                    and self.last_keep_perc < 0.99):
                                prop_unclip_q_stren, chosen_keep_perc_candidate = get_keep_perc_w_kl_div(
                                    avg_q, self.scale_q, self.last_keep_perc)
                                # Reset Q with chosen keep percentage
                                k = int(
                                    np.round(
                                        len(prop_unclip_q_stren) *
                                        (1.0 - chosen_keep_perc_candidate)))
                                new_idxs = np.argpartition(
                                    prop_unclip_q_stren,
                                    kth=min(k,
                                            len(prop_unclip_q_stren) - 1))
                                prop_mask = np.ones_like(prop_unclip_q_stren)
                                if (self.clamp_labeled):
                                    prop_q = np.copy(ccp_input.q_train[
                                        ccp_input.orig_unlab_idxs])
                                else:
                                    prop_q = np.copy(ccp_input.q_train)
                                prop_mask[new_idxs[:k]] = 0
                                prop_q[new_idxs[:k], :] = 0.0
                                if (self.clamp_labeled):
                                    new_q = np.copy(ccp_input.q_train)
                                    new_q[
                                        ccp_input.orig_unlab_idxs, :] = prop_q
                                    new_mask = np.copy(ccp_input.orig_y_mask)
                                    new_mask[
                                        ccp_input.orig_unlab_idxs] = prop_mask
                                else:
                                    new_q = prop_q
                                    new_mask = prop_mask
                                self.last_keep_perc = chosen_keep_perc_candidate
                                self.max_kl_div = self.max_kl_div / MAX_KL_DIV_DIVIDER
                            else:
                                print('--> Not resetting any Q vectors...')
                                new_mask = np.ones_like(ccp_input.orig_y_mask)
                                new_q = np.copy(ccp_input.q_train)

                            if (self.clamp_labeled):
                                kept_idxs = np.logical_and(
                                    ccp_input.orig_y_mask == 0, new_mask == 1)
                                unkept_idxs = np.logical_and(
                                    ccp_input.orig_y_mask == 0, new_mask == 0)
                            else:
                                kept_idxs = new_mask == 1
                                unkept_idxs = new_mask == 0
                            print('--> Number of propagations kept: %d' %
                                  np.sum(kept_idxs))
                            print('--> Number of propagations reset: %d' %
                                  np.sum(unkept_idxs))
                            ccp_input.update_q_train(new_q)
                            print('--> Acc among kept propagations: %f' %
                                  ccp_input.calc_acc(kept_idxs))
                            print(
                                '--> Average error among kept propagations: %f'
                                % ccp_input.calc_avg_err(kept_idxs))
                            print(
                                '--> Average strength of unknown samples: %f' %
                                ccp_input.calc_avg_unknown_stren())

                            # Check for exit of CCP
                            if (self.cur_iter >= self.n_iters):
                                if (cls_epochs_per_eval > 0):
                                    print(
                                        '\n---- Beginning classifier training... ----\n'
                                    )
                                    self.in_warmup = False
                                    self.in_ccp = False
                                    self.in_cls = True
                                    ccp_input.balance_train_batches = False
                                    ccp_input.update_y_mask(
                                        np.ones_like(ccp_input.orig_y_mask))
                                    ccp_input.use_unlab = False
                                    ccp_input.train_batch_size = cls_batch_size
                                    sess.run(global_step.assign(0))
                                    self.base_learning_rate = init_learning_rate
                                else:
                                    self.in_warmup = False
                                    self.in_ccp = False
                                    self.in_cls = False
                                    early_exit = True
                            else:
                                print(
                                    '\n---- Starting next iteration... ----\n')
                                self.q_running_avgs = {
                                    'q_vec_sums': [
                                        np.zeros([self.n_classes],
                                                 dtype=np.float32)
                                        for i in range(ccp_input.n_train)
                                    ],
                                    'counts':
                                    [0 for i in range(ccp_input.n_train)]
                                }
                                self.in_warmup = False
                                self.in_ccp = True
                                self.in_cls = False
                                self.sc_ema = None
                                self.best_sc_ema = np.inf
                                self.n_epochs_since_overwrite = 0
                                self.base_learning_rate = init_learning_rate
                                ccp_input.balance_train_batches = True
                                reset_network()
                            if (do_warmup > 0 or warmup_chk_path != ''):
                                print(
                                    '--> Loading weights of warmed network...')
                                resources['saver'].restore(
                                    sess, resources['warmup_chk_path'])
                            ccp_input.reset()
                            new_step_num = 0

                    ###########################################################
                    # PROCEED WITH CLASSIFIER TRAINING
                    ###########################################################

                    elif (self.in_cls):
                        # In classifier building phase (with static label vectors)
                        feed_dict[self.use_trans] = self.trans_in_cls
                        feed_dict[self.is_pretraining] = False
                        if (self.use_comb_loss):
                            _, sc_loss, ce_loss, y_pred = sess.run([
                                train_op_comb, self.avg_sc_loss,
                                self.avg_ce_loss, self.y_batch_pred
                            ], feed_dict)
                        else:
                            _, ce_loss, y_pred = sess.run([
                                train_op_ce, self.avg_ce_loss,
                                self.y_batch_pred
                            ], feed_dict)
                        known_idxs = batch['y_clean'] != -1
                        known_y_true = batch['y_clean'][known_idxs]
                        if (self.trans_in_cls):
                            known_y_true = np.concatenate(
                                (known_y_true, known_y_true), axis=0)
                            known_y_pred = y_pred[np.concatenate(
                                (known_idxs, known_idxs), axis=0)]
                        else:
                            known_y_pred = y_pred[known_idxs]
                        if (len(known_y_pred) > 0):
                            batch_acc = (known_y_true
                                         == known_y_pred).sum() / float(
                                             len(known_y_pred))
                        else:
                            batch_acc = -1.0
                        if (new_step_num % STEPS_PER_PRINT == 0):
                            print(
                                '%s: step %d, epoch %d, avg_sc_loss %.5f, avg_ce_loss %.5f, batch_acc %.5f'
                                % (utils.datetime_str(), new_step_num,
                                   ccp_input.num_epochs, sc_loss, ce_loss,
                                   batch_acc))
                    else:
                        early_exit = True

                    if record_batch_evals:
                        CNN_ops.record_result(batch_summary_file, [
                            new_step_num, ccp_input.num_epochs, ce_loss,
                            sc_loss, avg_corr_prop_stren,
                            avg_incorr_prop_stren, prop_acc, batch_acc
                        ])
                    return new_step_num, self.in_cls, early_exit

                def test_step(cur_step_num, batch_iter):
                    """
                    Do an evaluation (used during classifier training).
                    """
                    all_ce_losses = []
                    y_true, y_score = [], []
                    # Starting processing test input
                    for batch in batch_iter:
                        feed_dict = {
                            self.X_batch: batch['X'],
                            self.q_batch: ccp_input.one_hot(batch['y']),
                            self.selector: batch['selector'],
                            self.is_training: False,
                            self.use_trans: False,
                            self.drop_keep_prob: 1.0
                        }
                        # Calculate results
                        ys, ce_losses = sess.run(
                            [self.y_batch_score, self.ce_losses], feed_dict)
                        # Collect results
                        all_ce_losses.extend(ce_losses.tolist())
                        y_true.extend(batch['y'].tolist())
                        y_score.extend(ys.tolist())
                    # Convert to numpy arrays
                    all_ce_losses = np.array(all_ce_losses)
                    y_true = np.array(y_true)
                    y_score = np.array(y_score)
                    # Aggregate final results
                    result = {}
                    # Get average loss
                    result['avg_loss'] = np.mean(all_ce_losses)
                    # Compute verdicts
                    y_pred = np.argmax(y_score, axis=1)
                    # Compute accuracy
                    result['acc'] = (y_true == y_pred).sum() / len(y_pred)
                    # Print summarized results
                    print(
                        '{}: step {}, epoch {}, ce_loss {:g}, acc {:g}'.format(
                            utils.datetime_str(), cur_step_num,
                            ccp_input.num_epochs, result['avg_loss'],
                            result['acc']))
                    # Generate and print confusion matrix
                    result['opt_acc_cm'] = confusion_matrix(
                        y_true,
                        y_pred,
                        labels=[x for x in range(self.n_classes)])
                    print('Confusion Matrix:')
                    utils.print_cm(result['opt_acc_cm'],
                                   self.class_labels,
                                   normalize=False)
                    print('Confusion Matrix Normalized:')
                    utils.print_cm(result['opt_acc_cm'], self.class_labels)
                    return result

                # Feed input data into feeder class instance
                ccp_input = CNN_Input(X,
                                      y=y,
                                      y_clean=y_clean,
                                      y_mask=y_mask,
                                      X_val=X_val,
                                      y_val=y_val,
                                      random_seed=self.random_seed,
                                      n_classes=self.n_classes,
                                      use_unlab=True,
                                      train_batch_size=contrastive_batch_size,
                                      balance_train_batches=True)

                # Optionally load saved Q vectors
                if (self.init_q_fp is not None):
                    print('Loading saved Q vectors from %s...' %
                          self.init_q_fp)
                    q_vecs = np.load(self.init_q_fp,
                                     mmap_mode=None,
                                     allow_pickle=True)['q_vecs']
                    print('--> Loaded Q vectors!')
                    ccp_input.update_q_train(
                        adjust_q_vecs(q_vecs,
                                      clip=True,
                                      scale=self.scale_q,
                                      cred=self.scale_q))
                    print('----> Overall prop accuracy: %f' %
                          ccp_input.calc_prop_acc())
                    print('----> Overall prop error: %f' %
                          ccp_input.calc_prop_err())
                    print('----> Labeled data accuracy: %f' %
                          ccp_input.calc_l_prop_acc())
                    print('----> Labeled data error: %f' %
                          ccp_input.calc_l_prop_err())
                    print('----> Average strength of unknown samples: %f' %
                          ccp_input.calc_avg_unknown_stren())
                    if (self.max_kl_div > 0.0):
                        print('Doing subsampling via KL Divergence...')
                        prop_unclip_q_stren, chosen_keep_perc_candidate = get_keep_perc_w_kl_div(
                            q_vecs, self.scale_q, self.last_keep_perc)
                        # Reset Q with chosen keep percentage
                        k = int(
                            np.round(
                                len(prop_unclip_q_stren) *
                                (1.0 - chosen_keep_perc_candidate)))
                        new_idxs = np.argpartition(
                            prop_unclip_q_stren,
                            kth=min(k,
                                    len(prop_unclip_q_stren) - 1))
                        prop_mask = np.ones_like(prop_unclip_q_stren)
                        if (self.clamp_labeled):
                            prop_q = np.copy(
                                ccp_input.q_train[ccp_input.orig_unlab_idxs])
                        else:
                            prop_q = np.copy(ccp_input.q_train)
                        prop_mask[new_idxs[:k]] = 0
                        prop_q[new_idxs[:k], :] = 0.0
                        if (self.clamp_labeled):
                            new_q = np.copy(ccp_input.q_train)
                            new_q[ccp_input.orig_unlab_idxs, :] = prop_q
                            new_mask = np.copy(ccp_input.orig_y_mask)
                            new_mask[ccp_input.orig_unlab_idxs] = prop_mask
                        else:
                            new_q = prop_q
                            new_mask = prop_mask
                        self.last_keep_perc = chosen_keep_perc_candidate
                    else:
                        print('Not resetting any Q vectors...')
                        new_mask = np.ones_like(ccp_input.orig_y_mask)
                        new_q = np.copy(ccp_input.q_train)

                    kept_idxs = np.logical_and(ccp_input.orig_y_mask == 0,
                                               new_mask == 1)
                    unkept_idxs = np.logical_and(ccp_input.orig_y_mask == 0,
                                                 new_mask == 0)
                    print('--> Number of propagations kept: %d' %
                          np.sum(kept_idxs))
                    print('--> Number of propagations reset: %d' %
                          np.sum(unkept_idxs))
                    ccp_input.update_q_train(new_q)
                    print('--> New selected prop accuracy: %f' %
                          ccp_input.calc_acc(kept_idxs))
                    print('--> New selected prop error: %f' %
                          ccp_input.calc_avg_err(kept_idxs))
                    print('----> Average strength of unknown samples: %f' %
                          ccp_input.calc_avg_unknown_stren())
                else:
                    reset_q_unlab()
                    new_mask = np.copy(ccp_input.orig_y_mask)
                ccp_input.update_y_mask(new_mask)

                # Make sure all our output directories are set up
                resources = CNN_ops._define_output_dirs(
                    self.base_save_path, self.name)

                # Decide what kind of training we should start
                self.in_warmup = False
                self.in_ccp = False
                self.in_cls = False
                self.sc_ema = None
                self.best_sc_ema = np.inf
                self.n_epochs_since_overwrite = 0
                self.ccp_perf_history = [
                ]  # For printing results at the end of training
                if (do_warmup):
                    print('--> Starting warmup...')
                    self.in_warmup = True
                    self.in_ccp = False
                    self.in_cls = False
                    ccp_input.use_unlab = True
                    ccp_input.balance_train_batches = False
                elif (self.n_iters > 0):
                    assert (int(np.sum(1.0 - ccp_input.orig_y_mask)) > 0
                            or not self.clamp_labeled)
                    print('--> Starting CCP iterations...')
                    self.in_warmup = False
                    self.in_ccp = True
                    self.in_cls = False
                    self.cur_iter = 0
                    ccp_input.use_unlab = True
                    if (alter_first_iter):
                        print('--> Altering the first iteration...')
                        self.base_learning_rate = ALTER_FIRST_ITER_LR
                        ccp_input.balance_train_batches = True
                    else:
                        self.base_learning_rate = init_learning_rate
                        ccp_input.balance_train_batches = True
                    ccp_input.update_y_mask(np.copy(ccp_input.orig_y_mask))
                elif (cls_epochs_per_eval > 0):
                    print('--> Starting classifier training...')
                    self.in_warmup = False
                    self.in_ccp = False
                    self.in_cls = True
                    ccp_input.use_unlab = False
                    ccp_input.train_batch_size = cls_batch_size
                    ccp_input.balance_train_batches = False
                else:
                    assert (False)
                # Maintain the running average of Q vectors for each sample during CCP iterations
                self.q_running_avgs = {
                    'q_vec_sums': [
                        np.zeros([self.n_classes], dtype=np.float32)
                        for i in range(ccp_input.n_train)
                    ],
                    'counts': [0 for i in range(ccp_input.n_train)]
                }
                ccp_input.reset()

                # Define training procedures
                global_step = tf.Variable(0,
                                          name='global_step',
                                          trainable=False)
                optimizer = tf.compat.v1.train.MomentumOptimizer(
                    learning_rate=self.lr, momentum=0.9, use_nesterov=True)

                update_ops = tf.compat.v1.get_collection(
                    tf.compat.v1.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    # SC loss (Used in warmup and CCP -- no need to track EMA of weights)
                    minimize_op_sc = optimizer.minimize(
                        self.sc_loss_reg, global_step=global_step)
                    # CE loss
                    minimize_op_ce = optimizer.minimize(
                        self.ce_loss_reg, global_step=global_step)
                    # Combined loss
                    minimize_op_comb = optimizer.minimize(
                        self.comb_loss_reg, global_step=global_step)
                # Implement EMA of model weights (for warmup and when training a classifier)
                self.model_vars = tf.compat.v1.get_collection(
                    tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
                # Make EMA object and update internal variables after optimization step
                self.ema_object = tf.train.ExponentialMovingAverage(
                    decay=0.999)
                with tf.control_dependencies([minimize_op_ce]):
                    train_op_ce = self.ema_object.apply(self.model_vars)
                with tf.control_dependencies([minimize_op_comb]):
                    train_op_comb = self.ema_object.apply(self.model_vars)
                # Op to make backup variables
                with tf.compat.v1.variable_scope('BackupVariables'):
                    backup_vars = [
                        tf.compat.v1.get_variable(
                            var.op.name,
                            dtype=var.value().dtype,
                            trainable=False,
                            initializer=var.initialized_value())
                        for var in self.model_vars
                    ]

                def ema_to_weights():
                    return tf.group(*(tf.compat.v1.assign(
                        var,
                        self.ema_object.average(var).read_value())
                                      for var in self.model_vars))

                def save_weight_backups():
                    return tf.group(
                        *(tf.compat.v1.assign(bck, var.read_value())
                          for var, bck in zip(self.model_vars, backup_vars)))

                def restore_weight_backups():
                    return tf.group(
                        *(tf.compat.v1.assign(var, bck.read_value())
                          for var, bck in zip(self.model_vars, backup_vars)))

                def to_training():
                    print('--> To training weights...')
                    return restore_weight_backups()

                def to_ema():
                    print('--> To EMA weights...')
                    with tf.control_dependencies([save_weight_backups()]):
                        return ema_to_weights()

                # Initialize all variables
                sess.run(tf.compat.v1.global_variables_initializer())

                if (warmup_chk_path != ''):
                    print('--> Loading weights of warmed network...')
                    resources['warmup_chk_path'] = warmup_chk_path
                    resources['saver'].restore(sess,
                                               resources['warmup_chk_path'])

                # Enter the training loop
                result = CNN_ops.training_loop(train_step,
                                               test_step,
                                               to_training,
                                               to_ema,
                                               ccp_input,
                                               eval_on_train_data,
                                               cls_min_epochs,
                                               cls_max_evals_since_overwrite,
                                               cls_epochs_per_eval,
                                               DEBUG_MODE,
                                               sess,
                                               resources,
                                               self.class_labels,
                                               in_cls=self.in_cls)

                resources['train_summary_file'].close()
                resources['val_summary_file'].close()
                print('\n---- Finished training ----\n')
        return {
            'name': self.name,
            'train_summary_path': resources['train_summary_path'],
            'val_summary_path': resources['val_summary_path'],
            'loss': result['min_val_loss'],
            'acc': result['best_val_acc'],
            'cm': result['best_val_cm'],
            'ema_loss': result['ema_min_val_loss'],
            'ema_acc': result['ema_best_val_acc'],
            'ema_cm': result['ema_best_val_cm']
        }
