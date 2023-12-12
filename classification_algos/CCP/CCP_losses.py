"""
CCP_losses.py

Implements the loss functions of CCP as well as credibility propagation.

@author: Brody Kutt (bkutt@paloaltonetworks.com)
"""

import tensorflow as tf


def cred_adjust(v):
    """
    Implements the credibility adjustment in TF.
    """
    n = tf.shape(v)[0]
    n_classes = tf.shape(v)[1]
    v_reshape = tf.repeat(v[:, tf.newaxis, :], n_classes,
                          axis=1)  # <n, n_c, n_c>
    idxs = tf.math.logical_not(tf.eye(n_classes, dtype=tf.bool))  # <n_c, n_c>
    idxs_reshape = tf.repeat(idxs[tf.newaxis, :, :], n, axis=0)
    v_reshape = tf.reshape(v_reshape[idxs_reshape],
                           (n, n_classes, n_classes - 1))
    next_v = tf.reduce_max(v_reshape, axis=2)  # <n, n_c>
    return v - next_v


def compute_q_u(A, q_ij_lu, n_classes, use_cred, epsilon=1e-6):
    """
    Compute propagated credibility vectors and do credibility adjustment for
    unlabeled data only for the sake of efficiency.
    """
    # Set up
    n_u = tf.shape(A)[0]
    n = tf.shape(A)[1]
    n_l = n - n_u
    A_reshape = tf.repeat(A[:, tf.newaxis, :], n_classes,
                          axis=1)  # <n_u, n_c, n>
    Q_reshape = tf.repeat(tf.transpose(q_ij_lu)[tf.newaxis, :, :], n_u,
                          axis=0)  # <n_u, n_c, n>
    # Remove self-pairs among unlabeled samples
    idxs = tf.math.logical_not(tf.eye(n, dtype=tf.bool))[n_l:, :]  # <n_u, n>
    idxs_reshape = tf.repeat(idxs[:, tf.newaxis, :], n_classes,
                             axis=1)  # <n_u, n_c, n>
    A_reshape = tf.reshape(A_reshape[idxs_reshape],
                           (n_u, n_classes, n - 1))  # <n_u, n_c, n-1>
    Q_reshape = tf.reshape(Q_reshape[idxs_reshape],
                           (n_u, n_classes, n - 1))  # <n_u, n_c, n-1>
    # Construct phi
    phi = A_reshape * Q_reshape  # <n_u, n_c, n-1>
    # Clip negative values
    phi = tf.clip_by_value(phi, 0.0, 1.0)
    Q_reshape = tf.clip_by_value(Q_reshape, 0.0, 1.0)
    # Reduce by weighted mean, <n_u, n_c>
    cls_evi = tf.reduce_sum(phi, axis=2) / tf.maximum(
        tf.reduce_sum(Q_reshape, axis=2), epsilon)
    if (use_cred):
        # Adjust for credibility
        q_ij_u = cred_adjust(cls_evi)
    else:
        # Use softmax (for ablation analysis)
        q_ij_u = tf.nn.softmax(cls_evi, axis=1)
    return q_ij_u


def split_and_avg(q_ij):
    """
    """
    # Split transformed pairs
    q_i, q_j = tf.split(q_ij, num_or_size_splits=2, axis=0)
    # Average across transformed pairs
    return (q_i + q_j) / 2.0


def compute_cp(z_ij, q_ij, selector_ij, sim_func, n_classes, tol, use_trans,
               use_cred):
    """
    Compute credibility propagation. Implements Algorithm 2 in the paper.
    """
    # Isolated labeled and unlabeled data
    z_ij_l = z_ij[selector_ij]
    z_ij_u = z_ij[tf.math.logical_not(selector_ij)]
    z_ij_lu = tf.concat([z_ij_l, z_ij_u], axis=0)
    q_ij_l = q_ij[selector_ij]
    q_ij_u = q_ij[tf.math.logical_not(selector_ij)]
    q_ij_lu = tf.concat([q_ij_l, q_ij_u], axis=0)  # <n, n_c>
    # Compute partial adjacency matrix
    A = sim_func(z_ij_u, z_ij_lu, pre_normalized=True)  # <n_u, n>
    # Do credibility propagation
    new_q_ij_u = compute_q_u(A, q_ij_lu, n_classes, use_cred)
    # Get the credibility vectors to return
    new_q_u = tf.cond(use_trans, lambda: split_and_avg(new_q_ij_u),
                      lambda: new_q_ij_u)
    return new_q_u


def get_sc_losses(zx_ij, q_ij, sim_func, tau, n_classes, epsilon,
                  unsupervised):
    """
    Soft contrastive (SC) loss. Implements Eq (1) in the paper. Also implements
    SimCLR loss if unsupervised=True. If all data is labeled and full strength,
    this implements SupCon loss.
    """
    n = tf.shape(zx_ij)[0]
    half_n = n // 2
    # Ignore negative values in Q
    pos_q_ij = tf.clip_by_value(q_ij, 0.0, 1.0)
    # Compute pairwise matching matrix, <n x n>
    M = tf.cond(unsupervised, lambda: tf.tile(tf.eye(half_n), [2, 2]),
                lambda: tf.linalg.matmul(pos_q_ij, tf.transpose(pos_q_ij)))
    # Self pairs should be excluded
    M = M * (1.0 - tf.eye(n))
    # Compute the strength matrix
    S = tf.cond(
        unsupervised, lambda: tf.ones([n, n], dtype=tf.float32),
        lambda: tf.tile(tf.reshape(tf.reduce_max(pos_q_ij, axis=1), [1, n]),
                        [n, 1]))  # <n, n>
    # Compute adjacency matrix, <n x n>
    A = sim_func(zx_ij, zx_ij, pre_normalized=True)
    # Use exponential and scale by temperature
    A = tf.math.exp(A / tau)
    # Self pairs will be excluded
    A = A * (1.0 - tf.eye(n))
    # Divide pairs by denominators and take the log
    A = tf.math.log((A /
                     (tf.reduce_sum(A * S, axis=1, keepdims=True) + epsilon)) +
                    epsilon)
    # Compute the weighted average, <n>
    weighted_avgs = -tf.reduce_sum(A * M, axis=1) / (tf.reduce_sum(M, axis=1) +
                                                     epsilon)
    # Scale by confidence & return
    conf_vec = tf.cond(unsupervised,
                       lambda: tf.ones_like(tf.reduce_max(pos_q_ij, axis=1)),
                       lambda: tf.reduce_max(pos_q_ij, axis=1))
    return conf_vec * weighted_avgs


def get_ce_losses(scores, q, epsilon=1e-6):
    """
    Soft cross-entropy loss used during classifier learning. Implements Eq (2)
    in the paper.
    """
    # Ignore negative values in Q
    pos_q = tf.clip_by_value(q, 0.0, 1.0)
    pos_scores = tf.clip_by_value(scores, 0.0, 1.0)
    return -tf.reduce_sum(pos_q * tf.math.log(pos_scores + epsilon), axis=-1)
