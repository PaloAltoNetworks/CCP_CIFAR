"""
train_CCP.py

Initiates training with CCP. Reads a config file and calls the necessary
processes.

@author: Brody Kutt (bkutt@paloaltonetworks.com)
"""

import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
import utils
import random
import argparse
import numpy as np
import configparser
from CCP_Builder import CCP_Builder


def process_config(config):
    """
    Take a loaded config instance and transfer it into a dictionary with the
    correct datatypes.
    """
    args = {}
    # Grab critical params
    args['train_fp'] = config.get('Critical', 'train_fp')
    if (not os.path.isfile(args['train_fp'])):
        print('ERR: train_CCP.py: Failed to load train data.')
        assert (False)
    args['val_fp'] = config.get('Critical', 'val_fp')
    if (not os.path.isfile(args['val_fp'])):
        print('ERR: train_CCP.py Failed to load val data.')
        assert (False)
    args['y_mask_fp'] = config.get('Critical', 'y_mask_fp')
    if (not os.path.isfile(args['y_mask_fp'])):
        print('ERR: train_CCP.py Failed to load the y mask.')
        assert (False)
    args['save_dir'] = config.get('Critical', 'save_dir')
    args['z_dims'] = config.get('Critical', 'z_dims')
    if (args['z_dims'] == ''):
        args['z_dims'] = []
    else:
        args['z_dims'] = [int(i.strip()) for i in args['z_dims'].split(',')]
    args['g_dims'] = config.get('Critical', 'g_dims')
    if (args['g_dims'] == ''):
        args['g_dims'] = []
    else:
        args['g_dims'] = [int(i.strip()) for i in args['g_dims'].split(',')]

    args['sim_metric'] = config.get('Critical', 'sim_metric')
    args['drop_keep_prob'] = config.getfloat('Critical', 'drop_keep_prob')
    args['contrastive_batch_size'] = config.getint('Critical',
                                                   'contrastive_batch_size')
    args['cls_batch_size'] = config.getint('Critical', 'cls_batch_size')
    args['cls_min_epochs'] = config.getint('Critical', 'cls_min_epochs')
    args['cls_epochs_per_eval'] = config.getint('Critical',
                                                'cls_epochs_per_eval')
    args['cls_max_evals_since_overwrite'] = config.getint(
        'Critical', 'cls_max_evals_since_overwrite')
    args['do_warmup'] = config.get('Critical',
                                   'do_warmup').strip().startswith('T')
    args['activation'] = config.get('Critical', 'activation')
    args['tau'] = config.getfloat('Critical', 'tau')
    args['sc_ema_weight'] = config.getfloat('Critical', 'sc_ema_weight')
    args['sc_max_epochs_since_overwrite'] = config.getint(
        'Critical', 'sc_max_epochs_since_overwrite')
    args['wu_max_epochs_since_overwrite'] = config.getint(
        'Critical', 'wu_max_epochs_since_overwrite')
    args['ccp_iter_schedule'] = config.get('Critical', 'ccp_iter_schedule')
    if (args['ccp_iter_schedule'] == ''):
        args['ccp_iter_schedule'] = []
    else:
        args['ccp_iter_schedule'] = [
            int(i.strip()) for i in args['ccp_iter_schedule'].split(',')
        ]
    args['alter_first_iter'] = config.get(
        'Critical', 'alter_first_iter').strip().startswith('T')

    # Grab optional params
    try:
        args['l2_lambda'] = config.getfloat('Optional', 'l2_lambda')
    except configparser.NoOptionError:
        args['l2_lambda'] = 0.0
    try:
        args['learning_rate'] = config.getfloat('Optional', 'learning_rate')
    except configparser.NoOptionError:
        args['learning_rate'] = 0.06
    try:
        args['random_seed'] = config.getint('Optional', 'random_seed')
    except configparser.NoOptionError:
        args['random_seed'] = utils.random_seed()
    random.seed(args["random_seed"])
    np.random.seed(args["random_seed"])
    try:
        args['eval_on_train_data'] = config.get(
            'Optional', 'eval_on_train_data').strip().startswith('T')
    except configparser.NoOptionError:
        args['eval_on_train_data'] = True
    try:
        args['record_batch_evals'] = config.get(
            'Optional', 'record_batch_evals').strip().startswith('T')
    except configparser.NoOptionError:
        args['record_batch_evals'] = True
    try:
        args['trans_in_ccp'] = config.get(
            'Optional', 'trans_in_ccp').strip().startswith('T')
    except configparser.NoOptionError:
        args['trans_in_ccp'] = True
    try:
        args['trans_in_cls'] = config.get(
            'Optional', 'trans_in_cls').strip().startswith('T')
    except configparser.NoOptionError:
        args['trans_in_cls'] = True
    try:
        args['scale_q'] = config.get('Optional',
                                     'scale_q').strip().startswith('T')
    except configparser.NoOptionError:
        args['scale_q'] = True
    try:
        args['init_q_fp'] = config.get('Optional', 'init_q_fp')
    except configparser.NoOptionError:
        args['init_q_fp'] = None
    try:
        args['warmup_chk_path'] = config.get('Optional', 'warmup_chk_path')
    except configparser.NoOptionError:
        args['warmup_chk_path'] = ''
    try:
        args['max_kl_div'] = config.getfloat('Optional', 'max_kl_div')
    except configparser.NoOptionError:
        args['max_kl_div'] = 0.0
    try:
        args['noise_ratio'] = config.getfloat('Optional', 'noise_ratio')
    except configparser.NoOptionError:
        args['noise_ratio'] = 0.0
    try:
        args['cifar_version'] = config.getint('Optional', 'cifar_version')
    except configparser.NoOptionError:
        args['cifar_version'] = 10
    assert (args['cifar_version'] == 10 or args['cifar_version'] == 100)
    try:
        args['use_cred'] = config.get('Optional',
                                      'use_cred').strip().startswith('T')
    except configparser.NoOptionError:
        args['use_cred'] = True
    try:
        args['use_comb_loss'] = config.get(
            'Optional', 'use_comb_loss').strip().startswith('T')
    except configparser.NoOptionError:
        args['use_comb_loss'] = True
    return args


def print_ccp_params(params):
    """
    Print the hyperparameters of CCP
    """
    print(('CIFAR version: %d' % params['cifar_version']))
    print(('Contrastive batch size: %d' % params['contrastive_batch_size']))
    print(('Classifier batch size: %d' % params['cls_batch_size']))
    print(('Minimum number of epochs during classifier training: %d' %
           params['cls_min_epochs']))
    print((
        'Maximum evaluations since an overwrite during classifier training: %d'
        % params['cls_max_evals_since_overwrite']))
    print(('Epochs until an evaluation during classifier training: %d' %
           params['cls_epochs_per_eval']))
    print(('Doing warmup: %s' % str(params['do_warmup'])))
    if (params['warmup_chk_path'] != ''):
        print('Using path to warmed up network: %s' %
              params['warmup_chk_path'])
    print(('Activation function: %s' % params['activation']))
    print(('Z projection head layers: %s' % str(params['z_dims'])))
    print(('G projection head layers: %s' % str(params['g_dims'])))
    print(('Similarity metric: %s' % params['sim_metric']))
    print(('Tau: %f' % params['tau']))
    print(('Dropout keep probability: %f' % params['drop_keep_prob']))
    print(('Learning rate: %f' % params['learning_rate']))
    print(('L2 lambda: %f' % params['l2_lambda']))
    print(('Random seed: %d' % params['random_seed']))
    print(('Using transformations in CCP iterations: %s' %
           str(params['trans_in_ccp'])))
    print(('Using transformations in classifier training: %s' %
           str(params['trans_in_cls'])))
    print(('Soft Contrastive EMA weight: %f' % params['sc_ema_weight']))
    print(('Max epochs since an overwrite during CCP iterations: %d' %
           params['sc_max_epochs_since_overwrite']))
    print(('Max epochs since an overwrite during warmup: %d' %
           params['wu_max_epochs_since_overwrite']))
    print(('CCP iteration schedule: %s' % str(params['ccp_iter_schedule'])))
    print(('Alter the first CCP iteration: %s' %
           str(params['alter_first_iter'])))
    if (params['init_q_fp'] is not None):
        print(('Loading Q vectors from: %s' % str(params['init_q_fp'])))
    if (params['max_kl_div'] >= 0.0):
        print(('Max KL_Divergence: %f' % params['max_kl_div']))
    if (params['noise_ratio'] >= 0.0):
        print(('Noise ratio: %f' % params['noise_ratio']))
    print(('Scaling the Q vectors: %s' % str(params['scale_q'])))
    print(('Using credibility: %s' % str(params['use_cred'])))
    print(('Using combo loss: %s\n' % str(params['use_comb_loss'])))


def train_ccp(args, data):
    """
    """
    print('\nCreating the CCP builder instance...')
    ccp = CCP_Builder(os.path.basename(args['save_dir']),
                      os.path.dirname(args['save_dir']), args['z_dims'],
                      args['g_dims'], args['sim_metric'], args['activation'],
                      args['l2_lambda'], args['random_seed'],
                      data['int2label'], args['tau'], args['trans_in_ccp'],
                      args['trans_in_cls'], args['ccp_iter_schedule'],
                      args['scale_q'], args['max_kl_div'],
                      args['noise_ratio'] == 0.0, args['cifar_version'],
                      args['use_cred'], args['use_comb_loss'])
    print('Training CCP with the following hyperparameters:\n')
    print_ccp_params(utils.merge_dicts([args, data]))
    ccp.train(data['X'], data['y'], data['y_clean'], data['y_mask'],
              args['contrastive_batch_size'], args['cls_batch_size'],
              args['cls_min_epochs'], args['cls_epochs_per_eval'],
              args['cls_max_evals_since_overwrite'], data['X_val'],
              data['y_val'], args['drop_keep_prob'], args['learning_rate'],
              args['eval_on_train_data'], args['record_batch_evals'],
              args['init_q_fp'], args['do_warmup'], args['warmup_chk_path'],
              args['alter_first_iter'], args['sc_ema_weight'],
              args['sc_max_epochs_since_overwrite'],
              args['wu_max_epochs_since_overwrite'])


if __name__ == '__main__':
    # Parse the arguments and options
    parser = argparse.ArgumentParser(description=('Train a CCP model.'))
    parser.add_argument(
        "--config_files",
        nargs="+",
        required=True,
        help=("The list of configuration files. Each configuration will be"
              "executed in sequence in the order that they were passed in."))
    parser.add_argument(
        "--gpu_id",
        required=False,
        default=None,
        help=("Specificy a GPU to use by its ID (default: None)."))
    args = parser.parse_args()

    if args.gpu_id is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Search for the config files
    configs = [configparser.ConfigParser() for c in args.config_files]
    for (i, config) in enumerate(configs):
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'configs', args.config_files[i])
        if (config.read(full_path) == []):
            full_path += '.ini'  # Try adding the file extension
            if (config.read(full_path) == []):
                print(('ERR: train_CCP.py: Config file "%s" is missing '
                       'or empty.') % full_path)
                sys.exit(-1)  # Exit failure
    # Process the config files
    arg_sets = [process_config(config) for config in configs]
    for (i, args) in enumerate(arg_sets):
        print('-' * 80)
        print('Processing config #%d of %d...\n' % (i + 1, len(arg_sets)))
        # Load full dataset
        data = utils.load_dataset(args['train_fp'],
                                  args['val_fp'],
                                  args['y_mask_fp'],
                                  noise_ratio=args['noise_ratio'])
        # If data is loaded, start the training
        train_ccp(args, data)
    print('\nExiting...')
    print('-' * 80)
