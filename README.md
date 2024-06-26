An implementation of the Contrastive Credibility Propagation (CCP) algorithm in Tensorflow. This implementation will allow one to recreate all CIFAR-10/CIFAR-100 experiments documented inside [our paper](https://arxiv.org/abs/2211.09929) as well as test new data scenarios. Data scenarios can be created with this repo and exported for use with other algorithms. This project has been tested with Python 3 and the package versions listed in `requirements_p3.py`.

## Organizing raw data on disk

Download the Python version of CIFAR-10 and CIFAR-100 pickled files containing the training and test data from https://www.cs.toronto.edu/~kriz/cifar.html

Place the training batches in one folder and the test batch in another. E.g., organize the pickled files as follows:

For CIFAR-10:

    raw_data/cifar_10/train/data_batch_*
    raw_data/cifar_10/test/test_batch

For CIFAR-100:

    raw_data/cifar_100/train/train
    raw_data/cifar_100/test/test

## Encode the data splits

`dataset_construct.py` will load each pickled file and save it in a custom NPZ archive. E.g.,

    python3 dataset_construct.py --raw_data_dir raw_data/cifar_10/train --cifar_version 10 --save_fp vector_data/cifar_10/imgs/train/data.npz
    python3 dataset_construct.py --raw_data_dir raw_data/cifar_10/test --cifar_version 10 --save_fp vector_data/cifar_10/imgs/test/data.npz

    python3 dataset_construct.py --raw_data_dir raw_data/cifar_100/train --cifar_version 100 --save_fp vector_data/cifar_100/imgs/train/data.npz
    python3 dataset_construct.py --raw_data_dir raw_data/cifar_100/test --cifar_version 100 --save_fp vector_data/cifar_100/imgs/test/data.npz

## Create a label mask

`create_y_mask.py` will create a mask over training data labels. A unique Y mask is responsible for implementing each desired data scenario. The mask will decide which samples to treat as unlabeled (even if the true label is known) and which samples to keep or remove. Samples with kept labels will have a value of 1. Samples with hidden labels will have a mask value of 0. Samples marked for removal will have a mask value of -1. An example call to create a mask with 30% of labels from each class hidden (stratified by class) is as follows:

    python3 create_y_mask.py --npz_fp vector_data/cifar_10/imgs/train/data.npz --save_fp vector_data/cifar_10/y_masks/my_mask.npz --perc_on 0.7

You can also specify which classes to keep inside the labeled data and unlabeled data separately and which to remove via the `--labeled_classes` and `--unlabeled_classes` flags. This allows for open-set experiments. Also, specify the `--labeled_imbalance_factor` and `--unlabeled_imbalance_factor` flags to create imbalanced distributions.

The following commands will create Y masks used to recreate the CIFAR-10 data scenario experiments in [our paper](https://arxiv.org/abs/2211.09929):

    python3 create_y_mask.py --npz_fp vector_data/cifar_10/imgs/train/data.npz --perc_on 1.0 --labeled_classes 0,1,2,3 --unlabeled_classes 0,1,2,3 --save_fp vector_data/cifar_10/y_masks/control.npz
    python3 create_y_mask.py --npz_fp vector_data/cifar_10/imgs/train/data.npz --perc_on 0.08 --labeled_classes 0,1,2,3 --unlabeled_classes 0,1,2,3 --save_fp vector_data/cifar_10/y_masks/1600_labels.npz

    python3 create_y_mask.py --npz_fp vector_data/cifar_10/imgs/train/data.npz --perc_on 0.005 --labeled_classes 0,1,2,3 --unlabeled_classes 0,1,2,3 --save_fp vector_data/cifar_10/y_masks/1_few.npz
    python3 create_y_mask.py --npz_fp vector_data/cifar_10/imgs/train/data.npz --perc_on 0.0008 --labeled_classes 0,1,2,3 --unlabeled_classes 0,1,2,3 --save_fp vector_data/cifar_10/y_masks/2_few.npz
    python3 create_y_mask.py --npz_fp vector_data/cifar_10/imgs/train/data.npz --perc_on 0.0004 --labeled_classes 0,1,2,3 --unlabeled_classes 0,1,2,3 --save_fp vector_data/cifar_10/y_masks/3_few.npz

    python3 create_y_mask.py --npz_fp vector_data/cifar_10/imgs/train/data.npz --perc_on 0.08 --labeled_classes 0,1,2,3 --unlabeled_classes 0,1,2,3,4,5 --save_fp vector_data/cifar_10/y_masks/1_ood.npz
    python3 create_y_mask.py --npz_fp vector_data/cifar_10/imgs/train/data.npz --perc_on 0.08 --labeled_classes 0,1,2,3 --unlabeled_classes 0,1,2,3,4,5,6,7 --save_fp vector_data/cifar_10/y_masks/2_ood.npz
    python3 create_y_mask.py --npz_fp vector_data/cifar_10/imgs/train/data.npz --perc_on 0.08 --labeled_classes 0,1,2,3 --unlabeled_classes 0,1,2,3,4,5,6,7,8,9 --save_fp vector_data/cifar_10/y_masks/3_ood.npz

    python3 create_y_mask.py --npz_fp vector_data/cifar_10/imgs/train/data.npz --perc_on 0.08 --labeled_classes 0,1,2,3 --unlabeled_classes 0,1,2,3 --unlabeled_imbalance_factor 0.2 --save_fp vector_data/cifar_10/y_masks/1_misalign.npz
    python3 create_y_mask.py --npz_fp vector_data/cifar_10/imgs/train/data.npz --perc_on 0.08 --labeled_classes 0,1,2,3 --unlabeled_classes 0,1,2,3 --unlabeled_imbalance_factor 0.1 --save_fp vector_data/cifar_10/y_masks/2_misalign.npz
    python3 create_y_mask.py --npz_fp vector_data/cifar_10/imgs/train/data.npz --perc_on 0.08 --labeled_classes 0,1,2,3 --unlabeled_classes 0,1,2,3 --unlabeled_imbalance_factor 0.0 --save_fp vector_data/cifar_10/y_masks/3_misalign.npz

    python3 create_y_mask.py --npz_fp vector_data/cifar_10/imgs/train/data.npz --perc_on 0.08 --labeled_classes 0,1,2,3 --unlabeled_classes 0,1,2,3 --labeled_imbalance_factor 0.0625 --save_fp vector_data/cifar_10/y_masks/4_misalign.npz
    python3 create_y_mask.py --npz_fp vector_data/cifar_10/imgs/train/data.npz --perc_on 0.08 --labeled_classes 0,1,2,3 --unlabeled_classes 0,1,2,3 --labeled_imbalance_factor 0.01 --save_fp vector_data/cifar_10/y_masks/5_misalign.npz
    python3 create_y_mask.py --npz_fp vector_data/cifar_10/imgs/train/data.npz --perc_on 0.08 --labeled_classes 0,1,2,3 --unlabeled_classes 0,1,2,3 --labeled_imbalance_factor 0.005 --save_fp vector_data/cifar_10/y_masks/6_misalign.npz

(Noise experiments are handled in the training config with the noise_ratio parameter; use 1600_labels.npz for the y_mask in these experiments)

The following commands will create Y masks used to recreate the CIFAR-100 data scenario experiments in [our paper](https://arxiv.org/abs/2211.09929):

    python3 create_y_mask.py --npz_fp vector_data/cifar_100/imgs/train/data.npz --perc_on 1.0 --labeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --unlabeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --save_fp vector_data/cifar_100/y_masks/control.npz
    python3 create_y_mask.py --npz_fp vector_data/cifar_100/imgs/train/data.npz --perc_on 0.2 --labeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --unlabeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --save_fp vector_data/cifar_100/y_masks/4000_labels.npz

    python3 create_y_mask.py --npz_fp vector_data/cifar_100/imgs/train/data.npz --perc_on 0.05 --labeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --unlabeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --save_fp vector_data/cifar_100/y_masks/1_few.npz
    python3 create_y_mask.py --npz_fp vector_data/cifar_100/imgs/train/data.npz --perc_on 0.008 --labeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --unlabeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --save_fp vector_data/cifar_100/y_masks/2_few.npz
    python3 create_y_mask.py --npz_fp vector_data/cifar_100/imgs/train/data.npz --perc_on 0.004 --labeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --unlabeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --save_fp vector_data/cifar_100/y_masks/3_few.npz

    python3 create_y_mask.py --npz_fp vector_data/cifar_100/imgs/train/data.npz --perc_on 0.2 --labeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --unlabeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59 --save_fp vector_data/cifar_100/y_masks/1_ood.npz
    python3 create_y_mask.py --npz_fp vector_data/cifar_100/imgs/train/data.npz --perc_on 0.2 --labeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --unlabeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79 --save_fp vector_data/cifar_100/y_masks/2_ood.npz
    python3 create_y_mask.py --npz_fp vector_data/cifar_100/imgs/train/data.npz --perc_on 0.2 --labeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --unlabeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99 --save_fp vector_data/cifar_100/y_masks/3_ood.npz

    python3 create_y_mask.py --npz_fp vector_data/cifar_100/imgs/train/data.npz --perc_on 0.2 --labeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --unlabeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --unlabeled_imbalance_factor 0.2 --save_fp vector_data/cifar_100/y_masks/1_misalign.npz
    python3 create_y_mask.py --npz_fp vector_data/cifar_100/imgs/train/data.npz --perc_on 0.2 --labeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --unlabeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --unlabeled_imbalance_factor 0.1 --save_fp vector_data/cifar_100/y_masks/2_misalign.npz
    python3 create_y_mask.py --npz_fp vector_data/cifar_100/imgs/train/data.npz --perc_on 0.2 --labeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --unlabeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --unlabeled_imbalance_factor 0.0 --save_fp vector_data/cifar_100/y_masks/3_misalign.npz

    python3 create_y_mask.py --npz_fp vector_data/cifar_100/imgs/train/data.npz --perc_on 0.2 --labeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --unlabeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --labeled_imbalance_factor 0.25 --save_fp vector_data/cifar_100/y_masks/4_misalign.npz
    python3 create_y_mask.py --npz_fp vector_data/cifar_100/imgs/train/data.npz --perc_on 0.2 --labeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --unlabeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --labeled_imbalance_factor 0.04 --save_fp vector_data/cifar_100/y_masks/5_misalign.npz
    python3 create_y_mask.py --npz_fp vector_data/cifar_100/imgs/train/data.npz --perc_on 0.2 --labeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --unlabeled_classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 --labeled_imbalance_factor 0.02 --save_fp vector_data/cifar_100/y_masks/6_misalign.npz

(Noise experiments are handled in the training config with the noise_ratio parameter; use 4000_labels.npz for the y_mask in these experiments)

## Train and evaluate models

CCP's training script is `classification_algos/CCP/train_CCP.py`. Config files specify all hyperparameter settings. See the config file template for an explanation of the hyperparameter choices in each entry in a config file. E.g., to run a training job with my_config on a GPU with ID 0:

    python3 train_CCP.py --config_files my_config --gpu_id 0

It is recommended to capture the output of `train_CCP.py` as lots of useful information will be printed to the terminal. This can be accomplished with:

    python3 -u train_CCP.py --config_files my_config --gpu_id 0 > logs/my_config.log 2>&1

The result of calling this script will create a directory (specified in the config file) which stores some of the results. An example:

    my_model/
        models/
            warmup/                             (A copy of the full model when warmup was finished, if it was used)
                my_model.data-00000-of-00001
                my_model.index
                my_model.meta
                checkpoint
        q_vecs/                                 (Copies of the state of Q vectors after each CCP iteration)
            iter_1_q.npz
            iter_2_q.npz
            ...
        my_model_batch_summary.csv              (Records the results of each batch over time)
        my_model_train_summary.csv              (Records the results of each evaluation of the train set over time)
        my_model_val_summary.csv                (Records the results of each evaluation of the eval set over time)

Sample config files are provided to recreate the experiments in [our paper](https://arxiv.org/abs/2211.09929) inside the repo. For CIFAR-10: `classification_algos/CCP/configs/cifar_10/*`, for CIFAR-100: `classification_algos/CCP/configs/cifar_100/*`.

The three steps of building of classifier are split into three separate config files (although they don't have to be). `warmup_model.ini` defines the procedure to build a warmed up network state using self-supervised loss only. Training configs that define CCP iterations will use the warmup model as a starting point. Config files that define CCP iterations are named with a prefix `{1_, 2_, 3_}` that indicates the data scenario severity level followed by the experimental data variable. E.g., for the few-label experiment at the second most severe level, the config file is `2_few.ini`. The config file which defines building the classifier has a suffix of `-cls.ini` e.g. `2_few-cls.ini`. This config will borrow the desired state of Q vectors that were computed in `2_few.ini`. Supervised baseline experiments have a suffix of `-baseline` e.g. `2_few-baseline.ini`. The base case of CIFAR-10 experiments is called `1600_labels` while the base case of CIFAR-100 is called `4000_labels` which indicates the total amount of labeled data in that experiment.

## Create and save an altered copy of CIFAR for use with other algorithms

Use `get_processed_dataset.py` to create an altered version of CIFAR reflecting a data scenario and then save it to disk. You can specify the Y mask and noise ratio that implements a desired data scenario. The saved NPZ archive can then be imported for use with other algorithms. E.g., to create the dataset that implements the noisy-label scenario of CIFAR-100 at maximum severity, you'd use:

    python3 get_processed_dataset.py --train_fp vector_data/cifar_100/imgs/train/data.npz --test_fp vector_data/cifar_100/imgs/test/data.npz --y_mask_fp vector_data/cifar_100/y_masks/4000_labels.npz --noise_ratio 0.6 --save_fp vector_data/cifar_100/preprocessed_dataset/3_noise.npz

## Cite

```
@article{
    Kutt_Ramteke_Mignot_Toman_Ramanan_Rokka Chhetri_Huang_Du_Hewlett_2024,
    title={Contrastive Credibility Propagation for Reliable Semi-supervised Learning},
    volume={38},
    url={https://ojs.aaai.org/index.php/AAAI/article/view/30124},
    DOI={10.1609/aaai.v38i19.30124},
    abstractNote={Producing labels for unlabeled data is error-prone, making semi-supervised learning (SSL) troublesome. Often, little is known about when and why an algorithm fails to outperform a supervised baseline. Using benchmark datasets, we craft five common real-world SSL data scenarios: few-label, open-set, noisy-label, and class distribution imbalance/misalignment in the labeled and unlabeled sets. We propose a novel algorithm called Contrastive Credibility Propagation (CCP) for deep SSL via iterative transductive pseudo-label refinement. CCP unifies semi-supervised learning and noisy label learning for the goal of reliably outperforming a supervised baseline in any data scenario. Compared to prior methods which focus on a subset of scenarios, CCP uniquely outperforms the supervised baseline in all scenarios, supporting practitioners when the qualities of labeled or unlabeled data are unknown.},
    number={19},
    journal={Proceedings of the AAAI Conference on Artificial Intelligence},
    author={Kutt, Brody and Ramteke, Pralay and Mignot, Xavier and Toman, Pamela and Ramanan, Nandini and Rokka Chhetri, Sujit and Huang, Shan and Du, Min and Hewlett, William},
    year={2024},
    month={Mar.},
    pages={21294-21303}
}
```
