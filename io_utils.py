import numpy as np
import os
import glob
import argparse
import backbone

def parse_args(mode):
    parser = argparse.ArgumentParser(description='CD-FSL ({} mode)'.format(mode))
    parser.add_argument('--dataset'     , default='miniImageNet',        help='training base model')
    parser.add_argument('--model'       , default='ResNet10',      help='backbone architecture')  # refers to (ssl) method in new modules
    parser.add_argument('--method'      , default='baseline',   help='baseline/protonet/maml')
    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training')
    parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ')
    parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support')
    parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ')
    parser.add_argument('--freeze_backbone'   , action='store_true', help='Freeze the backbone network for finetuning')
    parser.add_argument('--models_to_use', '--names-list', nargs='+', default=['miniImageNet', 'caltech256', 'DTD', 'cifar100', 'CUB'], help='pretained model to use')
    parser.add_argument('--fine_tune_all_models'   , action='store_true',  help='fine-tune each model before selection') #still required for save_features.py and test.py to find the model path correctly

    # New parameters
    parser.add_argument('--source_dataset', default='miniImageNet')  # replaces dataset
    parser.add_argument('--target_dataset', type=str, nargs='+')  # replaces dataset_names / HOTFIX: changed to list to allow for multiple targets with one CLI command
    parser.add_argument('--backbone', default='resnet10', help='Refer to backbone._backbone_class_map')  # replaces model
    # parser.add_argument('--model', default='base', help='Refer to model.model_class_map')  # similar to method

    # Model parameters (make sure to prepend with `model_`)
    parser.add_argument('--model_simclr_projection_dim', default=128, type=int)
    parser.add_argument('--model_simclr_temperature', default=1.0, type=float)

    # Pre-train params (determines pre-trained model output directory)
    # These must be specified during evaluation and fine-tuning to select pre-trained model
    parser.add_argument('--pls', action='store_true', help='Second-step pre-training on top of model trained with source labeled data')
    parser.add_argument('--ls', action='store_true', help='Use labeled source data for pre-training')
    parser.add_argument('--us', action='store_true', help='Use unlabeled source data for pre-training')
    parser.add_argument('--ut', action='store_true', help='Use unlabeled target data for pre-training')
    parser.add_argument('--tag', default='default', type=str, help='Tag used to differentiate output directories for pre-trained models')  # similar to aug_mode
    parser.add_argument('--pls_tag', default=None, type=str, help='Tag of pre-trained previous model (LS type) used for pls. Uses --tag by default.')

    """
    Type 1: --ls
    Type 2: --us
    Type 3: --ut
    Type 4: --ls --ut
    Type 5: --us --ut
    Type 6: --pls --ut
    Type 7: --pls --ls --ut
    Type 8: --pls --us --ut
    """

    # Pre-train params (non-identifying, i.e., does not affect output directory)
    # You must specify --tag to differentiate models with different non-identifying parameters)
    parser.add_argument('--augmentation', default='strong', type=str, help="Augmentation used for pre-training {'base', 'strong'}")  # similar to aug_mode
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for pre-training.')  # similar to aug_mode
    parser.add_argument('--lr', default=None, type=float, help='LR for pre-training.')
    parser.add_argument('--gamma', default=0.5, type=float, help='Gamma value for {LS,US} + UT.')  # similar to aug_mode
    parser.add_argument('--epochs', default=1000, type=int, help='Pre-training epochs.')  # similar to aug_mode
    parser.add_argument('--model_save_interval', default=50, type=int, help='Save model state every N epochs during pre-training.')  # similar to aug_mode
    parser.add_argument('--optimizer', default=None, type=str, help="Optimizer used during pre-training {'sgd', 'adam'}. Default if None")  # similar to aug_mode
    parser.add_argument('--num_workers', default=2, type=int)

    # New ft params
    parser.add_argument('--n_way', default=5, type=int)
    parser.add_argument('--n_query_shot', default=15, type=int)

    parser.add_argument('--ft_head', default='linear', help='See `model.classifier_head.CLASSIFIER_HEAD_CLASS_MAP`')
    parser.add_argument('--ft_tag', default='default', type=str, help='Tag used to differentiate output directories for fine-tuned models')
    parser.add_argument('--ft_epochs', default=100, type=int)
    parser.add_argument('--ft_pretrain_epoch', default=None, type=int)
    parser.add_argument('--ft_batch_size', default=4, type=int)
    parser.add_argument('--ft_augmentation', default=None, type=str, help="Augmentation used for fine-tuning {None, 'base', 'strong'}")
    parser.add_argument('--ft_parts', default='head', type=str, help="Where to fine-tune: {'full', 'body', 'head'}")
    parser.add_argument('--ft_features', default=None, type=str, help='Specify which features to use from the base model (see model/base.py)')
    parser.add_argument('--ft_intermediate_test', action='store_true', help='Evaluate on query set during fine-tuning')
    parser.add_argument('--ft_episode_seed', default=0, type=int)

    if mode == 'train' or mode == 'pretrain':
        parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=50, type=int, help='Save frequency')
        parser.add_argument('--pretrain_type', default=None, type=int, help='How to pre-train')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=400, type=int, help ='Stopping epoch') # for meta-learning methods, each epoch contains 100 episodes

        # For pre-trained model (related to BN)
        parser.add_argument('--track_bn'   , action='store_true',  help='tracking BN stats')
        parser.add_argument('--freeze_bn', action='store_true',  help='freeze bn stats, i.e., use accumulated stats of pretrained model during inference. Note, track_bn must be on to do this.')
        parser.add_argument('--reinit_bn_stats'   , action='store_true',  help='Re-initialize BN running statistics every iteration')

        # For SimCLR
        parser.add_argument('--aug_mode', default=None, help='augmentation for pre-training [base, strong]')
        parser.add_argument('--use_base_classes'   , action='store_true',  help='supervised training using base classes with self-training')
        parser.add_argument('--use_base_classes_as_unlabeled'   , action='store_true',  help='unsupervised training using base classes with self-training')
        parser.add_argument('--no_rerand'   , action='store_true',  help='No re-randomization before SimCLR traininig')
        parser.add_argument('--no_base_pretraining'   , action='store_true',  help='No use pre-trained model based on base classes')

        # For fine-tuning
        parser.add_argument('--mv_init', action='store_true', help ='Re-initialize all weights with existing mean-var stats')
        parser.add_argument('--simclr_finetune', action='store_true', help ='Fine-tuning using the model trained by SimCLR')
        parser.add_argument('--simclr_finetune_source', action='store_true', help ='Fine-tuning using the model trained by source+SimCLR')
        parser.add_argument('--simclr_epochs', nargs='+', type=int, default=[1000, 800, 600, 400, 200, 0], help ='Which epochs to fine-tune for SimCLR (near finetune.py:486)')
        parser.add_argument('--reinit_stem', action='store_true', help ='Re-initialize Stem')
        parser.add_argument('--reinit_blocks', nargs='+', type=int, help ='Re-initialize ResNet blocks (select within range [1, 4])')

        parser.add_argument('--reset_layers', default=None, nargs='+', type=str, help='Re-randomize (or re-init) layers. Refer to `reset_layer.py` for layer names. E.g., 4.c2, 4.b2, 4.cs, 4.bs')
        parser.add_argument('--reset_layer_method', default='rerandomize', help='rerandomize, reinit')
        parser.add_argument('--unlabeled_stats', action='store_true', help ='Use statistics of unlabeled target dataset for BN running stats')

        parser.add_argument('--finetune_parts', default=None, type=str, help='head, body, full')
        parser.add_argument('--fusion_method', default=None, type=str, help='concat, etc, ...')

        parser.add_argument('--no_tracking', action='store_true', help='No tracking the test accuracy for every epoch')
        parser.add_argument('--dataset_names', nargs='+', type=str, default=["miniImageNet_test", "CropDisease", "EuroSAT", "ISIC", "ChestX"], help='CD-FSL datasets to fine-tune')
        parser.add_argument('--use_simclr_clf', action='store_true', help ='Use pre-trained SimCLR projection head')

        # For STARTUP-like split (deprecated. Update with finetune.py)
        parser.add_argument('--startup_split', action='store_true', help ='Use 80% of dataset, similar to STARTUP. Enabled automatically for simclr_finetune.')
        # For split (split may be used depending on pretrain_type
        parser.add_argument('--unlabeled_ratio', default=20, type=int, help ='Percentage of dataset used for unlabeled split')
        parser.add_argument('--split_seed', default=1, type=int, help ='Random seed used for split. If set to 1 and unlabeled_ratio==20, will use split defined by STARTUP')
    elif mode == 'save_features':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--save_iter', default=-1, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1')
    elif mode == 'test':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--adaptation'  , action='store_true', help='further adaptation in test time or not')
    else:
        raise ValueError('Unknown script')

    params = parser.parse_args()

    # Double-checking parameters
    if params.freeze_bn and not params.track_bn:
        raise AssertionError('Invalid parameter combination')
    if params.reinit_bn_stats:
        raise AssertionError('Namgyu thinks there is a problem with params.reinit_bn_stats. Plz consult.')
    if params.ut and not params.target_dataset:
        raise AssertionError('Invalid parameter combination')
    if params.ft_parts not in ["head", "body", "full"]:
        raise AssertionError('Invalid params.ft_parts: {}'.format(params.ft_parts))

    # Assign num_classes
    if params.dataset == 'miniImageNet':
        params.num_classes = 64
    elif params.dataset == 'tieredImageNet':
        params.num_classes = 351
    elif params.dataset == 'ImageNet':
        params.num_classes = 1000
    elif params.dataset == 'none':
        params.num_classes = 5
    else:
        raise ValueError('Invalid `dataset` argument: {}'.format(params.dataset))

    # Assign num_classes (*_new)
    if params.source_dataset == 'miniImageNet':
        params.num_classes = 64
    elif params.source_dataset == 'tieredImageNet':
        params.num_classes = 351
    elif params.source_dataset == 'ImageNet':
        params.num_classes = 1000
    elif params.source_dataset == 'none':
        params.num_classes = 5
    else:
        raise ValueError('Invalid `source_dataset` argument: {}'.format(params.source_dataset))

    # Default optimizers
    if params.optimizer is None:
        if params.model in ['simsiam', 'byol']:
            params.optimizer = 'adam'
        else:
            params.optimizer = 'sgd'
        print("Using default optimizer for model {}: {}".format(params.model, params.optimizer))

    # Default learning rate
    if params.lr is None:
        if params.model in ['simsiam', 'byol']:
            params.lr = 3e-4
        elif params.model in ['moco']:
            params.lr = 0.01
        else:
            params.lr = 0.1
        print("Using default lr for model {}: {}".format(params.model, params.lr))

    params.ft_train_body = params.ft_parts in ['body', 'full']
    params.ft_train_head = params.ft_parts in ['head', 'full']

    if params.pls_tag is None:
        params.pls_tag = params.pls

    return params


def get_init_file(checkpoint_dir):
    init_file = os.path.join(checkpoint_dir, 'initial.tar')
    return init_file

def get_assigned_file(checkpoint_dir, num, dataset_name=None):
    if dataset_name is None:
        assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    else:
        assign_file = os.path.join(checkpoint_dir, '{}_{:d}.tar'.format(dataset_name, num))
    return assign_file

def get_resume_file(checkpoint_dir, dataset_name=None):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        print('Warning: unable to locate *.tar checkpoint file in {}'.format(checkpoint_dir))
        return None

    if dataset_name is None:
        filelist = [ x for x in filelist if os.path.basename(x) != 'best_model.tar' and os.path.basename(x) != 'initial.tar']
        epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
        max_epoch = np.max(epochs)
        resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    else:
        filelist = [ x for x in filelist if os.path.basename(x) != 'best_model.tar' and os.path.basename(x) != '{}_initial.tar'.format(dataset_name)]
        epochs = np.array([int(os.path.splitext(os.path.basename(x))[0].split('_')[1]) for x in filelist])
        max_epoch = np.max(epochs)
        resume_file = os.path.join(checkpoint_dir, '{}_{:d}.tar'.format(dataset_name, max_epoch))

    return resume_file

def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
