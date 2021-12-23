import numpy as np
import os
import glob
import argparse
import backbone

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--dataset'     , default='miniImageNet',        help='training base model')
    parser.add_argument('--model'       , default='ResNet10',      help='backbone architecture') 
    parser.add_argument('--method'      , default='baseline',   help='baseline/protonet/maml') 
    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training')
    parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ')
    parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support') 
    parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ') 
    parser.add_argument('--freeze_backbone'   , action='store_true', help='Freeze the backbone network for finetuning')
    parser.add_argument('--models_to_use', '--names-list', nargs='+', default=['miniImageNet', 'caltech256', 'DTD', 'cifar100', 'CUB'], help='pretained model to use')
    parser.add_argument('--fine_tune_all_models'   , action='store_true',  help='fine-tune each model before selection') #still required for save_features.py and test.py to find the model path correctly

    if script == 'train':
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
        parser.add_argument('--dataset_names', nargs='+', type=str, default=["miniImageNet", "CropDisease", "EuroSAT", "ISIC", "ChestX"], help='CD-FSL datasets to fine-tune')
        parser.add_argument('--use_simclr_clf', action='store_true', help ='Use pre-trained SimCLR projection head')

        # For STARTUP-like split
        parser.add_argument('--startup_split', action='store_true', help ='Use 80% of dataset, similar to STARTUP. Enabled automatically for simclr_finetune.')
    elif script == 'save_features':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter', default=-1, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1')
    elif script == 'test':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--adaptation'  , action='store_true', help='further adaptation in test time or not')
    else:
        raise ValueError('Unknown script')

    params = parser.parse_args()

    # Double-checking parameters
    if params.freeze_bn and not params.track_bn:
        raise AssertionError('Plz check freeze_bn and track_bn arguments.')
    if params.reinit_bn_stats:
        raise AssertionError('Namgyu thinks there is a problem with params.reinit_bn_stats. Plz consult.')

    return params


def get_init_file(checkpoint_dir):
    init_file = os.path.join(checkpoint_dir, 'initial.tar')
    return init_file

def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
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
