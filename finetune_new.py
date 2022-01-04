import copy
import json
import math
import os

import pandas as pd
import torch.nn as nn

from backbone import get_backbone_class
from datasets.dataloader import get_labeled_episodic_dataloader
from io_utils import parse_args
from model import get_model_class
from model.classifier_head import get_classifier_head_class
from paths import get_output_directory, get_ft_output_directory, get_ft_train_history_path, get_ft_test_history_path, \
    get_final_pretrain_state_path, get_pretrain_state_path, get_ft_params_path
from utils import *


def main(params):
    base_output_dir = get_output_directory(params)
    output_dir = get_ft_output_directory(params)
    print('Running fine-tune with output folder:')
    print(output_dir)
    print()

    # Settings
    n_episodes = 600
    bs = params.ft_batch_size
    w = params.n_way
    s = params.n_shot
    q = params.n_query_shot
    # Whether to optimize for fixed features (when there is no augmentation and only head is updated)
    use_fixed_features = params.ft_augmentation is None and params.ft_parts == 'head'

    # Model
    backbone = get_backbone_class(params.backbone)()
    body = get_model_class(params.model)(backbone, params)
    if params.ft_features is not None:
        if params.ft_features not in body.supported_feature_selectors:
            raise ValueError(
                'Feature selector "{}" is not supported for model "{}"'.format(params.ft_features, params.model))

    # Dataloaders
    # Note that both dataloaders sample identical episodes, via episode_seed
    support_epochs = 1 if use_fixed_features else params.ft_epochs
    support_loader = get_labeled_episodic_dataloader(params.target_dataset, n_way=w, n_shot=s, support=True,
                                                     n_query_shot=q, n_episodes=n_episodes, n_epochs=support_epochs,
                                                     augmentation=params.ft_augmentation,
                                                     unlabeled_ratio=params.unlabeled_ratio,
                                                     split_seed=params.split_seed, episode_seed=params.ft_episode_seed)
    query_loader = get_labeled_episodic_dataloader(params.target_dataset, n_way=w, n_shot=s, support=False,
                                                   n_query_shot=q, n_episodes=n_episodes, augmentation=None,
                                                   unlabeled_ratio=params.unlabeled_ratio, split_seed=params.split_seed,
                                                   episode_seed=params.ft_episode_seed)
    assert (len(query_loader) == n_episodes)
    assert (len(support_loader) == n_episodes * support_epochs)

    query_iterator = iter(query_loader)
    support_iterator = iter(support_loader)
    support_batches = math.ceil(w * s / bs)

    # Output (history, params)
    train_history_path = get_ft_train_history_path(output_dir)
    test_history_path = get_ft_test_history_path(output_dir)
    params_path = get_ft_params_path(output_dir)
    print('Saving finetune params to {}'.format(params_path))
    print('Saving finetune train history to {}'.format(train_history_path))
    print('Saving finetune validation history to {}'.format(train_history_path))
    with open(params_path, 'w') as f:
        json.dump(vars(params), f, indent=4)
    df_train = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                            columns=['epoch{}'.format(e + 1) for e in range(params.ft_epochs)])
    df_test = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                           columns=['epoch{}'.format(e + 1) for e in range(params.ft_epochs)])

    # Pre-train state
    if params.ft_pretrain_epoch is None:
        body_state_path = get_final_pretrain_state_path(base_output_dir)
    else:
        body_state_path = get_pretrain_state_path(base_output_dir, params.ft_pretrain_epoch)
    if not os.path.exists(body_state_path):
        raise ValueError('Invalid pre-train state path: ' + body_state_path)
    print('Using pre-train state:')
    print(body_state_path)
    print()
    state = torch.load(body_state_path)

    # Loss function
    loss_fn = nn.CrossEntropyLoss().cuda()

    print('Starting fine-tune')
    if use_fixed_features:
        print('Running optimized fixed-feature fine-tuning (no augmentation, fixed body)')
    print()

    for episode in range(n_episodes):
        # Reset models for each episode
        body.load_state_dict(copy.deepcopy(state))  # note, override model.load_state_dict to change this behavior.
        head = get_classifier_head_class(params.ft_head)(body.final_feat_dim, params.n_way,
                                                         params)  # TODO: apply ft_features
        body.cuda()
        head.cuda()

        opt_params = []
        if params.ft_train_head:
            opt_params.append({'params': head.parameters()})
        if params.ft_train_body:
            opt_params.append({'params': body.parameters()})
        optimizer = torch.optim.SGD(opt_params, lr=1e-2, momentum=0.9, dampening=0.9, weight_decay=0.001)

        # Labels are always [0, 0, ..., 1, ..., w-1]
        x_support = None
        f_support = None
        y_support = torch.arange(w).repeat_interleave(s).cuda()
        x_query = next(query_iterator)[0].cuda()
        f_query = None
        y_query = torch.arange(w).repeat_interleave(q).cuda()

        if use_fixed_features:  # load data and extract features once per episode
            with torch.no_grad():
                x_support, _ = next(support_iterator)
                x_support = x_support.cuda()
                f_support = body.forward_features(x_support, params.ft_features)
                f_query = body.forward_features(x_query, params.ft_features)

        train_acc_history = []
        test_acc_history = []
        for epoch in range(params.ft_epochs):
            # Train
            body.train()
            head.train()
            optimizer.zero_grad()

            if not use_fixed_features:  # load data every epoch
                x_support, _ = next(support_iterator)
                x_support = x_support.cuda()

            total_loss = 0
            correct = 0
            indices = np.random.permutation(w * s)
            for i in range(support_batches):
                start_index = i * bs
                end_index = min(i * bs + bs, w * s)
                batch_indices = indices[start_index:end_index]
                y = y_support[batch_indices]

                if use_fixed_features:
                    f = f_support[batch_indices]
                else:
                    f = body.forward_features(x_support[batch_indices], params.ft_features)
                p = head(f)

                correct += torch.eq(y, p.argmax(dim=1)).sum()
                loss = loss_fn(p, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            train_loss = total_loss / support_batches
            train_acc = correct / (w * s)

            # Evaluation
            body.eval()
            head.eval()

            if params.ft_intermediate_test or epoch == params.ft_epochs - 1:
                with torch.no_grad():
                    if not use_fixed_features:
                        f_query = body.forward_features(x_query, params.ft_features)
                    p_query = head(f_query)
                test_acc = torch.eq(y_query, p_query.argmax(dim=1)).sum() / (w * q)
            else:
                test_acc = torch.tensor(0)

            print_epoch_logs = False
            if print_epoch_logs and (epoch + 1) % 10 == 0:
                fmt = 'Epoch {:03d}: Loss={:6.3f} Train ACC={:6.3f} Test ACC={:6.3f}'
                print(fmt.format(epoch + 1, train_loss, train_acc, test_acc))

            train_acc_history.append(train_acc.item())
            test_acc_history.append(test_acc.item())

        df_train.loc[episode + 1] = train_acc_history
        df_train.to_csv(train_history_path)
        df_test.loc[episode + 1] = test_acc_history
        df_test.to_csv(test_history_path)

        fmt = 'Episode {:03d}: train_loss={:6.4f} train_acc={:6.2f} test_acc={:6.2f}'
        print(fmt.format(episode, train_loss, train_acc_history[-1] * 100, test_acc_history[-1] * 100))

    fmt = 'Final Results: Acc={:5.2f} Std={:5.2f}'
    print(fmt.format(df_test.mean()[-1] * 100, 1.96 * df_test.std()[-1] / np.sqrt(n_episodes) * 100))

    print('Saved history to:')
    print(train_history_path)
    print(test_history_path)
    df_train.to_csv(train_history_path)
    df_test.to_csv(test_history_path)


if __name__ == '__main__':
    np.random.seed(10)
    params = parse_args('train')

    targets = params.target_dataset
    if targets is None:
        targets = [targets]
    elif len(targets) > 1:
        print('#' * 80)
        print("Running finetune iteratively for multiple target datasets: {}".format(targets))
        print('#' * 80)

    for target in targets:
        params.target_dataset = target
        main(params)
