import math
from typing import List

import torch
from torch import nn

import backbone

BLOCK_NAMES = {
    'ResNet10': {
        '1.c1': 'trunk.4.C1',
        '1.b1': 'trunk.4.BN1',
        '1.c2': 'trunk.4.C2',
        '1.b2': 'trunk.4.BN2',
        '2.c1': 'trunk.5.C1',
        '2.b1': 'trunk.5.BN1',
        '2.c2': 'trunk.5.C2',
        '2.b2': 'trunk.5.BN2',
        '2.cs': 'trunk.5.shortcut',
        '2.bs': 'trunk.5.BNshortcut',
        '3.c1': 'trunk.6.C1',
        '3.b1': 'trunk.6.BN1',
        '3.c2': 'trunk.6.C2',
        '3.b2': 'trunk.6.BN2',
        '3.cs': 'trunk.6.shortcut',
        '3.bs': 'trunk.6.BNshortcut',
        '4.c1': 'trunk.7.C1',
        '4.b1': 'trunk.7.BN1',
        '4.c2': 'trunk.7.C2',
        '4.b2': 'trunk.7.BN2',
        '4.cs': 'trunk.7.shortcut',
        '4.bs': 'trunk.7.BNshortcut',
    },
    'ResNet18': {
        '1.0.c1': 'layer1.0.conv1',
        '1.0.b1': 'layer1.0.bn1',
        '1.0.c2': 'layer1.0.conv2',
        '1.0.b2': 'layer1.0.bn2',
        '1.1.c1': 'layer1.1.conv1',
        '1.1.b1': 'layer1.1.bn1',
        '1.1.c2': 'layer1.1.conv2',
        '1.1.b2': 'layer1.1.bn2',
        '2.0.c1': 'layer2.0.conv1',
        '2.0.b1': 'layer2.0.bn1',
        '2.0.c2': 'layer2.0.conv2',
        '2.0.b2': 'layer2.0.bn2',
        '2.0.cs': 'layer2.0.downsample.0',
        '2.0.bs': 'layer2.0.downsample.1',
        '2.1.c1': 'layer2.1.conv1',
        '2.1.b1': 'layer2.1.bn1',
        '2.1.c2': 'layer2.1.conv2',
        '2.1.b2': 'layer2.1.bn2',
        '3.0.c1': 'layer3.0.conv1',
        '3.0.b1': 'layer3.0.bn1',
        '3.0.c2': 'layer3.0.conv2',
        '3.0.b2': 'layer3.0.bn2',
        '3.0.cs': 'layer3.0.downsample.0',
        '3.0.bs': 'layer3.0.downsample.1',
        '3.1.c1': 'layer3.1.conv1',
        '3.1.b1': 'layer3.1.bn1',
        '3.1.c2': 'layer3.1.conv2',
        '3.1.b2': 'layer3.1.bn2',
        '4.0.c1': 'layer4.0.conv1',
        '4.0.b1': 'layer4.0.bn1',
        '4.0.c2': 'layer4.0.conv2',
        '4.0.b2': 'layer4.0.bn2',
        '4.0.cs': 'layer4.0.downsample.0',
        '4.0.bs': 'layer4.0.downsample.1',
        '4.1.c1': 'layer4.1.conv1',
        '4.1.b1': 'layer4.1.bn1',
        '4.1.c2': 'layer4.1.conv2',
        '4.1.b2': 'layer4.1.bn2',
    },
    'ResNet18_84x84': {
        '1.0.c1': 'layer1.0.conv1',
        '1.0.b1': 'layer1.0.bn1',
        '1.0.c2': 'layer1.0.conv2',
        '1.0.b2': 'layer1.0.bn2',
        '1.0.c3': 'layer1.0.conv3',
        '1.0.b3': 'layer1.0.bn3',
        '1.0.cs': 'layer1.0.downsample.0',
        '1.0.bs': 'layer1.0.downsample.1',
        '2.0.c1': 'layer2.0.conv1',
        '2.0.b1': 'layer2.0.bn1',
        '2.0.c2': 'layer2.0.conv2',
        '2.0.b2': 'layer2.0.bn2',
        '2.0.c3': 'layer2.0.conv3',
        '2.0.b3': 'layer2.0.bn3',
        '2.0.cs': 'layer2.0.downsample.0',
        '2.0.bs': 'layer2.0.downsample.1',
        '3.0.c1': 'layer3.0.conv1',
        '3.0.b1': 'layer3.0.bn1',
        '3.0.c2': 'layer3.0.conv2',
        '3.0.b2': 'layer3.0.bn2',
        '3.0.c3': 'layer3.0.conv3',
        '3.0.b3': 'layer3.0.bn3',
        '3.0.cs': 'layer3.0.downsample.0',
        '3.0.bs': 'layer3.0.downsample.1',
        '3.1.c1': 'layer3.1.conv1',
        '3.1.b1': 'layer3.1.bn1',
        '3.1.c2': 'layer3.1.conv2',
        '3.1.b2': 'layer3.1.bn2',
        '3.1.c3': 'layer3.1.conv3',
        '3.1.b3': 'layer3.1.bn3',
        '4.0.c1': 'layer4.0.conv1',
        '4.0.b1': 'layer4.0.bn1',
        '4.0.c2': 'layer4.0.conv2',
        '4.0.b2': 'layer4.0.bn2',
        '4.0.c3': 'layer4.0.conv3',
        '4.0.b3': 'layer4.0.bn3',
        '4.0.cs': 'layer4.0.downsample.0',
        '4.0.bs': 'layer4.0.downsample.1',
        '4.1.c1': 'layer4.1.conv1',
        '4.1.b1': 'layer4.1.bn1',
        '4.1.c2': 'layer4.1.conv2',
        '4.1.b2': 'layer4.1.bn2',
        '4.1.c3': 'layer4.1.conv3',
        '4.1.b3': 'layer4.1.bn3',
    }
}


def reset_layers(model, layers: List[str], model_name=None, init_state_dict=None):
    """
    Re-randomize or re-init layers. Re-init if `init_state_dict` is provided.

    :param model:
    :param layers:
    :param model_name:
    :param init_state_dict:
    :return:
    """
    if not model_name:
        model_name = model.feature.__class__.__name__
    mapper = BLOCK_NAMES[model_name]
    targets = dict()  # Dict[target_name, layer_type]
    for layer in layers:
        try:
            name = mapper[layer]
        except KeyError:
            raise KeyError('Invalid layer specifier {} for model {}'.format(layer, model_name))
        targets[name] = 'bn' if 'b' in layer else 'conv'

    consumed = set()
    with torch.no_grad():
        for name, p in model.named_parameters():
            for target, layer_type in targets.items():
                if target in name:
                    if init_state_dict:
                        p.data = init_state_dict[name]
                    else:
                        if layer_type == 'bn':
                            if 'weight' in name:
                                p.data.fill_(1.)
                            else:
                                p.data.fill_(0.)
                        else:
                            nn.init.kaiming_uniform_(p.data, a=math.sqrt(5))
                    consumed.add(target)

    remaining = set(targets.keys()) - consumed
    if remaining:
        raise AssertionError('Missing layers during rerandomization: {}'.format(remaining))


def main():
    print('Running unit tests for rerandomization module...')
    for model_name, mappings in BLOCK_NAMES.items():
        Model = getattr(backbone, model_name)
        model = Model()

        # Make sure all layer names are valid
        layers = list(mappings.values())
        consumed = set()
        for name, param in model.named_parameters():
            for target in list(layers):
                if target in name:
                    consumed.add(target)

        remaining = set(layers) - consumed
        if remaining:
            raise Exception(
                'Invalid layer names defined in code for {}: {}'.format(model_name, sorted(list(remaining))))

    print('All tests passed!')


if __name__ == '__main__':
    main()
