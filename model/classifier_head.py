from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, params):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.fc(x)
        return x


class TwoLayerMLPClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, params):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


CLASSIFIER_HEAD_CLASS_MAP = {
    'linear': LinearClassifier,
    'two_layer_mlp': TwoLayerMLPClassifier,
}

def get_classifier_head_class(key):
    if key in CLASSIFIER_HEAD_CLASS_MAP:
        return CLASSIFIER_HEAD_CLASS_MAP[key]
    else:
        raise ValueError('Invalid classifier head specifier: {}'.format(key))
