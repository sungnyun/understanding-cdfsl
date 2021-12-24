from model.base import BaseModel
from model.simclr import SimCLR

_model_class_map = {
    'base': BaseModel,
    'simclr': SimCLR,
}


def get_model_class(key):
    if key in _model_class_map:
        return _model_class_map[key]
    else:
        raise ValueError('Invalid model: {}'.format(key))
