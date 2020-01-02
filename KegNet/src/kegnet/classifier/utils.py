import numpy as np

from kegnet.classifier.models import lenet
from kegnet.utils import data


def init_classifier(dataset):
    """
    Initialize a classifier based on the dataset.
    """
    d = data.to_dataset(dataset)
    if dataset == 'mnist':
        return lenet.LeNet5()


def count_parameters(model):
    """
    Count the parameters of a classifier.
    """
    size = 0
    for parameter in model.parameters():
        size += np.prod(parameter.shape)
    return size
