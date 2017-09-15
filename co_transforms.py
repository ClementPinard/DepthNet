from __future__ import division
import torch
import random
import numpy as np
import types

'''Set of tranform random routines that takes both input and target as arguments,
in order to have random but coherent transformations.
inputs are ndarrays pairs and targets are ndarrays'''


class Compose(object):
    """Compose several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target, displacement):
        for t in self.co_transforms:
            input, target, displacement = t(input, target, displacement)
        return input, target, displacement


class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert(isinstance(array, np.ndarray))
        if array.ndim == 3:
            array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        return tensor.float()


class Clip(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, array):
        assert(isinstance(array, np.ndarray))
        return np.clip(array, self.x, self.y)


class Lambda(object):
    """Applies a lambda as a transform"""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, input, target, displacement):
        return self.lambd(input, target, displacement)


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, inputs, target, displacement):
        if random.random() < 0.5:
            inputs[0] = np.copy(np.fliplr(inputs[0]))
            inputs[1] = np.copy(np.fliplr(inputs[1]))
            target = np.copy(np.fliplr(target))
            displacement[0] *= -1
        return inputs, target, displacement


class RandomVerticalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, inputs, target, displacement):
        if random.random() < 0.5:
            inputs[0] = np.copy(np.flipud(inputs[0]))
            inputs[1] = np.copy(np.flipud(inputs[1]))
            target = np.copy(np.flipud(target))
            displacement[1] *= -1
        return inputs, target, displacement