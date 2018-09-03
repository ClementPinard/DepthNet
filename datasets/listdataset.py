import torch.utils.data as data
from imageio import imread
import numpy as np


def default_loader(root, path_imgs, path_depth):
    imgs = [imread(root/path) for path in path_imgs]
    depth = np.load(root/path_depth)
    return [imgs, depth]


class ListDataset(data.Dataset):
    def __init__(self, root, path_list, transform=None, target_transform=None,
                 co_transform=None, loader=default_loader):

        self.root = root
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader

    def __getitem__(self, index):
        inputs, target, displacement = self.path_list[index]
        inputs, target = self.loader(self.root, inputs, target)
        if self.co_transform is not None:
            inputs, target, displacement = self.co_transform(inputs, target, displacement)
        if self.transform is not None:
            inputs[0] = self.transform(inputs[0])
            inputs[1] = self.transform(inputs[1])
        if self.target_transform is not None:
            target = self.target_transform(target)

        return inputs, target, displacement

    def __len__(self):
        return len(self.path_list)
