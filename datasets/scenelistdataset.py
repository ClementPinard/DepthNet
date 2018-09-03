import torch.utils.data as data
from imageio import imread
import numpy as np


def default_loader(root, path_imgs, path_depth):
    imgs = [imread(root/path) for path in path_imgs]
    depth = np.load(root/path_depth)
    return [imgs, depth]


class SceneListDataset(data.Dataset):
    def __init__(self, root, scene_list, shift=3, transform=None, target_transform=None,
                 co_transform=None, loader=default_loader):

        self.root = root
        self.scene_list = scene_list
        self.indices = []
        for i, scene in enumerate(scene_list):
            self.indices.extend([i for j in scene['imgs']])
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader
        self.shift = shift

    def __getitem__(self, index):
        scene = self.scene_list[self.indices[index]]

        i1 = np.random.randint(0, len(scene['imgs']))
        shift = round(2*self.shift*np.random.uniform())
        i2 = min(len(scene['imgs'])-1, i1+shift)
        displacement = scene['time_step']*np.array(scene['speed']).astype(np.float32)*self.shift

        if np.random.uniform() > 0.5:
            # swap i1 and i2
            i1, i2 = i2, i1
            displacement *= -1

        inputs = [scene['imgs'][i1], scene['imgs'][i2]]
        target = scene['depth'][i2]
        inputs, target = self.loader(self.root/scene['subdir'], inputs, target)

        if i1 == i2:
            target.fill(100)
        else:
            target *= self.shift/np.abs(i2-i1)
        if self.co_transform is not None:
            inputs, target, displacement = self.co_transform(inputs, target, displacement)
        if self.transform is not None:
            inputs[0] = self.transform(inputs[0])
            inputs[1] = self.transform(inputs[1])
        if self.target_transform is not None:
            target = self.target_transform(target)
        return inputs, target, displacement

    def __len__(self):
        return len(self.indices)
