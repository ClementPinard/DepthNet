import random
import math
from .listdataset import ListDataset
from .scenelistdataset import SceneListDataset
import json
from path import Path
import numpy as np


def make_dataset(root_dir, split=0, shift=3, seed=None):
    """Will search for subfolder and will read metadata json files."""
    global args
    random.seed(seed)
    scenes = []
    for sub_dir in root_dir.dirs():
        metadata_path = sub_dir/'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        for scene in metadata['scenes']:
            scene['subdir'] = sub_dir.basename()
        scenes.extend(metadata['scenes'])

    assert(len(scenes) > 0)
    random.shuffle(scenes)
    split_index = math.floor(len(scenes)*split/100)
    assert(split_index >= 0 and split_index <= len(scenes))
    train_scenes = scenes[:split_index]
    test_images = []
    if split_index < len(scenes):
        for scene in scenes[split_index+1:]:
            imgs = scene['imgs']
            for i in range(len(imgs)-shift):
                img_pair = [str(scene['subdir']/imgs[i]), str(scene['subdir']/imgs[i+shift])]
                depth = str(scene['subdir']/scene['depth'][i + shift])
                displacement = np.array(scene['speed']).astype(np.float32)*shift*scene['time_step']
                test_images.append(
                    [img_pair,
                     depth,
                     displacement]
                )
    return (train_scenes, test_images)


def still_box(root, transform=None, target_transform=None,
              co_transform=None, split=80, shift=3, seed=None):
    root = Path(root)
    train_scenes, test_list = make_dataset(root, split, shift, seed)
    train_dataset = SceneListDataset(root, train_scenes, shift, transform, target_transform, co_transform)
    test_dataset = ListDataset(root, test_list, transform, target_transform)

    return train_dataset, test_dataset