import torch.nn as nn
import torch.nn.functional as F


def depth_metric_reconstruction_loss(depth, target, weights=None, loss='L1', normalize=False):
    def one_scale(depth, target, loss_function, normalize):
        b, h, w = depth.size()

        target_scaled = F.interpolate(target.unsqueeze(1), size=(h, w), mode='area')[:,0]

        diff = depth-target_scaled

        if normalize:
            diff = diff/target_scaled

        return loss_function(diff, depth.detach()*0)

    if weights is not None:
        assert(len(weights) == len(depth))
    else:
        weights = [1 for d in depth]
    if type(depth) not in [list, tuple]:
        depth = [depth]

    if type(loss) is str:
        assert(loss in ['L1', 'MSE', 'SmoothL1'])

        if loss == 'L1':
            loss_function = nn.L1Loss()
        elif loss == 'MSE':
            loss_function = nn.MSELoss()
        elif loss == 'SmoothL1':
            loss_function = nn.SmoothL1Loss()
    else:
        loss_function = loss

    loss_output = 0
    for d, w in zip(depth, weights):
        loss_output += w*one_scale(d, target, loss_function, normalize)
    return loss_output