import torch

from scipy.misc import imread, imsave, imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

import torch.nn.functional as F
from models import DepthNet
from util import tensor2array

parser = argparse.ArgumentParser(description='Inference script for DepthNet img must be with no rotation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
parser.add_argument("--output-raw", action='store_true', help="save raw numpy depth array")

parser.add_argument("--pretrained", required=True, type=str, help="pretrained DepthNet path")
parser.add_argument("--frame-shift", default=1, type=int, help="temporal shift between imgs of the pairs feeded to the network")
parser.add_argument("--img-height", default=512, type=int, help="Image height")
parser.add_argument("--img-width", default=512, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")

parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")


@torch.no_grad()
def main():
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if not(args.output_disp or args.output_depth):
        print('You must at least output one value !')
        return

    weights = torch.load(args.pretrained)
    depth_net = DepthNet(batch_norm=weights['bn'],
                         depth_activation=weights['activation_function'],
                         clamp=weights['clamp']).to(device)
    print("running inference with {} ...".format(weights['arch']))
    depth_net.load_state_dict(weights['state_dict'])
    depth_net.eval()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = [dataset_dir/file for file in f.read().splitlines()]
    else:
        test_files = sorted(sum([dataset_dir.files('*.{}'.format(ext)) for ext in args.img_exts], []))

    print('{} files to test'.format(len(test_files)))

    for file1, file2 in tqdm(zip(test_files[:-args.frame_shift], test_files[args.frame_shift:])):

        img1 = imread(file1).astype(np.float32)
        img2 = imread(file2).astype(np.float32)

        h,w,_ = img1.shape
        assert(img1.shape == img2.shape), "img1 and img2 must be the same size"
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            img1 = imresize(img1, (args.img_height, args.img_width)).astype(np.float32)
            img2 = imresize(img2, (args.img_height, args.img_width)).astype(np.float32)
        imgs = np.concatenate([np.transpose(img1, (2, 0, 1)), np.transpose(img2, (2, 0, 1))])

        tensor_imgs = torch.from_numpy(imgs).unsqueeze(0).to(device)
        tensor_imgs = ((tensor_imgs/255 - 0.5)/0.2)

        output_depth = depth_net(tensor_imgs)

        upscaled_output = F.interpolate(output_depth.unsqueeze(1), (h,w), mode='bilinear', align_corners=False)[0,0]

        if args.output_disp:
            disp = 1/upscaled_output
            disp = (255*tensor2array(disp, max_value=None, colormap='bone')).astype(np.uint8)
            imsave(output_dir/'{}_disp{}'.format(file2.namebase, file2.ext), disp.transpose(1,2,0))
        if args.output_depth:
            depth = (255*tensor2array(upscaled_output, max_value=100, colormap='rainbow')).astype(np.uint8)
            imsave(output_dir/'{}_depth{}'.format(file2.namebase, file2.ext), depth.transpose(1,2,0))
        if args.output_raw:
            np.save(output_dir/'{}_depth.npy'.format(file2.namebase), output_depth.cpu())


if __name__ == '__main__':
    main()
