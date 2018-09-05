import shutil
import datetime
import torch
from torch.autograd import Variable
from path import Path
import numpy as np


def set_arguments(parser):
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--activation-function', default=None,
                        help='activation function to apply to DepthNet')
    parser.add_argument('--bn', action='store_true',
                        help='activate batchNorm (overwritten if pretrained model)')
    parser.add_argument('--clamp', action='store_true',
                        help='activate depth clamping to (10,60) in forward pass')
    parser.add_argument('--solver', default='sgd', choices=['adam', 'sgd'],
                        help='solvers: adam | sgd')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=55, type=int, metavar='N',
                        help='number of total epochs to run (default: 55')
    parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                        help='manual epoch size (will match dataset size if not set)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                        help='beta parameters for adam')
    parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                        metavar='W', help='weight decay (default: 4e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', default=None,
                        help='path to pre-trained model')
    parser.add_argument('--seed', default=0, type=int, help='seed for random functions, test/train split, network initialization')
    parser.add_argument('-s', '--split', default=90, type=float, metavar='%',
                        help='split percentage of train samples vs test (default: 90)')
    parser.add_argument('--log-summary', default='progress_log_summary.csv',
                        help='csv where to save per-epoch train and test stats')
    parser.add_argument('--log-full', default='progress_log_full.csv',
                        help='csv where to save per-gradient descent train stats')
    parser.add_argument('--no-date', action='store_true',
                        help='don\'t append date timestamp to folder')
    parser.add_argument('--loss', default='L1', help='loss function to apply to multiScaleCriterion : L1 (default)| SmoothL1| MSE')
    parser.add_argument('--log-output', action='store_true', help='logs in tensorboard some outputs of the network during test phase. Needs OpenCV 3')


def set_params(parser, with_confidence=False):
    args = parser.parse_args()
    args.data = Path(args.data)
    folder_name = args.data.normpath().name
    arch_string = 'DepthNet'
    if with_confidence:
        arch_string += '_confidence'
    if args.activation_function is not None:
        arch_string += '_'+args.activation_function
    if args.bn:
        arch_string += '_bn'
    if args.clamp:
        arch_string += '_clamp'
    args.arch = arch_string

    save_path = '{},{}epochs{},b{},lr{}'.format(
        args.solver,
        args.epochs,
        ',epochSize'+str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.lr)
    save_path = Path(save_path)
    if not args.no_date:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_path = save_path/timestamp
    args.save_path = Path('Results')/arch_string/folder_name/save_path
    print('=> will save everything to {}'.format(save_path))
    args.save_path.makedirs_p()
    return args


def save_checkpoint(save_path, state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, save_path/filename)
    if is_best:
        shutil.copyfile(save_path/filename, save_path/'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    # Set the learning rate to the initial LR decayed by 2 after 300K iterations, 400K and 500K

    if epoch == 19 or epoch == 44:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/2
    if epoch == 30 or epoch == 53:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/5


def tensor2array(tensor, max_value=255, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            import cv2
            if int(cv2.__version__[0]) >= 3:
                color_cvt = cv2.COLOR_BGR2RGB
            else:  # 2.4
                color_cvt = cv2.cv.CV_BGR2RGB
            if colormap == 'rainbow':
                colormap = cv2.COLORMAP_RAINBOW
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (255*tensor.squeeze().numpy()/max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32)/255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy()/max_value).clip(0,1)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy()*0.5
    return array