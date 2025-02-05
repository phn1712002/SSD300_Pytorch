from datasets import coco128, detection_collate
from utils import tools
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import wandb

# Lấy đường dẫn tuyệt đối của file hiện tại (train.py)
current_file_path = os.path.abspath(__file__)
current_folder_path = os.path.dirname(current_file_path)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--hyp', default=os.path.join(current_folder_path, 'hyp.yaml'), type=str,
                    help='File yaml hyp of model')
parser.add_argument('--data', default=os.path.join(current_folder_path, 'coco128.yaml'), type=str,
                    help='File yaml dataset of model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=False, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--project', default="SSD300", type=str,
                    help='Name project')
parser.add_argument('--name', default=tools.generate_random_name(4), type=str,
                    help='Name run exp')
parser.add_argument("--save_period", type=int, default=-1, 
                    help="Save checkpoint every x epochs (disabled if < 1)")
parser.add_argument("--log_wandb", type=bool, default=True, 
                    help="Enable wandb")
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if args.batch_size < 0:
    batch_size = 1

def train():
    
    hyp = tools.load_yaml_to_dict(args.hyp)
    cfg = hyp['cfg']
    aug = hyp['augmentations']
    opt = tools.convert_dict_values_to_float(hyp['opt'])
    p_detect = hyp['detect']

    if args.log_wandb:
        wandb.init(project=args.project, 
                    name=args.name, 
                    config=hyp, 
                    save_code=True)

    dataset = coco128.COCO_128Detection(path_yaml=args.data, transform=SSDAugmentation(**aug, size=300))

    ssd_net = build_ssd(phase='train', size=300, num_classes=dataset.num_classes, p_detect=p_detect, cfg=cfg)
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        path_model = os.path.join(current_folder_path, 'weights/') + 'vgg16_reducedfc.pth'
        print('Loading base network...')
        vgg_weights = torch.load(path_model)
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=opt['lr'], momentum=opt['momentum'],
                          weight_decay=opt['weight_decay'])
    criterion = MultiBoxLoss(dataset.num_classes, 0.5, True, 0, True, 3, 0.5,
                             False, cfg['variance'], args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if not args.cuda:
      data_loader = data.DataLoader(dataset, args.batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=True, collate_fn=detection_collate,
                                    pin_memory=True)
    else:
      data_loader = data.DataLoader(dataset, args.batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=True, collate_fn=detection_collate,
                                    pin_memory=True,
                                    generator=torch.Generator(device='cuda'))

    # create batch iterator
    path_folder_save = f"./{args.project}/{args.name}/"
    batch_iterator = iter(data_loader)
    for iteration in range(0, cfg['max_iter']):
        if iteration != 0 and (iteration % epoch_size == 0):
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in opt['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, opt['gamma'], step_index)

        # load train data
        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data
        conf_loss += loss_c.data

        # Log to wandb
        if args.log_wandb:
            wandb.log({"loc_loss": loc_loss, "conf_loss": conf_loss, "loss": loss})

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data), end=' ')

    
        if args.save_period != -1:
            if iteration != 0 and iteration % args.save_period == 0:
                if not os.path.exists(path_folder_save):
                    os.mkdir(path_folder_save)

                print('Saving state, iter:', iteration)
                torch.save(ssd_net.state_dict(), os.path.join(path_folder_save, f"iter_{repr(iteration)}.pth"))
    torch.save(ssd_net.state_dict(), os.path.join(path_folder_save, f"last.pth"))
    
    if args.log_wandb:
        wandb.finish()
               


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = opt['lr'] * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

if __name__ == '__main__':
    train()
