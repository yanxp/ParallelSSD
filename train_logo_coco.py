from __future__ import print_function
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
import argparse
import _init_paths
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
from data import Logoroot, Logo_512,Logo_300, VOCroot, COCOroot, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300, AnnotationTransform, COCODetection, VOCDetection, LogoDetection, detection_collate, BaseTransform, preproc,collate_minibatch
from layers.modules import MultiBoxLoss
from layers.functions import PriorBox
import time

import nn as mynn

import pickle
parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')
parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='Logo',
                    help='VOC ,Logo or COCO dataset')
parser.add_argument(
    '--basenet', default='./weights/vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=32,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate',
                    default=4e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument(
    '--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')
parser.add_argument('-max','--max_epoch', default=160,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True,
                    type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='./weights/',
                    help='Location to save checkpoint models')
args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset == 'Logo':
    train_sets = ['train']
    cfg = (Logo_300, Logo_512)[args.size == '512']
else:
    train_sets = [('2014', 'train'),('2014', 'valminusminival')]
    cfg = (COCO_300, COCO_512)[args.size == '512']

if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net
    cfg = COCO_mobile_300
else:
    print('Unkown version!')

img_dim = (300,512)[args.size=='512']
rgb_means = ((104, 117, 123),(103.94,116.78,123.68))[args.version == 'RFB_mobile']
p = (0.6,0.2)[args.version == 'RFB_mobile']
#num_classes = (150, 81)[args.dataset == 'COCO']
fs = open('data/logo_name.txt','r') 
LOGO_CLASSES = [ eval(name) for name in fs.readline().strip().split(',')]
fs.close()
num_classes = len(LOGO_CLASSES)


Logo_512['numpergpu'] = args.batch_size//args.ngpu

batch_size = args.batch_size
weight_decay = 0.0005
gamma = 0.1
momentum = 0.9

net = build_net('train', img_dim, num_classes)
print(net)
if args.resume_net == None:
    base_weights = torch.load(args.basenet)
    print('Loading base network...')
    net.base.load_state_dict(base_weights)

    def xavier(param):
        init.xavier_uniform(param)

    def weights_init(m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0

    print('Initializing weights...')
# initialize newly added layers' weights with kaiming_normal method
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)
    net.Norm.apply(weights_init)
    if args.version == 'RFB_E_vgg':
        net.reduce.apply(weights_init)
        net.up_reduce.apply(weights_init)

else:
# load resume network
    if 'RFB512_E_34_4' in args.resume_net:
        print('loading coco pretrain...')
        state_dict = torch.load(args.resume_net)
        from collections import OrderedDict
        conv0 = nn.Conv2d(512, num_classes*6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        conv1 = nn.Conv2d(1024,num_classes*6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        conv2 = nn.Conv2d(512, num_classes*6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        conv3 = nn.Conv2d(256, num_classes*6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        conv4 = nn.Conv2d(256, num_classes*6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        conv5 = nn.Conv2d(256, num_classes*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        conv6 = nn.Conv2d(256, num_classes*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
            if 'conf.0.weight' in k:
                new_state_dict[name] = conv0.weight
            if 'conf.0.bias' in k:
                new_state_dict[name] = conv0.bias
            if 'conf.1.weight' in k:
                new_state_dict[name] = conv1.weight
            if 'conf.1.bias' in k:
                new_state_dict[name] = conv1.bias
            if 'conf.2.weight' in k:
                new_state_dict[name] = conv2.weight
            if 'conf.2.bias' in k:
                new_state_dict[name] = conv2.bias
            if 'conf.3.weight' in k:
                new_state_dict[name] = conv3.weight
            if 'conf.3.bias' in k:
                new_state_dict[name] = conv3.bias
            if 'conf.4.weight' in k:
                new_state_dict[name] = conv4.weight
            if 'conf.4.bias' in k:
                new_state_dict[name] = conv4.bias
            if 'conf.5.weight' in k:
                new_state_dict[name] = conv5.weight
            if 'conf.5.bias' in k:
                new_state_dict[name] = conv5.bias
            if 'conf.6.weight' in k:
                new_state_dict[name] = conv6.weight
            if 'conf.6.bias' in k:
                new_state_dict[name] = conv6.bias
        net.load_state_dict(new_state_dict)
    else:
        # load resume network
        print('Loading trained logo network...')
        state_dict = torch.load(args.resume_net)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

net = mynn.DataParallel(net, cpu_keywords=['target'], minibatch=True)

if args.cuda:
    net.cuda()
    cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)

#optimizer = optim.RMSprop(net.parameters(), lr=args.lr,alpha = 0.9, eps=1e-08,
#                      momentum=args.momentum, weight_decay=args.weight_decay)

criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)
priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()



def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    if args.dataset == 'Logo':
        dataset = LogoDetection(Logoroot, train_sets, preproc(
            img_dim, rgb_means, p), AnnotationTransform())
    elif args.dataset == 'COCO':
        dataset = COCODetection(COCOroot, train_sets, preproc(
            img_dim, rgb_means, p))
    else:
        print('Only VOC and COCO are supported now!')
        return

    epoch_size = len(dataset) // args.batch_size
    max_iter = args.max_epoch * epoch_size

    stepvalues_Logo = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
    stepvalues_COCO = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
    stepvalues = (stepvalues_Logo,stepvalues_COCO)[args.dataset=='COCO']
    print('Training',args.version, 'on', dataset.name)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    lr = args.lr
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size,
                                                  shuffle=True, num_workers=args.num_workers, collate_fn=collate_minibatch))
            #batch_iterator = iter(data.DataLoader(dataset, batch_size,
            #                                      shuffle=True, num_workers=args.num_workers,collate_fn=collate_minibatch))
            loc_loss = 0
            conf_loss = 0
            if (epoch % 5 == 0 and epoch > 0) or (epoch % 5 ==0 and epoch > 200):
                torch.save(net.state_dict(), args.save_folder+args.version+'_'+args.dataset + '_epoches_'+
                           repr(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        
        # load train data
        samples = next(batch_iterator)
        # import pdb;pdb.set_trace()
        #from IPython import embed; embed()
        if args.cuda:
            # samples['image'] = Variable(samples['image'])

            for key in samples:
                if key != 'target':  # roidb is a list of ndarrays with inconsistent length
                    samples[key] = list(map(Variable, samples[key]))

            #targets = [Variable(anno.cuda()) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno) for anno in targets]
        # forward
        t0 = time.time()
        #out = net(images,targets)
        #samples = {'images':images,'targets':targets}
        # backprop
        optimizer.zero_grad()
        
        return_dict = net(**samples)
        
        loss_l = return_dict['loss_l'].mean()
        loss_c = return_dict['loss_c'].mean()
        #loss_l, loss_c = criterion(out, priors, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        load_t1 = time.time()
        if iteration % 10 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' || L: %.4f C: %.4f||' % (
                loss_l.item(),loss_c.item()) + 
                'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))

    torch.save(net.state_dict(), args.save_folder +
               'Final_' + args.version +'_' + args.dataset+ '.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * 5) 
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
