from __future__ import print_function
import sys
import os
import cv2
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from data import Logoroot, VOCroot,COCOroot 
from data import AnnotationTransform,LogoDetection, COCODetection, VOCDetection, BaseTransform, VOC_300,VOC_512,COCO_300,COCO_512, COCO_mobile_300,Logo_300, Logo_512
from data.data_augment import preproc
import torch.utils.data as data
from layers.functions import Detect,PriorBox
from utils.nms_wrapper import nms
from utils.timer import Timer
from tqdm import tqdm
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description='Receptive Field Block Net')

parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='Logo',
                    help='VOC,Logo or COCO version')
parser.add_argument('-m', '--trained_model', default='weights/RFB300_80_5.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--cpu', default=False, type=bool,
                    help='Use cpu nms')
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset == 'Logo':
    cfg = (Logo_300, Logo_512)[args.size == '512']
else:
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

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

fs = open('data/19logo_name.txt','r') 
LOGO_CLASSES = [ eval(name) for name in fs.readline().strip().split(',')]
fs.close()

def test_net(num_classes, save_folder, net, detector, cuda,transform, max_per_image=300, thresh=0.005):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    # num_images = len(testset)
    # num_classes = (150, 81)[args.dataset == 'COCO']

    jpeg = 'data/19/JPEGImages'
    anno = 'data/19/Annotations'
    path = 'data/19/ImageSets/Main/test.txt'
    res = 'result'
    for prefix_name in tqdm(open(path,'r').readlines()):
        
        img_name = prefix_name.strip() + '.jpg'
        img_path = os.path.join(jpeg,img_name)
        xml_name = prefix_name.strip() + '.xml'
        xml_path = os.path.join(anno,xml_name)
        image = cv2.imread(img_path)

        target =  ET.parse(xml_path).getroot()

        assert image.shape[2] == 3
        #img = testset.pull_image(i)
        scale = torch.Tensor([image.shape[1], image.shape[0],image.shape[1],image.shape[0]])
        img = transform(image)
        #img = testset.pull_image(i)
        with torch.no_grad():
            x = img[0].unsqueeze(0)
            if cuda:
                x = x.cuda()
                scale = scale.cuda()

        out = net(x)      # forward pass
        boxes, scores = detector.forward(out,priors)
        
        boxes = boxes[0]
        scores=scores[0]

        
        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                #all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            keep = nms(c_dets, 0.45, force_cpu=args.cpu)
            c_dets = c_dets[keep, :]
            inds = np.where(c_dets[:, -1] >= 0.5)[0]
            label=LOGO_CLASSES[j]
            if not os.path.exists(os.path.join(res,label)):
                os.mkdir(os.path.join(res,label))
            for i in inds:
                print(label)
                coords = c_dets[i, :4]
                score = c_dets[i, -1]
                cv2.rectangle(image, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), COLORS[0], 2)
                cv2.putText(image, '{label}: {score:.3f}'.format(label=LOGO_CLASSES[j], score=score), (int(coords[0]), int(coords[1])), FONT, 1, COLORS[0], 2)
        #cv2.imwrite(os.path.join(res,label,label+img_name), image)
        name = 'outliers'
        for obj in target.iter('object'):
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = float(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            name = obj.find('name').text.lower().strip()
            if name.endswith('text'):
                name = name.split('_')[0]
            if name not in LOGO_CLASSES:
                continue
            if not os.path.exists(os.path.join(res,name)):
                os.mkdir(os.path.join(res,name))
            cv2.rectangle(image, (int(bndbox[0]), int(bndbox[1])), (int(bndbox[2]), int(bndbox[3])), COLORS[1], 2)
            cv2.putText(image, '{label}'.format(label=name), (int(bndbox[0]), int(bndbox[1])), FONT, 1, COLORS[1], 2)
        # print(os.path.join(res,name,img_name))
        cv2.imwrite(os.path.join(res,name,img_name), image)

if __name__ == '__main__':
    # load net
    img_dim = (300,512)[args.size=='512']
    num_classes = 150
    net = build_net('test', img_dim, num_classes)    # initialize detector
    state_dict = torch.load(args.trained_model)
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
    net.eval()
    print('Finished loading model!')
    print(net)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()
    # evaluation
    #top_k = (300, 200)[args.dataset == 'COCO']
    top_k = 200
    detector = Detect(num_classes,0,cfg)
    save_folder = os.path.join(args.save_folder,args.dataset)
    rgb_means = ((104, 117, 123),(103.94,116.78,123.68))[args.version == 'RFB_mobile']
    basetransform = preproc(512,rgb_means,-2)
    test_net(num_classes, save_folder, net, detector, args.cuda,
             basetransform,
             top_k, thresh=0.01)
