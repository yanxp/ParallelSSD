import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import pickle
#from voc_eval import voc_eval
from .voc_eval import voc_eval
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import csv
from tqdm import tqdm

fs = open('data/logo_name.txt','r') 
LOGO_CLASSES = [ eval(name) for name in fs.readline().strip().split(',')]
fs.close()
# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

class AnnotationTransform(object):

    """Transforms a LOGO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of LOGO's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(LOGO_CLASSES, range(len(LOGO_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0,5)) 
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.strip()
            
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                bndbox.append(cur_pt)
            
            if int(bndbox[2])-int(bndbox[0])<30 and int(bndbox[3])-int(bndbox[1])<30:
                continue
            """
            if name.endswith('text'):
                name = name.split('_')[0]
            """
            if name not in LOGO_CLASSES:
                continue
         
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res,bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class LogoDetection(data.Dataset):

    """LOGO Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to LOGOdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'LOGOOPEN')
    """

    def __init__(self, root, image_sets, preproc=None, target_transform=AnnotationTransform(),
                 dataset_name='Logo'):
        self.root = root
        self.image_set = image_sets
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for name in image_sets:
            #self._year = year
            rootpath = os.path.join(self.root)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        sample = {}
        img_id = self.ids[index]
        imgname = self._imgpath % img_id
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        #
        # try:
        #     img = Image.open(imgname)
        #     img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        # except:
        #     print('imgname:{}'.format(imgname))
        
        if self.target_transform is not None:
            target = self.target_transform(target)


        if self.preproc is not None:
            img, target = self.preproc(img, target)
       
        if np.all(target == 0):
            print(img_id[1])
            import pdb;pdb.set_trace()

        if self.image_set == ['test']:
            return img, target
        else:
            sample['image'] = img
            sample['target'] = np.fromstring(pickle.dumps(target), dtype=np.uint8).astype(np.float32)
   
            return sample
        #return  target,img_id[1]

    def __len__(self):
        return len(self.ids)

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        # gt = self.target_transform(anno, 1, 1)
        # gt = self.target_transform(anno)
        # return img_id[1], gt
        if self.target_transform is not None:
            anno = self.target_transform(anno)
        return anno
        

    def pull_img_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno)
        height, width, _ = img.shape
        boxes = gt[:,:-1]
        labels = gt[:,-1]
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height
        labels = np.expand_dims(labels,1)
        targets = np.hstack((boxes,labels))
        
        return img, targets

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        to_tensor = transforms.ToTensor()
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes)
        aps,map = self._do_python_eval(output_dir)
        return aps,map

    def _get_voc_results_file_template(self):
        filename = 'comp4_det_test' + '_{:s}.txt'
        filedir = os.path.join(
            self.root, 'results', 'LOGO','Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(LOGO_CLASSES):
            cls_ind = cls_ind 
            if cls == '__background__':
                continue
            print('Writing {} LOGO results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.ids):
                    index = index[1]
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                dets[k, 0] + 1, dets[k, 1] + 1,
                                dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        rootpath = os.path.join(self.root)
        name = self.image_set[0]
        annopath = os.path.join(
                                rootpath,
                                'Annotations',
                                '{:s}.xml')
        imagesetfile = os.path.join(
                                rootpath,
                                'ImageSets',
                                'Main',
                                name+'.txt')
        cachedir = os.path.join(self.root, 'annotations_cache')
        aps = []
        # The PASCAL LOGO metric changed in 2010
        use_07_metric = True 
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        import datetime,time
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        csvfile = open('csv/'+str(now)+'.csv','w')
        writer = csv.writer(csvfile)
        for i, cls in enumerate(LOGO_CLASSES):

            if cls == '__background__':
                continue
            
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                                    filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                                    use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            writer.writerow([cls,ap])
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        csvfile.close()
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')
        return aps,np.mean(aps)

    def show(self, index):
        img, target = self.__getitem__(index)
        for obj in target:
            obj = obj.astype(np.int)
            cv2.rectangle(img, (obj[0], obj[1]), (obj[2], obj[3]), (255,0,0), 3)
        cv2.imwrite('./image.jpg', img)




## test
if __name__ == '__main__':
    ds = LogoDetection('data/aliyun', ['train'],None, AnnotationTransform())
    fs = open('train.txt','w')
    for i in tqdm(range(len(ds))):
        target,img_id = ds[i]
        if np.all(target == 0):
            print(target,img_id)
        fs.write(img_id+'\n')
    fs.close()

    #ds.show(100)
