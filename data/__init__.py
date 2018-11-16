# from .voc import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .logo import LogoDetection, AnnotationTransform, LOGO_CLASSES, collate_minibatch
from .coco import COCODetection
from .data_augment import *
from .config import *
