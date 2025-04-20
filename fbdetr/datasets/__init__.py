import os
import sys

import torch.utils.data
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from .coco import build as build_coco
from .coco import build_firefly



def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'fire_blight':
        return build_firefly(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
