import os
import sys

import torch.utils.data
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from .coco import build as build_coco
from .coco import build_firefly

from utils import Camera
from loader import MultiCamDataset, SetType, multicam_collate_fn


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

def build_multicam_dataloader(multicam_dataset, batch_size=4):
    return DataLoader(multicam_dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      collate_fn=multicam_collate_fn)

def build_multicam(type, base_dir):
    BASE_DIR = os.path.dirname(base_dir)
    # DATA_DIR = os.path.join(BASE_DIR, "rivendale_dataset")
    DATA_DIR = os.path.join(BASE_DIR, "erwiam_dataset")
    cam0 = Camera("cam0") # Use this for the erwiam dataset
    cameras = [cam0]

    # Define the transforms
    transforms = v2.Compose([
        v2.Resize((1024, 1024), antialias=True), # Higher for finer details
        v2.RandomHorizontalFlip(p=0.5),
    ])

    # Create the dataset
    dataset = MultiCamDataset(DATA_DIR, cameras, set_type=type, transforms=transforms)
    return dataset
