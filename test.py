from argparse import ArgumentParser
import torch
from fbdetr.datasets import build_dataset, build_multicam, build_multicam_dataloader
import fbdetr.util.misc as utils
from fbdetr.models import build_model
from fbdetr.main import get_args_parser
from jutils.jnn import LoRA
from jutils.nn_utils import loraify
from torch.utils.data import DataLoader
#TODO: change the build function, because you need to change the number of output classes.
#TODO: moreover, start looking into low rank approximation.
#TODO: hmm. Starting to think that I should just copy and paste the  argparse in main.py.

def get_weights(args):
    # args.frozen weights is the path to the checkpoint
    checkpoint = torch.load(args.frozen_weights, map_location='cpu')
    return checkpoint['model']


def test_detr_loraify():
    args = get_args_parser()
    dataset = build_dataset(args.image_set, args)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=utils.collate_fn)
    detr, *_ = build_model(args)
    weights = get_weights(args)
    detr.load_state_dict(weights)
    detr_lora = loraify(detr)
    for b in dataloader:
        pred = detr_lora(b)
        print(pred)
