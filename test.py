from argparse import ArgumentParser
from fbdetr.datasets import build_dataset, build_multicam, build_multicam_dataloader
import fbdetr.util.misc as utils
from fbdetr.models import build_model
from fbdetr.main import get_args_parser
#TODO: change the build function, because you need to change the number of output classes.
#TODO: moreover, start looking into low rank approximation.
#TODO: hmm. Starting to think that I should just copy and paste the  argparse in main.py.

def get_weights():
    # args.frozen weights is the path to the checkpoint
    checkpoint = torch.load(args.frozen_weights, map_location='cpu')
    return checkpoint['model']

def loraify(model, include=["attn", "linear"], r=4):
    pass

def test_detr_inference_on_fireblight():
    args = get_args_parser()
    dataset = build_dataset(args.image_set, args)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=utils.collate_fn)
    detr, *_ = build_model(args)
    weights = get_weights()
    detr.load_state_dict(weights)
    detr_headless = chop_off_head(detr)
    detr_lora = add_lora(detr_headless)
    for b in dataloader:
        pred = detr_lora(b)
