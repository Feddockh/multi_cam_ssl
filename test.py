import sys
from jutils.utils import add_module_to_path, pdb
from jutils.nn_utils import loraify
add_module_to_path("fbdetr")
import torch
from fbdetr.datasets import build_dataset
import fbdetr.util.misc as utils
from fbdetr.models import build_model
from fbdetr.main import get_args_parser
from torch.utils.data import DataLoader

def get_weights(args):
    # args.frozen weights is the path to the checkpoint
    checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu')
    return checkpoint['model']


def test_detr_loraify():
    print(sys.path)
    parser = get_args_parser()
    args = parser.parse_args()
    args.batch_size = 2
    args.no_aux_loss = True
    args.eval = True
    args.resume = "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"
    args.coco_path = "erwiam_dataset"
    args.image_set = "train"
    args.dataset_file = "fire_blight"
    args.masks = False
    args.device = "cuda"
    ### args stuff mostly done
    dataset = build_dataset(args.image_set, args)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=utils.collate_fn)
    args.dataset_file = "coco"
    detr, *_ = build_model(args)
    weights = get_weights(args)
    detr.load_state_dict(weights)
    detr_lora = loraify(detr)
    for b in dataloader:
        pdb()
        pred = detr_lora(b)
        print(pred)

def main():
    test_detr_loraify()

if __name__ == "__main__":
    main()
