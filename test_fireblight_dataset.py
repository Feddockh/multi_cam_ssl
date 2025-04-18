import sys
import os
sys.path.append(os.path.join(os.getcwd(), "fbdetr"))
from argparse import ArgumentParser
from fbdetr.datasets import build_dataset, build_multicam, build_multicam_dataloader
import fbdetr.util.misc as utils
from torch.utils.data import DataLoader

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_file",
                        type=str,
                        default="fire_blight",
                        choices=["coco", "coco_panoptic", "fire_blight"])
    parser.add_argument("--coco_path",
                        type=str,
                        default="erwiam_dataset")
    parser.add_argument("--masks",
                        action="store_true")
    parser.add_argument("--image_set",
                        type=str,
                        default="train")
    args = parser.parse_args()
    return args
def test1():
    args = get_args() # and the args.dataset_file will be 'fire_blight'
    dataset = build_dataset(args.image_set, args)
    for d in dataset:
        print(d)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=utils.collate_fn)
    # hell yah I get here
    for batch in dataloader:
        print(batch)

def test2():
    ds = build_multicam("train", ".")
    dl = build_multicam_dataloader(ds)
    for b in dl:
        print(b)

def main():
    test1()

if __name__ == "__main__":
    main()
