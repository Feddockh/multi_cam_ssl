from collections import OrderedDict
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from jutils.nn_utils import load_state_dict_up_to_classif_head


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    num_classes = 4 #TODO how many classes is it actually
    detr, criterion, _ = get_model(num_classes) #TODO
    load_state_dict_up_to_classif_head(detr, args) 
    detr_lora = loraify(detr)
    detr_lora.to("cuda")
    optimizer = torch.optim.Adam(detr.parameters(), lr, etc)
    trainset = build_dataset("train")
    valset = build_dataset("val")
    dataloader = DataLoader(trainset, args.batch_size, collate_fn=utils.collate_fn) 
    logger = get_writer()

    for epoch in tqdm(args.max_epoch):
        for imgs, targets in trainset: 
            # in the off chance that inference doesn't work, break point the eval 
            # function in detr and see how inference gets done
            # how is inference done in main.py?
            pred = detr_lora(imgs) # yeah this is how it's done
            loss_dict = criterion(imgs, targets) # it returns a loss_dict, which they
            # sum over.
            # TODO: ponder: can I use the existing training machinery with our dataset?
            # examine.
            loss = sum(loss_dict
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.log(loss, "loss", it) # TODO
            logger.log(gradients_of(detr_lora), "gradients", it) # TODO
