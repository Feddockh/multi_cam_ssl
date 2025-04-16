import Trainer
import detr
import dataset

def main(args=None):
    train, val = dataset.split("train", "val")
    model = detr()
    trainer = Trainer(args, model, train, val)
    trainer.train()

