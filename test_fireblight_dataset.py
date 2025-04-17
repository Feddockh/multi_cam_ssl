from datasets import build_dataset

def get_args():
    args = ArgumentParser()
    return args

args = get_args() # and the args.dataset_file will be 'fire_blight'
dataset = build_dataset(image_set, args)
dataloader = Dataloader(dataset, batches, etc)

for batch in dataloader:
    print(batch)
