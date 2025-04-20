import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
from visual import plot


class CocoDetDataset(CocoDetection):
    def __init__(self, root: str, annFile: str, transform=None, include_paths=False):
        super().__init__(root=root, annFile=annFile)
        self.transform = transform
        self.include_paths = include_paths

    def __getitem__(self, idx):
        img, targets = super().__getitem__(idx)

        # Perform the transformations if any
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        boxes = []
        labels = []
        for obj in targets:
            # Convert COCO boxes [x, y, w, h] --> [x1, y1, x2, y2]
            x, y, w, h = obj["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(obj["category_id"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        if self.include_paths:
            img_id = self.ids[idx]
            info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.root, info["file_name"])
            return img_tensor, target, img_path
        else:
            return img_tensor, target
    
def collate_fn(batch):
    out = list(zip(*batch))
    if len(out) == 2:
        images, targets = out
        images = torch.stack(images, 0)
        return images, targets
    elif len(out) == 3:
        images, targets, paths = out
        images = torch.stack(images, 0)
        return images, targets, paths

def demo():
    # Dummy values for image folder and annotation file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "erwiam_dataset", "cam0")
    IMG_FOLDER = os.path.join(DATA_DIR, "images")
    ANN_FILE = os.path.join(DATA_DIR, "val.json")

    # Define a simple transform: resize then convert to tensor
    tensor_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = CocoDetDataset(IMG_FOLDER, ANN_FILE, transform=tensor_transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    img, target = dataset[110]
    plot([(img, target)])
    
    # Display a batch from the loader
    for images, targets in loader:
        print("Images shape:", images.shape)
        print("First target keys:", list(targets[0].keys()))
        break

if __name__ == '__main__':
    demo()



