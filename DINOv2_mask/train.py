import sys
import os
from torch.utils.data import DataLoader
import torchvision 
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*xFormers.*")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from loader import MultiCamDataset
from utils_camera import Camera
from model import DinoMaskDETR
from loss import loss_fn
from utils import collate_fn, save_visualizations
from configs import *
from torch.utils.data import DataLoader

def custom_collate_fn(batch):
    batch = [b for b in batch if b]  
    if len(batch) == 0:
        return None
    images, targets = collate_fn(batch)
    return images, targets

import matplotlib.pyplot as plt

def plot_losses(logs, save_dir="DINOv2_mask/loss"):
    """
    Plot loss curves after training
    logs: list of dict, each dict contains "loss_cls", "loss_bbox", "loss_mask", "total_loss"
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs = list(range(1, len(logs)+1))
    cls_losses = [log["loss_cls"] for log in logs]
    bbox_losses = [log["loss_bbox"] for log in logs]
    mask_losses = [log["loss_mask"] for log in logs]
    total_losses = [log["total_loss"] for log in logs]

    plt.figure()
    plt.plot(epochs, total_losses, label="Total Loss")
    plt.plot(epochs, cls_losses, label="Cls Loss")
    plt.plot(epochs, bbox_losses, label="BBox Loss")
    plt.plot(epochs, mask_losses, label="Mask Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curves.png"))
    plt.close()



def train():
    best_loss = float('inf')
    logs = []
    
    device = torch.device(DEVICE)

    # Dataset
    cam0 = Camera("firefly_left")
    dataset = MultiCamDataset(base_dir="image_data", cameras=[cam0])
    dataloader = DataLoader(
                            dataset,
                            batch_size=2,
                            shuffle=True,
                            collate_fn=custom_collate_fn, 
                            num_workers=4,
                            pin_memory=True
                        )

    # Model
    model = DinoMaskDETR()
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Training Loop
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            if batch is None:
                continue
            images, targets = batch
            images = [torchvision.io.read_image(img) if isinstance(img, str) else img for img in images]
            images = [img.to(device) for img in images]   # ✅ 再.to(device)
            targets = [{k: v.to(device) for k, v in tgt.items()} for tgt in targets]

            images = torch.stack(images)  # Stack into (B, 3, H, W)
            images = images.float() / 255.0
            # images = F.interpolate(images, size=(896, 896), mode="bilinear", align_corners=False)
            images = F.interpolate(images, size=(672, 672), mode="bilinear", align_corners=False)

            
            if isinstance(images, dict):
                images = list(images.values())[0]

            outputs = model(images)
            loss_dict = loss_fn(outputs, targets)
            loss = loss_dict["total_loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] | Loss: {avg_loss:.4f} | "
                f"Cls: {loss_dict['loss_cls']:.4f} | Bbox: {loss_dict['loss_bbox']:.4f} | Mask: {loss_dict['loss_mask']:.4f}")
        
        logs.append({
            "loss_cls": loss_dict["loss_cls"].item(),
            "loss_bbox": loss_dict["loss_bbox"].item(),
            "loss_mask": loss_dict["loss_mask"].item(),
            "total_loss": avg_loss,
        })


        # Visualization every VISUALIZE_FREQ epochs
        if epoch % VISUALIZE_FREQ == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                outputs = model(images)
                save_visualizations(images, outputs, targets, epoch)

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt_path = os.path.join(SAVE_DIR, "best.pth")
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"[Info] Saved new best model at epoch {epoch} with loss {best_loss:.4f}")
            
        if epoch % 10 == 0:
            plot_losses(logs)

        
if __name__ == "__main__":
    train()
