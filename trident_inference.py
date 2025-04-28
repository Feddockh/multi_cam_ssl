import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection.iou import IntersectionOverUnion
from mmengine.structures import BaseDataElement
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Update Python path to include the Trident directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "Trident"))

from Trident.trident import Trident
from coco_det_dataset import CocoDetDataset, collate_fn
from trident_helpers import remap_trident_output, trident_to_target, \
                                filter_boxes, merge_close_boxes
from visual import plot, plot_trident, plot_pr_curves


# Skip all and just load / plot the metrics
LOAD_METRICS = False

# Visualize settings
VIS = True
VIS_IMAGES = ["20220622_CanonEOS90D_1019_KGH.JPG",
              "20220622_CanonEOS90D_1035_KGH.JPG",
              "20220622_CanonEOS90D_0566_KGH.JPG"]
SEED = 42

# Path settings
DATA_DIR = os.path.join(BASE_DIR, "erwiam_dataset", "cam0")
TRIDENT_DIR = os.path.join(BASE_DIR, "Trident")

# SAM / Trident settings 
SAM_CHECKPOINT = os.path.join(TRIDENT_DIR, "segment_anything", "sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COARSE_THRESH = 0.2

# Class / description map
# NOTE: background must remain at index 0
CLASS_LIST = [
    "bg",
    "fire blight"
]
DATASET_CLASS_LIST = [
    "bg",
    "flower",
    "shoot",
    "maybe",
    "leaf"
]
# DESCRIPTION_TO_CLASS_IDX = {
#     "healthy buds": 0,
#     "shriveled buds": 1,
#     "healthy shoots": 0,
#     "shriveled shoots": 1,
#     "healthy branches": 0,
#     "shriveled branches": 1,
#     "healthy leaves": 0,
#     "shriveled leaves": 1,
#     "grass": 0,
#     "sky": 0,
#     "ground": 0,
#     "tree": 0,
# }
DESCRIPTION_TO_CLASS_IDX = {
    "healthy buds": 0,
    "shriveled buds": 1,
    "healthy shoots": 0,
    "shriveled shoots": 1,
    "healthy branches": 0,
    "shriveled branches": 1,
    "healthy leaves": 0,
    "shriveled leaves": 1,
    "healthy fruit": 0,
    "infected fruit": 1,
    "healthy flowers": 0,
    "infected flowers": 1,
    "leaves on the ground": 0,
    "grass": 0,
    "sky": 0,
    "ground": 0,
    "tree bark": 0,
    "water": 0,
}
CLASS_MAPPING = list(DESCRIPTION_TO_CLASS_IDX.values())
NUM_CLASSES = len(CLASS_LIST)

# Save the class descriptions
with open(os.path.join(BASE_DIR, "class_descriptions.txt"), "w") as f:
    for class_desc in DESCRIPTION_TO_CLASS_IDX.keys():
        f.write(f"{class_desc}\n")

# Model creation
print("Loading Trident model ...")
model = Trident(
    clip_type = "openai",
    model_type = "ViT-B/16",
    vfm_model = "dino",
    name_path = "class_descriptions.txt",
    sam_refinement = True,
    coarse_thresh = COARSE_THRESH,
    minimal_area = 225,
    debug = False,
    sam_ckpt = SAM_CHECKPOINT,
    sam_model_type = SAM_MODEL_TYPE,
).to(DEVICE)
model.eval()

# Define transforms
mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]
tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
inv_norm = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean, std)],
    std=[1/s for s in std]
)

# Construct the dataset
IMG_FOLDER = os.path.join(DATA_DIR, "images")
ANN_FILE = os.path.join(DATA_DIR, "val.json")
dataset = CocoDetDataset(IMG_FOLDER, ANN_FILE, transform=tensor_transform, include_paths=True)

# Create a DataLoader for the dataset
torch.manual_seed(SEED)
loader = DataLoader(
    dataset = dataset, 
    batch_size = 1, # Batch size of 1 for Trident inference
    shuffle = True, 
    num_workers = 0,
    collate_fn = collate_fn
)

# Initialize metrics
# Reference: https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html
map_metric = MeanAveragePrecision(
    box_format = "xyxy", 
    iou_type = "bbox", 
    iou_thresholds = None, # [0.50, 0.55, ..., 0.95]
    rec_thresholds = None, # [0.01, 0.02, ..., 1.0]
    max_detection_thresholds = None, # [1, 10, 100]
    class_metrics = True, # Enable per-class metrics for mAP and mAR_100
    extended_summary = True, # Enable summary with additional metrics including IOU, precision and recall
)

print("Starting evaluation ...")
with torch.no_grad():
    for i, (img, target, img_path) in enumerate(tqdm(loader, total=len(loader), desc="Inference", unit="img")):
        # Skip the images that are not in the VIS_IMAGES list
        if VIS and os.path.basename(img_path[0]) not in VIS_IMAGES:
            continue

        # Break loop if we are loading metrics
        if LOAD_METRICS:
            break

        # Move the image to the device
        img = img.to(DEVICE)

        # Compress the dataset class labels to the current class labels
        target[0]["labels"] = (target[0]["labels"] > 0).long()

        # Merge the target boxes like we do later in the predictions
        new_boxes_list = []
        new_labels_list = []
        for class_id in torch.unique(target[0]["labels"]):
            # Get the bounding boxes for the current class
            mask = target[0]["labels"] == class_id
            class_boxes = target[0]["boxes"][mask]

            # Filter the class boxes and merge them
            class_boxes = filter_boxes(class_boxes, min_area=50.0)
            class_boxes = merge_close_boxes(class_boxes, iou_thresh=0.1, dist_thresh=50.0)
            
            # Update the lists with the new boxes and labels
            new_boxes_list.append(class_boxes)
            new_labels_list.append(class_id.repeat(class_boxes.shape[0]))

        # Concatentate the new boxes and labels and store them in the target
        if len(new_boxes_list) == 0:
            new_boxes_list = [torch.empty((0, 4), dtype=torch.float32)]
            new_labels_list = [torch.empty((0,), dtype=torch.int64)]
        target[0]["boxes"] = torch.cat(new_boxes_list, dim=0)
        target[0]["labels"] = torch.cat(new_labels_list, dim=0)
        
        # Build data samples for Trident prediction function
        B, C, H, W = img.shape
        
        data_sample = BaseDataElement()
        data_sample.set_metainfo({
            "ori_shape":  (H, W),
            "img_shape":  (H, W),
            "pad_shape":  (H, W),
            "padding_size": [0, 0, 0, 0],
        })
        data_sample.img_path = img_path[0]

        # Use model for inference and remove batch dimension
        pred = model.predict(img, data_samples=[data_sample])[0]

        # Plot the trident output
        if VIS:
            vis_img = inv_norm(img[0])
            plot_trident(
                image = vis_img,
                seg_pred = pred.pred_sem_seg.data,
                seg_logit = pred.seg_logits.data,
                name_list = list(DESCRIPTION_TO_CLASS_IDX.keys()),
                save_path = os.path.join(BASE_DIR, "results", "trident_out", os.path.basename(img_path[0]))
            )

        # Remap the class indices to the original dataset
        pred = remap_trident_output(pred, CLASS_MAPPING, NUM_CLASSES)
            
        # Grab the boxes and labels from the segmentation mask
        pred_target = trident_to_target(
            elem = pred,
            bg_idx = 0,
            min_area = 50.0,
            iou_thresh = 0.1,
            dist_thresh = 50.0,
            out_device="cpu"
        )

        # Add the metric update
        map_metric.update(preds=[pred_target], target=target)
        
        # Debugging (display unnormalized image and box predictions)
        if VIS:
            # plot(
            #     imgs = [(vis_img, pred_target), (vis_img, target[0])], 
            #     class_names=CLASS_LIST,
            #     col_title=["Predictions", "Ground Truth"],
            #     save_path=os.path.join(BASE_DIR, "results", "bbox_predictions", os.path.basename(img_path[0]))
            # )
            plot(
                imgs = [(vis_img, pred_target)],
                save_path=os.path.join(BASE_DIR, "results", "bbox_predictions", os.path.basename(img_path[0]))
            )

            # We don't need to save all the images, just a few for visualization
            if i >= 20:
                break

if VIS:
    print("Visualization complete. Check the results folder for images.")
    exit()

# Compute and save the mean average precision results
print("Computing metrics ...")
if LOAD_METRICS:
    results = torch.load(os.path.join(BASE_DIR, "results", "trident_results.pth"))
else:
    results = map_metric.compute()
torch.save(results, os.path.join(BASE_DIR, "results", "trident_results.pth"))
print("Evaluation Metrics:")

print(f"mAP@[.50:.95]        = {results['map']:.4f}")
print(f"mAP@.50              = {results['map_50']:.4f}")
print(f"mAP@.75              = {results['map_75']:.4f}")
print(f"mAP@.50:.95 (small)  = {results['map_small']:.4f}")
print(f"mAP@.50:.95 (medium) = {results['map_medium']:.4f}")
print(f"mAP@.50:.95 (large)  = {results['map_large']:.4f}")
print(f"mAR@1                = {results['mar_1']:.4f}")
print(f"mAR@10               = {results['mar_10']:.4f}")
print(f"mAR@100              = {results['mar_100']:.4f}")
print(f"mAR@100 (small)      = {results['mar_small']:.4f}")
print(f"mAR@100 (medium)     = {results['mar_medium']:.4f}")
print(f"mAR@100 (large)      = {results['mar_large']:.4f}")

# Plot the precision-recall curves
plot_pr_curves(
    results = results,
    metric = map_metric,
    class_names = CLASS_LIST[1:],
    max_x = 0.25,
    max_y = 1.05,
    save_dir = os.path.join(BASE_DIR, "results")
)