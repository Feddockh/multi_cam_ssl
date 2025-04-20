import cv2
import numpy as np
import torch
from torchvision.ops import box_iou
from mmengine.structures import BaseDataElement, PixelData
from typing import List, Tuple, Optional, Dict


def remap_trident_output(elem: BaseDataElement, class_mapping: List[int],
                         num_classes: int) -> BaseDataElement:
    """
    Remaps the class description channels to the final class indices.

    Args:
        elem (BaseDataElement): The input element containing segmentation logits.
        class_mapping (List[int]): A list mapping each class description to a class index.
        num_classes (int): The total number of classes.

    Returns:
        BaseDataElement: The updated element with remapped segmentation logits and predictions.
    """
    seg_logits = elem.seg_logits.data  # [D, H, W]
    device = seg_logits.device
    dtype  = seg_logits.dtype
    _, H, W   = seg_logits.shape

    # Build new logits tensor (C, H, W)
    new_logits = torch.empty((num_classes, H, W), dtype=dtype, device=device)
    new_logits.fill_(-float("inf")) # so max() works

    # scatter‑reduce: max over all description channels mapping to same class
    for desc_idx, cls_idx in enumerate(class_mapping):
        new_logits[cls_idx] = torch.maximum(new_logits[cls_idx], seg_logits[desc_idx])

    # Compute new hard prediction mask
    new_pred = new_logits.argmax(dim=0, keepdim=True) # (1,H,W)

    # Write back into the BaseDataElement
    elem.seg_logits = PixelData(data=new_logits)
    elem.pred_sem_seg = PixelData(data=new_pred)

    return elem

def binary_mask_to_boxes(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert a single H×W binary mask into bounding boxes by finding connected 
    components via OpenCV contours.

    Args:
        mask (torch.Tensor): Binary mask of shape [H, W].

    Returns:
        torch.Tensor: Tensor of shape [N, 4] with boxes in (x1, y1, x2, y2) format.
    """
    # Move the mask to CPU and convert to uint8
    np_mask = (mask.cpu().numpy().astype(np.uint8))

    # Find external contours using OpenCV
    contours, _ = cv2.findContours(np_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    
    # Create the list of boxes
    boxes: List[List[float]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([x, y, x + w - 1, y + h - 1]) # Convert to [x1, y1, x2, y2]

    # Convert back to tensor and return
    # If there are no boxes, return an empty tensor
    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32, device=mask.device)
    return torch.tensor(boxes, dtype=torch.float32, device=mask.device)

def filter_boxes(boxes: torch.Tensor, min_area: float) -> torch.Tensor:
    """
    Filter out boxes with area less than min_area.

    Args:
        boxes (torch.Tensor[N,4]): Input boxes as (x1,y1,x2,y2)
        min_area (float): Minimum area threshold for boxes.

    Returns:
        torch.Tensor: Filtered boxes of shape [M, 4] in (x1, y1, x2, y2) format.
    """
    if boxes.numel() == 0:
        return boxes
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return boxes[areas >= min_area]

def merge_close_boxes(boxes: torch.Tensor, iou_thresh: float, dist_thresh: float) -> torch.Tensor:
    """
    Merge boxes whose IoU > iou_thresh OR whose centers are within dist_thresh.

    Args:
        boxes (torch.Tensor[N,4]): Input boxes as (x1,y1,x2,y2)
        iou_thresh (float): If two boxes overlap this much, merge them
        dist_thresh(float): If two box centers are closer than this, merge them

    Returns:
        torch.Tensor: Merged boxes of shape [M, 4] in (x1, y1, x2, y2) format.
    """
    if boxes.numel() == 0:
        return boxes
    N = boxes.size(0)

    # Compute pairwise IoU and get [N, N] IoU matrix
    ious = box_iou(boxes, boxes)

    # Compute pairwise center distances
    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5  # shape [N, 2]
    dists = torch.cdist(centers, centers)     # shape [N, N]

    # Create an adjacency matrix which is true if IoU > iou_thresh or center distance < dist_thresh
    # See reference: https://www.geeksforgeeks.org/adjacency-matrix/
    adj = (ious > iou_thresh) | (dists < dist_thresh)
    adj.fill_diagonal_(False) # Set diagonal false to avoid self-loops

    # Find connected components via DFS
    # See reference: https://www.geeksforgeeks.org/implementation-of-dfs-using-adjacency-matrix/
    visited = set()
    clusters: List[List[int]] = []
    for i in range(N):
        # Skip already visited nodes (already in a cluster)
        if i in visited:
            continue

        stack = [i]
        comp = []
        while stack:
            # Pop a node from the stack and add it to the component (if not visited)
            j = stack.pop()
            if j in visited:
                continue
            visited.add(j)
            comp.append(j)

            # Find all neighbors of the current node and add them to the stack
            # Reference: https://pytorch.org/docs/stable/generated/torch.nonzero.html
            neigh = adj[j].nonzero(as_tuple=False).squeeze(1).tolist()
            stack.extend(neigh)
        clusters.append(comp)

    # Merge each cluster into its own box
    merged = []
    for comp in clusters:
        subset = boxes[comp]
        x1 = subset[:, 0].min()
        y1 = subset[:, 1].min()
        x2 = subset[:, 2].max()
        y2 = subset[:, 3].max()
        merged.append(torch.stack([x1, y1, x2, y2]))
    return torch.stack(merged, dim=0)

def trident_to_target(elem: BaseDataElement, bg_idx: int = 0, min_area: float = 0.0, 
                      iou_thresh: float = 0.1, dist_thresh: float = 20.0, 
                      out_device = "cuda") -> Dict[str, torch.Tensor]:
    """
    Extract bboxes and labels from the segmentation masks in the input element.
    This function assumes that the element segmentation mask is a single channel
    where each pixel value corresponds to a class index. We will return the 
    boounding boxes and labels in the target (bboxes, labels) format.

    Args
        elem (BaseDataElement): The input element containing segmentation mask.
        bg_idx (int): The index of the background class.
        min_area (float): Minimum area threshold for boxes.
        iou_thresh (float): IoU threshold for merging boxes.
        dist_thresh (float): Distance threshold for merging boxes.

    Returns
        Target: A dictionary containing:
            - boxes: Tensor of shape [N, 4] with bounding boxes in (x1, y1, x2, y2) format.
            - labels: Tensor of shape [N] with class labels.
    """
    if out_device == "cuda":
        out_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        out_device = torch.device("cpu")

    # Grab the segmentation mask
    seg = elem.pred_sem_seg.data
    
    # Grab the segmentation logits and perform a softmax to get probabilities
    seg_logits = elem.seg_logits.data # Shape [C, H, W]
    probs = torch.softmax(seg_logits, dim=0) # Shape [C, H, W]

    # Drop the channel dimension if it exists
    if seg.dim() == 3:
        seg = seg[0]

    # Box each class present in the mask (excluding background)
    boxes_list, labels_list, scores_list = [], [], []
    for class_id in torch.unique(seg):
        # Skip the background class
        class_id = class_id.item()
        if class_id == bg_idx:
            continue

        # Make a binary mask for this class
        binary_mask = (seg == class_id).to(torch.float32)  # [H, W]

        # Convert the binary mask to bounding boxes
        boxes = binary_mask_to_boxes(binary_mask) # shape [N, 4]

        # Filter out boxes with little area
        boxes = filter_boxes(boxes, min_area)

        # Merge boxes that are close together
        boxes = merge_close_boxes(boxes, iou_thresh, dist_thresh)

        # Compute the scores for each box and append the label
        for box in boxes:

            # Get the indexable coordinates for the box
            x1, y1, x2, y2 = tuple(box.long().tolist())

            # Get the bounds of the segmentation space
            x_max = probs.shape[2]
            y_max = probs.shape[1]

            # Ensure the box is within the bounds of the segmentation space
            x1 = min(max(x1, 0), x_max - 1)
            y1 = min(max(y1, 0), y_max - 1)
            x2 = min(max(x2, 0), x_max - 1)
            y2 = min(max(y2, 0), y_max - 1)

            # Extract the patch of pixel probabilities corresponding to the box for this class
            patch = probs[class_id, y1:(y2 + 1), x1:(x2 + 1)] # [H, W]

            # Compute the score as the mean of the probabilites in the patch
            score = patch.mean().item()

            # Append the box, label, and score to the lists
            boxes_list.append(box)
            labels_list.append(class_id)
            scores_list.append(score)

    # If there are no detections, return empty tensors
    if not boxes_list:
        boxes = torch.empty((0, 4), device=out_device)
        labels = torch.empty((0,), dtype=torch.long, device=out_device)
        scores = torch.empty((0,), dtype=torch.float32, device=out_device)

    # If there are detections, stack the boxes and labels
    else:
        boxes = torch.stack(boxes_list, dim=0).to(out_device)
        labels = torch.tensor(labels_list, dtype=torch.long, device=out_device)
        scores = torch.tensor(scores_list, dtype=torch.float32, device=out_device)

    # Create the target dictionary
    target = {
        "boxes": boxes,
        "labels": labels,
        "scores": scores
    }
    return target