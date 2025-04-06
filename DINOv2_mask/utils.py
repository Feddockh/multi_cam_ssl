import os
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms import functional as F
import torch.nn.functional as Fnn
import torchvision.ops as ops
from configs import VIS_DIR

def collate_fn(batch):
    images = []
    targets = []
    for sample in batch:
        cam_name = list(sample.keys())[0] 
        img, tgt = sample[cam_name]
        images.append(img)
        targets.append(tgt)
    return images, targets


def save_visualizations(images, outputs, targets, epoch):
    """
    Save both predictions and ground truths for visualization
    """
    os.makedirs(VIS_DIR, exist_ok=True)

    images = [img.detach().cpu() for img in images]
    pred_logits = outputs['pred_logits']
    pred_boxes = outputs['pred_boxes']
    pred_masks = outputs['pred_masks']

    scores = pred_logits.softmax(-1)[..., :-1].max(-1)[0]
    topk = scores.topk(5, dim=1)
    topk_idx = topk.indices

    for idx, (img, boxes, masks, logits, top_idx, tgt) in enumerate(zip(images, pred_boxes, pred_masks, pred_logits, topk_idx, targets)):
        img = (img * 255).clamp(0, 255).to(torch.uint8)
        img_pil = F.to_pil_image(img)

        W, H = img_pil.size

        ### Prediction Visualization
        pred_img_tensor = F.pil_to_tensor(img_pil)

        selected_boxes = boxes[top_idx]
        selected_boxes = ops.box_convert(selected_boxes, in_fmt='cxcywh', out_fmt='xyxy')
        selected_boxes = selected_boxes * torch.tensor([W, H, W, H], device=selected_boxes.device)

        selected_logits = logits[top_idx]
        selected_labels = selected_logits.argmax(-1)

        pred_img_tensor = draw_bounding_boxes(
            pred_img_tensor,
            selected_boxes,
            labels=[str(label.item()) for label in selected_labels],
            colors="yellow",
            width=3
        )

        selected_masks = masks[top_idx]
        masks_pred = selected_masks.sigmoid()
        masks_pred = Fnn.interpolate(
            masks_pred.unsqueeze(0).float(),
            size=pred_img_tensor.shape[1:], 
            mode="bilinear", 
            align_corners=False
        ).squeeze(0) > 0.5

        if masks_pred.shape[0] > 0:
            masks_pred = masks_pred.to(pred_img_tensor.device)
            pred_img_tensor = draw_segmentation_masks(pred_img_tensor, masks=masks_pred, alpha=0.4, colors="red")

        save_path_pred = os.path.join(VIS_DIR, f"epoch{epoch}_pred_img{idx}.png")
        F.to_pil_image(pred_img_tensor).save(save_path_pred)

        ### Ground Truth Visualization
        gt_img_tensor = F.pil_to_tensor(img_pil)

        gt_boxes = tgt["boxes"]
        gt_labels = tgt["labels"]
        gt_masks = tgt["masks"]
        
        # Scale GT to Pre  
        scale_x = 672 / 1440
        scale_y = 672 / 1080
        gt_boxes = gt_boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y], device=gt_boxes.device)
        
        gt_img_tensor = Fnn.interpolate(gt_img_tensor.unsqueeze(0).float(), size=(672, 672), mode="bilinear", align_corners=False).squeeze(0).to(torch.uint8)

        if gt_masks.shape[0] > 0:
            gt_masks = Fnn.interpolate(gt_masks.unsqueeze(1).float(), size=(672, 672), mode="nearest").squeeze(1).bool()

        gt_img_tensor = draw_bounding_boxes(
            gt_img_tensor,
            gt_boxes,
            labels=[str(label.item()) for label in gt_labels],
            colors="green",
            width=3
        )

        if gt_masks.shape[0] > 0:
            gt_masks = gt_masks.bool()
            gt_img_tensor = draw_segmentation_masks(gt_img_tensor, masks=gt_masks, alpha=0.4, colors="blue")

        save_path_gt = os.path.join(VIS_DIR, f"epoch{epoch}_gt_img{idx}.png")
        F.to_pil_image(gt_img_tensor).save(save_path_gt)