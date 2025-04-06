import torch
import torch.nn.functional as F

def loss_fn(outputs, targets):
    
    valid_targets = []
    skipped_indices = []

    for idx, v in enumerate(targets):
        if v["boxes"].numel() > 0:
            valid_targets.append(v)
        else:
            skipped_indices.append(idx)

    if len(valid_targets) == 0:
        raise ValueError("No valid targets in this batch.")

    targets = valid_targets

    """
    outputs: dict
        - pred_logits: (B, num_queries, num_classes+1)
        - pred_boxes: (B, num_queries, 4)
        - pred_masks: (B, num_queries, H, W)
    targets: list of dict
        - each dict with keys: boxes (Tensor[N, 4]), labels (Tensor[N]), masks (Tensor[N, H, W])
    """

    pred_logits = outputs['pred_logits'] # (B, Q, C+1)
    pred_boxes = outputs['pred_boxes'] # (B, Q, 4)
    pred_masks = outputs['pred_masks'] # (B, Q, H, W)

    batch_size, num_queries, _ = pred_logits.shape

    # Flatten predictions
    pred_logits = pred_logits.view(-1, pred_logits.size(-1)) # (B*Q, C+1)
    pred_boxes = pred_boxes.view(-1, 4) # (B*Q, 4)
    pred_masks = pred_masks.view(-1, pred_masks.shape[-2], pred_masks.shape[-1]) # (B*Q, H, W)

    # Flatten targets
    target_labels = torch.cat([v["labels"] for v in targets], dim=0) # (sum_targets,)
    target_boxes = torch.cat([v["boxes"] for v in targets], dim=0) # (sum_targets, 4)
    target_masks = torch.cat([v["masks"] for v in targets], dim=0) # (sum_targets, H, W)

    num_targets = target_labels.shape[0]

    # Select top-k predictions (for now, just first num_targets, simple matching)
    pred_logits = pred_logits[:num_targets]
    pred_boxes = pred_boxes[:num_targets]
    pred_masks = pred_masks[:num_targets]
    
    # --- Normalize target boxes ---
    img_h, img_w = target_masks.shape[-2:]
    target_boxes = target_boxes.clone()
    target_boxes[:, 0::2] /= img_w  # normalize x
    target_boxes[:, 1::2] /= img_h  # normalize y

    # print("pred_boxes range:", pred_boxes.min().item(), pred_boxes.max().item())
    # print("target_boxes range:", target_boxes.min().item(), target_boxes.max().item())

    # Classification loss
    cls_loss = F.cross_entropy(pred_logits, target_labels, reduction='mean')

    # BBox regression loss ---
    bbox_l1_loss = F.l1_loss(pred_boxes, target_boxes, reduction='mean')

    # Mask BCE loss
    # BCE loss for per-pixel binary prediction
    # Mask BCE loss
    pred_masks = pred_masks.sigmoid()
    pred_masks = pred_masks.unsqueeze(1)  # (B*Q, 1, H, W)
    pred_masks = F.interpolate(pred_masks, size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
    pred_masks = pred_masks.squeeze(1)  # (B*Q, H_target, W_target)
    
    bce_loss = F.binary_cross_entropy(pred_masks, target_masks.float(), reduction='mean')
    
    smooth = 1e-6
    pred_masks_flat = pred_masks.view(pred_masks.shape[0], -1)
    target_masks_flat = target_masks.float().view(target_masks.shape[0], -1)
    intersection = (pred_masks_flat * target_masks_flat).sum(dim=1)
    union = pred_masks_flat.sum(dim=1) + target_masks_flat.sum(dim=1)
    dice_loss = 1 - (2. * intersection + smooth) / (union + smooth)
    dice_loss = dice_loss.mean()
    mask_loss = 0.5 * bce_loss + 0.5 * dice_loss.mean()

    intersection = (pred_masks_flat * target_masks_flat).sum(dim=1)
    union = pred_masks_flat.sum(dim=1) + target_masks_flat.sum(dim=1)

    dice_loss = 1 - (2. * intersection + smooth) / (union + smooth)


    # Final loss
    total_loss = cls_loss + bbox_l1_loss + mask_loss

    loss_dict = {
        "loss_cls": cls_loss,
        "loss_bbox": bbox_l1_loss,
        "loss_mask": mask_loss,
        "total_loss": total_loss
    }

    return loss_dict
