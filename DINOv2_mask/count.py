from pycocotools.coco import COCO

ann_path = r"image_data\firefly_left\annotations.json"
coco = COCO(ann_path)

image_ids = coco.getImgIds()

num_with_annotations = 0
num_without_annotations = 0

for img_id in image_ids:
    ann_ids = coco.getAnnIds(imgIds=img_id)
    if len(ann_ids) > 0:
        num_with_annotations += 1
    else:
        num_without_annotations += 1

print(f"with annotations: {num_with_annotations}")
print(f"without annotations: {num_without_annotations}")
print(f"total: {len(image_ids)}")
