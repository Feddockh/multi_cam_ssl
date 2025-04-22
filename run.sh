MODE=$1
if [ "$MODE" == "train" ]; then
    python fbdetr/main.py --coco_path erwiam_dataset --dataset_file fire_blight --output_dir checkpoints --resume "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth" 
elif [ "$MODE" == "eval" ]; then
    python fbdetr/main.py --batch_size 2 --no_aux_loss --eval --resume "checkpoints/checkpoint0199.pth" --coco_path erwiam_dataset --dataset_file fire_blight
elif [ "$MODE" == "vis" ]; then
    python vis.py --coco_path erwiam_dataset --dataset_file fire_blight
fi
