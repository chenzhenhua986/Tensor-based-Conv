export DATASET_PATH=data/
CUDA_VISIBLE_DEVICES=3 python -u cityscapes_seg.py training configs/cityscapes_tconv.py
