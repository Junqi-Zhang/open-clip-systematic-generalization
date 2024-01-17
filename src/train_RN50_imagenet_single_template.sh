# Description: Training script for RN50 with single template on ImageNet
# Usage: bash train_RN50_imagenet_single_template.sh
OMP_NUM_THREADS=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node 8 -m training.main \
    --dataset-type folder \
    --multi-images-per-text \
    --normalize-labels \
    --train-data '../data/ImageNet/train' \
    --imagenet-val '../data/ImageNet/val' \
    --imagenet-zero-shot '../data/ImageNet/zero_shot' \
    --text-type imagenet_single_template \
    --model RN50 \
    --image-constant-key imagenet \
    --aug-cfg use_timm=True hflip=0.5 \
    --precision amp_bf16 \
    --workers 16 \
    --zeroshot-frequency 1 \
    --save-frequency 32 \
    --epochs 128 \
    --batch-size 256 \
    --warmup 0 \
    --log-every-n-steps 128 \
    --report-to tensorboard
