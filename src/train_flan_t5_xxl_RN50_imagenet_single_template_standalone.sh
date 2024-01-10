# Description: Training script for FLAN-T5-XXL-RN50 with single template on ImageNet (standalone)
# Usage: bash train_flan_t5_xxl_RN50_imagenet_single_template_standalone.sh
OMP_NUM_THREADS=2 \
CUDA_VISIBLE_DEVICES=0 \
python -m training.main \
    --dataset-type folder \
    --multi-images-per-text \
    --normalize-labels \
    --train-data '../data/ImageNet/train' \
    --imagenet-val '../data/ImageNet/val' \
    --imagenet-single-template '../data/ImageNet/zero_shot' \
    --imagenet-overall-prompt '../data/ImageNet/zero_shot' \
    --text-type imagenet_single_template \
    --text-embeds-path '../data/ImageNet/text_embeds/imagenet_single_template_flan_t5_xxl.pt' \
    --model flan-t5-xxl-RN50 \
    --image-constant-key imagenet \
    --aug-cfg use_timm=True hflip=0.5 \
    --precision amp_bf16 \
    --workers 4 \
    --zeroshot-frequency 4 \
    --save-frequency 16 \
    --epochs 256 \
    --batch-size 128 \
    --warmup 0 \
    --log-every-n-steps 128