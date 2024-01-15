# Description: Training script for FLAN-T5-XXL-RN50 with overall prompt on ImageNet
# Usage: bash train_flan_t5_xxl_RN50_imagenet_overall_prompt.sh
OMP_NUM_THREADS=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node 8 -m training.main \
    --dataset-type folder \
    --multi-images-per-text \
    --normalize-labels \
    --train-data '../data/ImageNet/train' \
    --imagenet-val '../data/ImageNet/val' \
    --imagenet-single-template '../data/ImageNet/zero_shot' \
    --imagenet-overall-prompt '../data/ImageNet/zero_shot' \
    --text-type imagenet_overall_prompt \
    --text-embeds-path '../data/ImageNet/text_embeds/imagenet_overall_prompt_flan_t5_xxl.pt' \
    --model flan-t5-xxl-RN50 \
    --image-constant-key imagenet \
    --aug-cfg use_timm=True hflip=0.5 \
    --precision amp_bf16 \
    --workers 16 \
    --zeroshot-frequency 1 \
    --save-frequency 32 \
    --epochs 256 \
    --batch-size 128 \
    --warmup 0 \
    --log-every-n-steps 128 \
    --report-to tensorboard