# Description: Overfit on a small subset of ImageNet for debugging purposes
# Usage: bash overfit.sh
OMP_NUM_THREADS=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node 8 -m training.main \
    --dataset-type folder \
    --multi-images-per-text \
    --normalize-labels \
    --train-data '../data/ImageNet/overfit' \
    --imagenet-val '../data/ImageNet/overfit' \
    --text-type imagenet_overall_prompt \
    --text-embeds-path '../data/ImageNet/text_embeds/imagenet_overall_prompt_flan_t5_xxl.pt' \
    --model flan-t5-xxl-TopDownHierConceptualPool2d-RN50 \
    --use-aux-logit-scale \
    --image-constant-key imagenet \
    --aug-cfg use_timm=True hflip=0.5 \
    --precision amp_bf16 \
    --workers 8 \
    --zeroshot-frequency 1 \
    --save-frequency 128 \
    --epochs 256 \
    --batch-size 16 \
    --warmup 0 \
    --log-every-n-steps 1 \