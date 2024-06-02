## Modified from Torch-Pruning Package (https://github.com/VainF/Torch-Pruning/tree/master)

torchrun --nproc_per_node=4 finetune.py \
    --model "pruned/model_taylor_0.25.pth" \
    --epochs 300 \
    --batch-size 64 \
    --opt adamw \
    --lr 0.000015 \
    --wd 0.3 \
    --lr-scheduler cosineannealinglr \
    --lr-warmup-method linear \
    --lr-warmup-epochs 0 \
    --lr-warmup-decay 0.033 \
    --amp \
    --label-smoothing 0.11 \
    --mixup-alpha 0.2 \
    --auto-augment ra \
    --clip-grad-norm 1 \
    --ra-sampler \
    --random-erase 0.25 \
    --cutmix-alpha 1.0 \
    --data-path "/p/vast1/MLdata/james-imagenet/" \
    --output-dir finetuned_output/ \
    --use_imagenet_mean_std \
    --path "./finetuned_output/model_finetune_taylor_0.0859375" \
