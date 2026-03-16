#!/bin/bash

# Execute the following command to run the script:
# chmod +x runner.sh
# ./runner.sh

models=(
"vit_tiny_patch16_224"
"mobilevitv2_100"
"mobilevitv2_125"
"tiny_vit_5m_224"
)

image_sizes=(
    224
    448 
    ) 

for model in "${models[@]}"
do
    for size in "${image_sizes[@]}"
    do
        echo "Running benchmark for $model with image size $size"

        python main.py \
            --model "$model" \
            --image_size "$size" \
            --device "Samsung Galaxy S25 (Family)" \
            --wandb_mode online \
            --wandb_project vit-tinys-benchmark

    done
done