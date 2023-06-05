CUDA_VISIBLE_DEVICES=0 \
    python extract_dift.py \
        --input_path ./assets/cat.png \
        --output_path dift_cat.pt \
        --img_size 0 \
        --t 261 \
        --up_ft_index 1 \
        --prompt 'a photo of a cat' \
        --ensemble_size 8