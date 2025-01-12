CUDA_VISIBLE_DEVICES=0 \
    python visualize.py \
    --load_model ./pretrained_models/checkpoint_256x256_v1.pt \
    --config ./config/imagenetgs_256x256_v1.yaml \
    --folder ./images/1 \
    --skip_mesh \
    --output_path ./demos/1 > ./log_training/log_text_nvs.txt 
