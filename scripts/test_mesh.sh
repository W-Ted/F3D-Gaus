CUDA_VISIBLE_DEVICES=0 \
    python visualize.py \
    --load_model ./pretrained_models/checkpoint_256x256_v1.pt \
    --config ./config/imagenetgs_256x256_v1.yaml \
    --folder ./images/2 \
    --output_path ./demos/2 > ./log_training/log_text_mesh.txt

# aggregation views
# CUDA_VISIBLE_DEVICES=0 \
#     python visualize.py \
#     --load_model ./pretrained_models/checkpoint_256x256_v1.pt \
#     --config ./config/imagenetgs_256x256_v1.yaml \
#     --folder ./images/2 \
#     --aug_mesh \
#     --output_path ./demos/2_aug > ./log_training/log_text_mesh_aug.txt