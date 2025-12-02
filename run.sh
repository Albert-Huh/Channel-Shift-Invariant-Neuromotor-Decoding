#!/bin/bash
set -e
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export CUDA_VISIBLE_DEVICES="2"
export CUDA_VISIBLE_DEVICES="3"

# Download data
# python -m generic_neuromotor_interface.scripts.download_data \
#     --task handwriting \
#     --output-dir data/

# Download models
# python -m generic_neuromotor_interface.scripts.download_models \
#     --task handwriting \
#     --output-dir models/

# update if necessary:
CONFIG=handwriting
DATA_DIR="$(pwd)/data" 
EPOCHS=40
STRATEGY=ddp

# 1) BASELINE (no augmentation)
# python -m generic_neuromotor_interface.train \
#     --config-name=${CONFIG} \
#     data_location=${DATA_DIR} \
#     trainer.max_epochs=${EPOCHS} \
#     trainer.devices=${DEVICES} \
#     trainer.strategy=${STRATEGY} \
#     lightning_module.network.use_set_transformer=false \
#     +callbacks.1.save_top_k=-1 \
#     +callbacks.1.every_n_epochs=1 \
#     data_module.channel_aug.train.rotation.enable=false \
#     data_module.channel_aug.train.permutation.enable=false \
#     data_module.channel_aug.val.rotation.enable=false \
#     data_module.channel_aug.val.permutation.enable=false \

# 2) BASELINE (train with rotation augmentation, val on augmented data)
# Note: to use a fixed rotation (e.g., k=4), replace the 3 rotation lines with:
    # data_module.channel_aug.train.rotation.enable=true \
    # data_module.channel_aug.train.rotation.random_k=true \
    # data_module.channel_aug.train.rotation.k_range=[-8,8] \
    # data_module.channel_aug.train.permutation.enable=false \
# python -m generic_neuromotor_interface.train \
#     --config-name=${CONFIG} \
#     data_location=${DATA_DIR} \
#     trainer.max_epochs=${EPOCHS} \
#     trainer.strategy=${STRATEGY} \
#     lightning_module.network.use_set_transformer=false \
#     +callbacks.1.save_top_k=-1 \
#     +callbacks.1.every_n_epochs=1 \
#     data_module.channel_aug.train.rotation.enable=true \
#     data_module.channel_aug.train.rotation.random_k=true \
#     data_module.channel_aug.train.rotation.k_range=[-8,8] \
#     data_module.channel_aug.train.permutation.enable=false \
#     data_module.channel_aug.val.rotation.enable=true \
#     data_module.channel_aug.val.rotation.random_k=true \
#     data_module.channel_aug.val.rotation.k_range=[-8,8] \
#     data_module.channel_aug.val.permutation.enable=false \

# 3) SET-TRANSFORMER (no augmentation)
# python -m generic_neuromotor_interface.train \
#     --config-name=${CONFIG} \
#     data_location=${DATA_DIR} \
#     trainer.max_epochs=${EPOCHS} \
#     trainer.strategy=${STRATEGY} \
#     lightning_module.network.use_set_transformer=true \
#     +callbacks.1.save_top_k=-1 \
#     +callbacks.1.every_n_epochs=1 \
#     data_module.channel_aug.train.rotation.enable=false \
#     data_module.channel_aug.train.permutation.enable=false \
#     data_module.channel_aug.val.rotation.enable=false \
#     data_module.channel_aug.val.permutation.enable=false \

# 4) SET-TRANSFORMER (train with rotation augmentation, val on augmented data)
# python -m generic_neuromotor_interface.train \
#     --config-name=${CONFIG} \
#     data_location=${DATA_DIR} \
#     trainer.max_epochs=${EPOCHS} \
#     trainer.strategy=${STRATEGY} \
#     lightning_module.network.use_set_transformer=true \
#     +callbacks.1.save_top_k=-1 \
#     +callbacks.1.every_n_epochs=1 \
#     data_module.channel_aug.train.rotation.enable=true \
#     data_module.channel_aug.train.rotation.random_k=true \
#     data_module.channel_aug.train.rotation.k_range=[-8,8] \
#     data_module.channel_aug.train.permutation.enable=false \
#     data_module.channel_aug.val.rotation.enable=true \
#     data_module.channel_aug.val.rotation.random_k=true \
#     data_module.channel_aug.val.rotation.k_range=[-8,8] \
#     data_module.channel_aug.val.permutation.enable=false \

# /home/ss99569/miniforge3/envs/fun/bin/python ~/bin/notify.py "handwriting settransformer model training done"                                                                                                                                                                                   
# /home/ss99569/miniforge3/envs/fun/bin/python ~/bin/notify.py "handwriting original model training done"                                                                                                                                                                                   

# Evaluation
python eval_results.py \
    --data_dir ${DATA_DIR} \
    --config_yaml models/handwriting/model_config.yaml \
    --repeats 1 \
    --baseline_noaug_ckpt logs/2025-12-01/12-02-50/lightning_logs/version_0/checkpoints/last.ckpt \
    --baseline_rot_ckpt   logs/2025-12-01/18-48-34/lightning_logs/version_0/checkpoints/last.ckpt \
    --set_noaug_ckpt      logs/2025-12-01/11-42-56/lightning_logs/version_0/checkpoints/last.ckpt \
    --set_rot_ckpt        logs/2025-12-01/18-43-04/lightning_logs/version_0/checkpoints/last.ckpt \
	--baseline_noaug_yaml logs/2025-12-01/12-02-50/hydra_configs/config.yaml \
	--baseline_rot_yaml   logs/2025-12-01/18-48-34/hydra_configs/config.yaml \
	--set_noaug_yaml      logs/2025-12-01/11-42-56/hydra_configs/config.yaml \
	--set_rot_yaml        logs/2025-12-01/18-43-04/hydra_configs/config.yaml
