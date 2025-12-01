#!/bin/bash

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

# Original Model
python -m generic_neuromotor_interface.train \
    --config-name=handwriting \
    lightning_module.network.use_set_transformer=false \
    trainer.max_epochs=40 \
    +callbacks.1.save_top_k=-1 \
    +callbacks.1.every_n_epochs=1 \
    data_location=$(pwd)/data \

# SetTransformer Model
# python -m generic_neuromotor_interface.train \
#     --config-name=handwriting \
#     lightning_module.network.use_set_transformer=true \
#     trainer.max_epochs=40 \
#     +callbacks.1.save_top_k=-1 \
#     +callbacks.1.every_n_epochs=1 \
#     data_location=$(pwd)/data \

# /home/ss99569/miniforge3/envs/fun/bin/python ~/bin/notify.py "handwriting settransformer model training done"                                                                                                                                                                                   
/home/ss99569/miniforge3/envs/fun/bin/python ~/bin/notify.py "handwriting original model training done"                                                                                                                                                                                   
