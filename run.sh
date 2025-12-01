#!/bin/bash

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="2"

# Download data
# python -m generic_neuromotor_interface.scripts.download_data \
#     --task handwriting \
#     --output-dir data/

# Download models
# python -m generic_neuromotor_interface.scripts.download_models \
#     --task handwriting \
#     --output-dir models/

# Original Model
# python -m generic_neuromotor_interface.train \
#     --config-name=handwriting \
#     data_location=$(pwd)/data \
#     # data_module/data_split=handwriting_mini_split

# SetTransformer Model
python -m generic_neuromotor_interface.train \
    --config-name=handwriting \
    lightning_module.network.use_set_transformer=true \
    data_location=$(pwd)/data \
    # data_module/data_split=handwriting_mini_split
