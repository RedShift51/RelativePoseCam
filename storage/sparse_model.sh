#!/bin/bash

DATASET_PATH=7
colmap feature_extractor --image_path $DATASET_PATH/images --database_path $DATASET_PATH/database.db
colmap exhaustive_matcher --database_path $DATASET_PATH/database.db

mkdir $DATASET_PATH/sparse
mkdir $DATASET_PATH/model_txt
colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse

python3 /code/read_write_model.py --input_model /storage/$DATASET_PATH/sparse/0 \
	--input_format ".bin" --output_model /storage/$DATASET_PATH/model_txt \
	--output_format ".txt"
