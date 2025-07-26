#!/bin/bash

python compute_features.py \
    --data_dir ../data/forecasting/test/data \
    --feature_dir ./features \
    --mode test \
    --obs_len 20 \
    --pred_len 30 \
    --small \
    --batch_size 5
