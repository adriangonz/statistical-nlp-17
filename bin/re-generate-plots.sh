#!/usr/bin/env bash

# $1 refers to the model's state_dict

modelName=$(echo $1 | sed -E 's/^models\/(.+)_model_[0-9]+.pth$/\1/')

echo "Generating figures for model $modelName"
python -m bin.test -v data/vocab.json -m $1 -e -a data/test.csv
python -m bin.heatmap results/$modelName_attention.npz
python -m bin.embeddings results/$modelName_embeddings.npz
