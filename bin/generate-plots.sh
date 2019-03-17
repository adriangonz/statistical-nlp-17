#!/usr/bin/env bash

# $1 refers to the model's state_dict

modelName=$(echo $1 | sed -E 's/^models\/(.+)_model_[0-9]+.pth$/\1/')

echo "Generating figures for model $modelName"
python -m bin.test -v data/vocab.json -m $1 -e data/test.csv

data_path="results/${modelName}_episode.npz"
python -m bin.attention $data_path
python -m bin.embeddings $data_path
