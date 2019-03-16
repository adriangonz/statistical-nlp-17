#!/usr/bin/env bash

for model in models/*;
do
  echo "Testing model $model..."
  python -m bin.test -v data/vocab.json -m $model data/test.csv
done


