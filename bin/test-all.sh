#!/usr/bin/env bash

for model in models/*;
do
  echo "Testing model $model..."
  python -m bin.test -v data/vocab.json -m $model -p data/test.csv
done


