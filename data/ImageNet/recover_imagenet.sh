#!/bin/bash

# Move all subdirectories from ./masked_train to ./train
for dir in ./masked_train/*; do
  if [ -d "$dir" ]; then
    mv "$dir" ./train/
  fi
done

# Move all subdirectories from ./zero_shot to ./val
for dir in ./zero_shot/*; do
  if [ -d "$dir" ]; then
    mv "$dir" ./val/
  fi
done

# Count the number of subdirectories in ./train and ./val
count_train=$(find ./train -maxdepth 1 -type d | wc -l)
count_val=$(find ./val -maxdepth 1 -type d | wc -l)

# Subtract 1 from each count to exclude the parent directory itself
count_train=$((count_train-1))
count_val=$((count_val-1))

# Check if the counts are 1000
if [ $count_train -eq 1000 ]; then
  echo "There are 1000 subdirectories in ./train"
else
  echo "There are not 1000 subdirectories in ./train. Actual count: $count_train"
fi

if [ $count_val -eq 1000 ]; then
  echo "There are 1000 subdirectories in ./val"
else
  echo "There are not 1000 subdirectories in ./val. Actual count: $count_val"
fi