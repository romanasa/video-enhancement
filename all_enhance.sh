#!/usr/bin/bash

qp=42
raw_path="/Users/roman/vkr/data/raw"
decoded_path="/Users/roman/vkr/data/decoded_qp_$qp"
enhanced_path="/Users/roman/vkr/data/results_$qp"
result_dir="/Users/roman/vkr/data/results_$qp"

for x in $(ls ~/vkr/data/raw); do
  python enhance.py --raw_path $raw_path/$x --decoded_path $decoded_path/$x --enhanced_path $enhanced_path/$x --result_dir $result_dir  --qp $qp
done
