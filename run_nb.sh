#!/bin/bash
# Execute the notebook in place with the earth2studio venv kernel.
cd "$(dirname "$0")"
export WORLD_SIZE=1 RANK=0
unset MASTER_ADDR MASTER_PORT
export CUDA_VISIBLE_DEVICES=0
/usr/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=earth2studio-venv \
  --ExecutePreprocessor.timeout=-1 \
  aiwp_stormscope_wyoming.ipynb
