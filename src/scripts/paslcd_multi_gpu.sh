#!/bin/bash

ROOT_DIR=../data/PASLCD
SAVE_BASE=../results/paslcd
GPUS="0,1,2,3,4,5,6,7"

########################################
# Run stride 5 / 10 / 15
########################################
for STRIDE in 10 15; do
  SAVE_DIR=${SAVE_BASE}/stride${STRIDE}

  CMD="python run_paslcd_batch.py \
    --data-root ${ROOT_DIR} \
    --save-root ${SAVE_DIR} \
    --gpus ${GPUS} \
    --stride ${STRIDE}"

  echo "=================================================="
  echo "[RUN] ${CMD}"
  echo "=================================================="

  eval ${CMD}
done


echo "All PASLCD batch runs finished ✅"
