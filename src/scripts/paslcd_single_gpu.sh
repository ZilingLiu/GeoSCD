#!/bin/bash

ROOT_DIR=../data/PASLCD
SAVE_BASE=../results/paslcd

########################################
# Run stride 5 / 10 / 15
########################################
for STRIDE in 5 10 15; do
  SAVE_DIR=${SAVE_BASE}/stride${STRIDE}

  CMD="python run_paslcd.py \
    --data-root ${ROOT_DIR} \
    --save-root ${SAVE_DIR} \
    --stride ${STRIDE}"

  echo "=================================================="
  echo "[RUN] ${CMD}"
  echo "=================================================="

  eval ${CMD}
done

echo "All PASLCD runs finished ✅"
