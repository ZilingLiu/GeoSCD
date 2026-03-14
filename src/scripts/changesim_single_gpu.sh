#!/bin/bash

ROOT_DIR=../data/changesim
SAVE_BASE=../results/changesim
# ../results/changesim_iou65_no_sem_light_25
GPUS="0"

for STRIDE in 5 10 15; do
  SAVE_DIR=${SAVE_BASE}/stride${STRIDE}

  CMD="python run_changesim.py \
    --root-dir ${ROOT_DIR} \
    --save-dir ${SAVE_DIR} \
    --gpus ${GPUS} \
    --stride ${STRIDE} "

  echo "=================================================="
  echo "[RUN] ${CMD}"
  echo "=================================================="

  eval ${CMD}

done

SAVE_DIR=${SAVE_BASE}/aligned

CMD="python run_changesim.py \
  --root-dir ${ROOT_DIR} \
  --save-dir ${SAVE_DIR} \
  --gpus ${GPUS} \
  --stride 5 \
  --align "

echo "=================================================="
echo "[RUN] ${CMD}"
echo "=================================================="

eval ${CMD}


echo "All ChangeSim runs finished ✅"
