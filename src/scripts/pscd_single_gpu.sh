
#!/bin/bash

ROOT_DIR=../data/PSCD
SAVE_BASE=../results/pscd
GPUS="0"

#!/bin/bash

set -e  # 只要有一步出错，立刻停止

########################################
# STRIDE = 1
########################################

STRIDE=1

echo "=================================================="
echo "[RUN] STRIDE=${STRIDE}"
echo "=================================================="

python unaligned_cd_dir_batch.py \
  --scene-dir ${ROOT_DIR} \
  --save-dir ${SAVE_DIR} \
  --dataset pscd \
  --mode occupy \
  --stride ${STRIDE} \
  --light \
  --bright_thresh 20 \
  --iou_thresh 0.30 \
  --gpus ${GPUS} \
  --sem_filter 0.6


########################################
# STRIDE = 2
########################################

STRIDE=2
SAVE_DIR=${SAVE_BASE}/stride${STRIDE}

echo "=================================================="
echo "[RUN] STRIDE=${STRIDE}"
echo "=================================================="

python unaligned_cd_dir_batch.py \
  --scene-dir ${ROOT_DIR} \
  --save-dir ${SAVE_DIR} \
  --dataset pscd \
  --mode occupy \
  --stride ${STRIDE} \
  --light \
  --bright_thresh 20 \
  --iou_thresh 0.30 \
  --gpus ${GPUS}


########################################
# ALIGN
########################################

SAVE_DIR=${SAVE_BASE}/align

echo "=================================================="
echo "[RUN] ALIGN MODE"
echo "=================================================="

python unaligned_cd_dir_batch.py \
  --scene-dir ${ROOT_DIR} \
  --save-dir ${SAVE_DIR} \
  --dataset pscd \
  --mode occupy \
  --stride 1 \
  --light \
  --bright_thresh 20 \
  --iou_thresh 0.3 \
  --sem_filter 0.6 \
  --gpus ${GPUS} \
  --align

echo "=================================================="
echo "ALL DONE"
echo "=================================================="
