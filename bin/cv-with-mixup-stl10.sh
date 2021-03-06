#!/usr/bin/env bash
# Run evaluate4 on STL10
CHECKPOINT=STL10_${RANDOM}
python ${UREG}/src/org/campagnelab/dl/pytorch/images/Evaluate5.py \
-n 5000 -x 8000 -u 100000 --mini-batch-size 100  \
--max-epochs 1000 \
--checkpoint-key ${CHECKPOINT} \
--problem STL10  \
--model PreActResNet18  \
--two-models \
--cv-fold-min-perf 60 \
--cross-validation-indices 1 \
--cross-validation-folds ${UREG}/data/stl10_binary/fold_indices.txt \
  "$@" 2>&1 |tee ${CHECKPOINT}.log