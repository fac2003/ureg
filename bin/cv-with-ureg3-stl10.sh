# Run evaluate3
CHECKPOINT=STL10_${RANDOM}
python ${UREG}/src/org/campagnelab/dl/pytorch/cifar10/Evaluate3.py \
-n 5000 -x 8000 -u 100000 --mini-batch-size 50 --ureg \
--checkpoint-key ${CHECKPOINT} \
--problem STL10  \
--model VGG16 --shaving-epochs 1 \
--cv-fold-min-perf 0.57 \
 --cross-validations-folds ${UREG}/data/stl10_binary/fold_indices.txt \
--max-examples-per-epoch 1000  "$@" 2>&1 |tee ${CHECKPOINT}.log