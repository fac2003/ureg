# Run evaluate3
CHECKPOINT=STL10_UREG3_100K-mode-combined-VGG16_${RANDOM}
python ${UREG}/src/org/campagnelab/dl/pytorch/cifar10/Evaluate3.py \
-n 1000 -x 4000 -u 100000 --mini-batch-size 64 --ureg \
--checkpoint-key ${CHECKPOINT} \
--problem STL10 --mode combined \
--model VGG16 --shaving-epochs 1 \
 --cross-validations-folds ${UREG}/data/stl10_binary/fold_indices.txt \
--max-examples-per-epoch 10000 --constant-learning-rates "$@" 2>&1 |tee ${CHECKPOINT}.log