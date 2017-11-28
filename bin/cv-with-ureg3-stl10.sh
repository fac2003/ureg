# Run evaluate3
CHECKPOINT=STL10_UREG3_100K-mode-combined-VGG16_${RANDOM}
python ${UREG}/src/org/campagnelab/dl/pytorch/cifar10/Evaluate3.py \
-n 5000 -x 8000 -u 100000 --mini-batch-size 50 --ureg \
--checkpoint-key ${CHECKPOINT} \
--problem STL10  \
--model VGG16 --shaving-epochs 1 \
--ureg-reset-every-n-epoch 10  \
 --cross-validations-folds ${UREG}/data/stl10_binary/fold_indices.txt \
--max-examples-per-epoch 1000 --constant-learning-rates "$@" 2>&1 |tee ${CHECKPOINT}.log