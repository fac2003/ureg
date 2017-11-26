# Run evaluate3
python src/org/campagnelab/dl/pytorch/cifar10/Evaluate3.py \
-n 5000 -x 8000 -u 100000 --mini-batch-size 256 --ureg \
--checkpoint-key STL10_UREG3_100K-mode-combined-VGG19_${RANDOM} \
--problem STL10 --mode combined \
--model VGG19 --shaving-epochs 1 \
--max-examples-per-epoch 10000 --constant-learning-rates $*