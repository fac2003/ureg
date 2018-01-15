#!/usr/bin/env bash
# Train a capsule model on the complete CIFAR10
CHECKPOINT=CIFAR10_CAPS_${RANDOM}
python ${UREG}/src/org/campagnelab/dl/pytorch/images/Evaluate5.py \
-n 50000 -x 10000 --mini-batch-size 10 --max-epochs 1000 \
--checkpoint-key CAPS_CIFAR_FOCUSED_3_24_8 \
--problem CIFAR10 --model CapsNet3_24_8 --mode capsules \
--lr 1e-5 --L2 1E-9
