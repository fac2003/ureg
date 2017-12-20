#!/usr/bin/env bash
# Pre-train a model with unlabeled samples.

python ${UREG}/src/org/campagnelab/dl/pytorch/cifar10/ModelToCpu.py "$@"
