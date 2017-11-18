# Run evaluate with a main model learning rate set to 0.1 supervised training only.

python src/org/campagnelab/dl/pytorch/cifar10/Evaluate.py \
-n 4000 --mini-batch-size 128  --checkpoint-key SUPERVISED \
--lr 0.1