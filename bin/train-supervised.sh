# Run evaluate with a main model learning rate set to 0.1 supervised training only.

python src/org/campagnelab/dl/pytorch/cifar10/Evaluate2.py \
-n 1000 --mini-batch-size 128  --checkpoint-key SUPERVISED_1000 \
--lr 0.1 --model VGG16