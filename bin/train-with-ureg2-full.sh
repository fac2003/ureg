# Run Evaluate2 on the full training set

python src/org/campagnelab/dl/pytorch/cifar10/Evaluate2.py \
-n 50000 --mini-batch-size 128 --ureg --checkpoint-key DEBUG_UREG1_FULL \
--lr 0.01 --ureg-learning-rate 0.01 --ureg-alpha 1 --ureg-num-features 640 \
--model VGG16 --resume --ureg-reset-every-n-epoch  20 --shaving-epochs 5
# nb. the 5 shaving epochs ensure we use all the training samples (5 times the number of unsupervised/test samples for cifar10)
