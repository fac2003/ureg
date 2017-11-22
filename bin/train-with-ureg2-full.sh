# Run Evaluate2 on the full training set


python src/org/campagnelab/dl/pytorch/cifar10/Evaluate2.py \
-n 50000 --mini-batch-size 128 --ureg --checkpoint-key UREG2_FULL \
--lr 0.01 --ureg-learning-rate 0.0001 --shave-lr 0.001 --ureg-alpha 1 --ureg-num-features 640 \
--model VGG16 --ureg-reset-every-n-epoch  50 --shaving-epochs 1 --num-epochs 2000
