# Run evaluate3

python src/org/campagnelab/dl/pytorch/cifar10/Evaluate3.py \
-n 50000 -x 10000 -u 10000 --mini-batch-size 128 --ureg --checkpoint-key UREG3_FULL \
--lr 0.01 --ureg-learning-rate 0.001 --shave-lr 0.001 --ureg-alpha 0.5 \
--ureg-num-features 640 --num-epochs 10000 \
--model VGG16 --ureg-reset-every-n-epoch 30