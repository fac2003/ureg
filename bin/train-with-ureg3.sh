# Run evaluate3

python src/org/campagnelab/dl/pytorch/cifar10/Evaluate3.py \
-n 1000 -x 10000 -u 10000 --mini-batch-size 128 --ureg --checkpoint-key DEBUG_UREG3_1000 \
--lr 0.01 --ureg-learning-rate 0.001 --shave-lr 0.01 --ureg-alpha 0.5 \
--ureg-num-features 640 --num-epochs 10000 \
--model VGG16 --shaving-epochs 1 --ureg-reset-every-n-epoch 50