# Run evaluate with a main model learning rate set to 0.1 and ureg alpha=0.1.

python src/org/campagnelab/dl/pytorch/cifar10/Evaluate3.py \
-n 4000 --mini-batch-size 128 --ureg --checkpoint-key DEBUG_UREG2_4000 \
--lr 0.01 --ureg-learning-rate 0.0001 --shave-lr 0.001 --ureg-alpha 0.5 \
--ureg-num-features 640 --num-epochs 10000 \
--model VGG16 --shaving-epochs 1 --ureg-reset-every-n-epoch 30