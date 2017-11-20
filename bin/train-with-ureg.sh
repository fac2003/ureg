# Run evaluate with a main model learning rate set to 0.1 and ureg alpha=0.1.

python src/org/campagnelab/dl/pytorch/cifar10/Evaluate.py \
-n 4000 --mini-batch-size 128 --ureg --checkpoint-key DEBUG \
--lr 0.01 --ureg-learning-rate 0.01 --ureg-alpha 0.5 --ureg-num-features 640 \
--model VGG16 --ureg-reset-every-n-epoch 5