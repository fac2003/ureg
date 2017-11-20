# Run evaluate with a main model learning rate set to 0.1 and ureg alpha=0.1.

python src/org/campagnelab/dl/pytorch/cifar10/Evaluate2.py \
-n 4000 --mini-batch-size 128 --ureg --checkpoint-key DEBUG \
--lr 0.1 --ureg-learning-rate 0.001 --ureg-alpha 0.5 --ureg-num-features 640 \
--model VGG16