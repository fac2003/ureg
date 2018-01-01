# Run evaluate with a main model learning rate set to 0.1 and ureg alpha=0.1.

python src/org/campagnelab/dl/pytorch/images/Evaluate.py \
-n 50000 --mini-batch-size 128 --ureg --checkpoint-key DEBUG_UREG1_FULL \
--lr 0.01 --ureg-learning-rate 0.01 --ureg-alpha 0.5 --ureg-num-features 640 \
--model VGG16