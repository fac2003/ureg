# Run evaluate with a main model learning rate set to zero. This prevent the main
# model from learning anything about the images, and offers a constant source of
# activations for the ureg model to train.

# In this setting, the ureg model should train to 100% accuracy.
python src/org/campagnelab/dl/pytorch/images/Evaluate.py \
-n 4000 --mini-batch-size 128 --ureg --checkpoint-key DEBUG \
--lr 0.000 --ureg-learning-rate 0.1 --ureg-alpha 0 --ureg-num-features 64