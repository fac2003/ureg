# Run evaluate3
export CUDA_VISIBLE_DEVICES=4
python src/org/campagnelab/dl/pytorch/cifar10/Evaluate3.py \
-n 5000 -x 8000 -u 100000 --mini-batch-size 128 --ureg --checkpoint-key STL10_UREG3_100K \
--lr 0.1 --ureg-learning-rate 0.1 --shave-lr 0.1 --ureg-alpha 0.5 \
--ureg-num-features 522 --num-epochs 10000 --problem STL10 \
--model VGG16 --shaving-epochs 1 --ureg-reset-every-n-epoch 50 \
--max-examples-per-epoch 10000