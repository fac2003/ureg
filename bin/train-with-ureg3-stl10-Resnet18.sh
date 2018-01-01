# Run evaluate3
export CUDA_VISIBLE_DEVICES=4
python src/org/campagnelab/dl/pytorch/images/Evaluate3.py \
-n 5000 -x 8000 -u 10000 --mini-batch-size 64 --ureg --checkpoint-key STL10_UREG3_DPN92_10K-mode-combined \
--lr 5.00e-03 --ureg-learning-rate 1.25e-05 --shave-lr 5.00e-05 --ureg-alpha 0.5 \
--ureg-num-features 522 --num-epochs 10000 --problem STL10 --mode combined \
--model ResNet18 --shaving-epochs 1 --ureg-reset-every-n-epoch 20 \
--max-examples-per-epoch 10000 --constant-learning-rates