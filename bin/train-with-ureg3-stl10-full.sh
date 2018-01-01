# Run evaluate3
export CUDA_VISIBLE_DEVICES=4
python src/org/campagnelab/dl/pytorch/images/Evaluate3.py \
-n 5000 -x 8000 -u 100000 --mini-batch-size 128 --ureg --checkpoint-key STL10_UREG3_100K-mode-combined \
--lr 0.01 --ureg-learning-rate 0.0001 --shave-lr 0.0001 --ureg-alpha 0.5 \
--ureg-num-features 522 --num-epochs 10000 --problem STL10 --mode combined \
