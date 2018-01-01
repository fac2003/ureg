# Run evaluate3
export CUDA_VISIBLE_DEVICES=4
python src/org/campagnelab/dl/pytorch/images/Evaluate3.py \
-n 5000 -x 8000 -u 10000 --mini-batch-size 64 --ureg --checkpoint-key STL10_UREG3_10K-mode-combined-96 \
--lr 0.01 --ureg-learning-rate 0.0001 --shave-lr 0.0001 --ureg-alpha 0.5 \
--ureg-num-features 522 --num-epochs 10000 --problem STL10 --mode combined \
--model VGG16 --shaving-epochs 1 --ureg-reset-every-n-epoch 10 \
--max-examples-per-epoch 10000 --resume